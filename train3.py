#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pair_classifier_masked.py
generate_dcor_mask.py で作ったビットマスクを読み込み、
mask==1 の投稿だけをデータセットに流し 0/1 BCE 学習する版。
ノイズ負例・ランダム負例・--resume など従来仕様と同じ。
"""

import os, csv, random, argparse, hashlib, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from collections import defaultdict

# ─────────── paths & params ───────────
VAST_DIR       = "/workspace/edit_agent/vast"
POSTS_CSV      = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY    = os.path.join(VAST_DIR, "account_vectors.npy")
MASK_DIR       = os.path.join(VAST_DIR, "dcor_masks")   # ← ① の出力先
CKPT_PATH      = os.path.join(VAST_DIR, "pair_classifier_masked.ckpt")

POST_DIM     = 3072
BATCH_SIZE   = 128
EPOCHS       = 500
LR           = 1e-4
WEIGHT_DECAY = 1e-5
NEG_RATIO    = 5
VAL_RATIO    = 0.1
DROPOUT_RATE = 0.1
NOISE_STD    = 0.2
PATIENCE     = 15
MIN_DELTA    = 1e-4

# ───── util ─────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'): s = s[1:-1]
    l,r = s.find('['), s.rfind(']')
    if 0<=l<r: s=s[l+1:r]
    v = np.fromstring(s.replace(',',' '), dtype=np.float32, sep=' ')
    return v if v.size==dim else None

def uid_to_val(uid, ratio):
    import hashlib
    h=int(hashlib.md5(uid.encode()).hexdigest(),16)
    return (h%10000)/10000<ratio

def l2_norm(v):
    n=np.linalg.norm(v); return v/n if n else v

# ───── dataset ─────
class MaskedPairStream(IterableDataset):
    def __init__(self, split="train"):
        self.acc_dict = np.load(ACCOUNT_NPY, allow_pickle=True).item()
        self.uids     = list(self.acc_dict.keys())
        self.split    = split
        self.rw_dim   = next(iter(self.acc_dict.values())).shape[0]
        # マスクを読み込んでメモリに保持（数 MB）
        self.masks = {}
        for uid in self.uids:
            mpath = os.path.join(MASK_DIR, f"{uid}.npy")
            if os.path.exists(mpath):
                self.masks[uid] = np.load(mpath)
            else:
                self.masks[uid] = None   # mask 無い UID は全部 skip
        print(f"[Dataset] loaded masks for {sum(m is not None for m in self.masks.values())} UID")

    def __iter__(self):
        idx_counter = defaultdict(int)   # uid -> local idx
        with open(POSTS_CSV, encoding="utf-8") as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_str in rdr:
                if uid not in self.acc_dict: continue
                if uid_to_val(uid, VAL_RATIO) ^ (self.split=="val"): continue

                mask = self.masks.get(uid)
                j = idx_counter[uid]; idx_counter[uid]+=1
                # ---- mask check ----
                if mask is None or j>=len(mask) or mask[j]==0:     # 採用外
                    continue

                post_np = parse_vec(vec_str, POST_DIM)
                if post_np is None: continue
                post_t  = torch.from_numpy(post_np)
                acc_np  = self.acc_dict[uid].astype(np.float32)
                acc_t   = torch.from_numpy(acc_np)

                # 正例
                yield post_t, acc_t, torch.tensor(1.)

                # ノイズ負例
                noise = np.random.normal(0, NOISE_STD, size=acc_np.shape).astype(np.float32)
                acc_noise = torch.from_numpy(l2_norm(acc_np+noise))
                yield post_t, acc_noise, torch.tensor(0.)

                # ランダム負例
                for _ in range(NEG_RATIO):
                    neg_uid = random.choice(self.uids)
                    while neg_uid==uid: neg_uid = random.choice(self.uids)
                    acc_neg = torch.from_numpy(self.acc_dict[neg_uid].astype(np.float32))
                    yield post_t, acc_neg, torch.tensor(0.)

# ───── model ─────
class PairClassifier(nn.Module):
    def __init__(self, post_dim, rw_dim, hidden=512, drop=DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim+rw_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )
    def forward(self, p, a):
        return self.net(torch.cat([p,a],1)).squeeze(1)

# ───── train loop ─────
def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", dev)

    tr_ds = MaskedPairStream("train")
    va_ds = MaskedPairStream("val")
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)

    model = PairClassifier(POST_DIM, tr_ds.rw_dim).to(dev)
    optim_ = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit   = nn.BCEWithLogitsLoss()

    start=1; best=float("inf"); wait=0
    if args.resume and os.path.exists(CKPT_PATH):
        ck=torch.load(CKPT_PATH,map_location=dev)
        model.load_state_dict(ck["model_state"])
        optim_.load_state_dict(ck["optim_state"])
        start, best, wait = ck["epoch"]+1, ck["best_val"], ck["patience"]
        print(f"[Resume] epoch={ck['epoch']} best={best:.4f}")

    for ep in range(start, EPOCHS+1):
        # train
        model.train(); tr_loss=n_tr=0
        for p,a,lbl in tr_loader:
            p,a,lbl=p.to(dev),a.to(dev),lbl.to(dev)
            loss = crit(model(p,a), lbl)
            optim_.zero_grad(); loss.backward(); optim_.step()
            tr_loss+=loss.item()*p.size(0); n_tr+=p.size(0)
        tr_loss/=n_tr

        # val
        model.eval(); va_loss=n_va=0
        with torch.no_grad():
            for p,a,lbl in va_loader:
                p,a,lbl=p.to(dev),a.to(dev),lbl.to(dev)
                loss=crit(model(p,a),lbl)
                va_loss+=loss.item()*p.size(0); n_va+=p.size(0)
        va_loss/=n_va
        print(f"Ep{ep:03d} train={tr_loss:.4f} val={va_loss:.4f}")

        if va_loss<best-MIN_DELTA:
            best,wait=va_loss,0
            torch.save({"epoch":ep,"model_state":model.state_dict(),
                        "optim_state":optim_.state_dict(),
                        "best_val":best,"patience":wait}, CKPT_PATH)
            print("  ✔ checkpoint saved")
        else:
            wait+=1
            if wait>=PATIENCE:
                print("Early stopping."); break
    print(f"[Done] best_val={best:.4f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--resume",action="store_true")
    train(ap.parse_args())
