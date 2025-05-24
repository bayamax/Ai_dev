#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pair_cosine_hardneg.py
CosineEmbeddingLoss + Hard-Negative Mining 版ストリーミング学習
"""

import os, csv, random, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# ───────── paths & hparams ─────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH       = os.path.join(VAST_DIR, "pair_cosine_rw_stream.ckpt")

POST_DIM     = 3072
BATCH_SIZE   = 128
EPOCHS       = 500
LR           = 1e-4
WEIGHT_DECAY = 1e-5
NEG_RATIO    = 4          # ランダム他人＋ノイズ負例で計4本
VAL_RATIO    = 0.1
DROPOUT_RATE = 0.1
NOISE_STD    = 0.2        # ノイズ負例の標準偏差
MARGIN       = 0.2
PATIENCE     = 15
MIN_DELTA    = 1e-4

# ───────── util ─────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'): s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r: s = s[l+1:r]
    vec = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return vec if vec.size == dim else None

def uid_to_val(uid, ratio):
    h = int(hashlib.md5(uid.encode()).hexdigest(), 16)
    return (h % 10000) / 10000 < ratio

def l2_norm(a):
    n = np.linalg.norm(a)
    return a / n if n else a

# ───────── dataset ─────────
class PairStreamDataset(IterableDataset):
    def __init__(self, posts_csv, acc_npy, split="train"):
        self.posts_csv = posts_csv
        self.acc_dict  = np.load(acc_npy, allow_pickle=True).item()
        self.uids      = list(self.acc_dict.keys())
        self.split     = split
        self.rw_dim    = next(iter(self.acc_dict.values())).shape[0]

    def __iter__(self):
        np.random.seed()  # each epoch fresh
        with open(self.posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_str in rdr:
                if uid not in self.acc_dict: continue
                if uid_to_val(uid, VAL_RATIO) ^ (self.split == "val"): continue

                post_np = parse_vec(vec_str, POST_DIM)
                if post_np is None: continue
                post_t = torch.from_numpy(post_np)

                # 正例 (+1)
                acc_pos = torch.from_numpy(self.acc_dict[uid])
                yield post_t, acc_pos, torch.tensor(1.)

                # ノイズ負例 (1 本)
                noise = np.random.normal(0, NOISE_STD, size=acc_pos.shape).astype(np.float32)
                acc_noise = torch.from_numpy(l2_norm(self.acc_dict[uid] + noise))
                yield post_t, acc_noise, torch.tensor(-1.)

                # ランダム他人負例
                for _ in range(NEG_RATIO - 1):
                    neg_uid = random.choice(self.uids)
                    while neg_uid == uid:
                        neg_uid = random.choice(self.uids)
                    acc_neg = torch.from_numpy(self.acc_dict[neg_uid])
                    yield post_t, acc_neg, torch.tensor(-1.)

# ───────── model ─────────
class CosineMapper(nn.Module):
    def __init__(self, post_dim, rw_dim, hid=512, drop=DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim, hid), nn.ReLU(True),
            nn.Dropout(drop), nn.Linear(hid, rw_dim)
        )
    def forward(self, x): return self.net(x)

# ───────── train ─────────
def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", dev)

    tr_ds = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, "train")
    va_ds = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, "val")
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)

    model = CosineMapper(POST_DIM, tr_ds.rw_dim).to(dev)
    optim_ = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(optim_, "min", factor=0.3, patience=3)
    crit   = nn.CosineEmbeddingLoss(margin=MARGIN)

    start, best, wait = 1, float("inf"), 0
    if args.resume and os.path.isfile(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=dev)
        model.load_state_dict(ck["model_state"])
        optim_.load_state_dict(ck["optim_state"])
        start, best, wait = ck["epoch"] + 1, ck["best_val"], ck["patience"]
        print(f"[Resume] epoch={ck['epoch']} best={best:.4f}")

    for ep in range(start, EPOCHS + 1):
        # ---- train ----
        model.train(); tr_loss = n_tr = 0
        for post, acc, tgt in tr_loader:
            post, acc, tgt = post.to(dev), acc.to(dev), tgt.to(dev)
            post_emb = nn.functional.normalize(model(post))
            acc_emb  = nn.functional.normalize(acc)
            # In-batch hard negatives
            B = post_emb.size(0)
            p_rep = post_emb.unsqueeze(1).expand(-1, B, -1).reshape(-1, post_emb.size(1))
            a_rep = acc_emb.repeat(B, 1)
            t_rep = torch.where(
                torch.arange(B*B, device=dev).view(B, B).diag().bool(),
                torch.ones(1, device=dev),
                -torch.ones(1, device=dev)
            ).view(-1)
            loss = crit(p_rep, a_rep, t_rep)
            optim_.zero_grad(); loss.backward(); optim_.step()
            tr_loss += loss.item() * B; n_tr += B
        tr_loss /= n_tr

        # ---- val ----
        model.eval(); va_loss = n_va = 0
        with torch.no_grad():
            for post, acc, tgt in va_loader:
                post, acc, tgt = post.to(dev), acc.to(dev), tgt.to(dev)
                p = nn.functional.normalize(model(post))
                a = nn.functional.normalize(acc)
                loss = crit(p, a, tgt)
                va_loss += loss.item() * post.size(0); n_va += post.size(0)
        va_loss /= n_va; sched.step(va_loss)
        print(f"Ep{ep:03d} train={tr_loss:.4f} val={va_loss:.4f}")

        # ---- ckpt / early stop ----
        if va_loss < best - MIN_DELTA:
            best, wait = va_loss, 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optim_state": optim_.state_dict(),
                        "best_val": best, "patience": wait}, CKPT_PATH)
            print("  ✔ checkpoint saved")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping."); break
    print(f"[Done] best_val={best:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    train(ap.parse_args())