#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_post2rw.py
──────────────────────────────────────────────────────────
投稿ベクトル (3072d) からアカウント RW ベクトルを直接回帰する DNN。

  • ストリーミング DataSet (CSV 1 行ずつ読み) なので RAM は一定
  • 損失 = λ * CosineLoss + (1-λ) * MSELoss
  • ―resume でチェックポイント継続学習
----------------------------------------------------------------
"""

import os, csv, argparse, hashlib, random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# ─────────── paths ───────────
VAST_DIR    = "/workspace/edit_agent/vast"
POSTS_CSV   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY    = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT        = os.path.join(VAST_DIR, "post2rw.ckpt")

# ─────────── hyper ───────────
POST_DIM      = 3072
BATCH_SIZE    = 256
EPOCHS        = 100
LR            = 3e-4
WEIGHT_DECAY  = 1e-5
VAL_RATIO     = 0.1          # UID hash split
HIDDEN_DIMS   = [1024, 512]
DROPOUT       = 0.1
LAMBDA_COS    = 0.3          # cos-loss weight (0→MSEのみ, 1→cosのみ)
PATIENCE      = 10
MIN_DELTA     = 1e-4

# ─────────── utils ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l,r = s.find('['),s.rfind(']')
    if 0<=l<r: s = s[l+1:r]
    v = np.fromstring(s.replace(',', ' '), sep=' ', dtype=np.float32)
    return v if v.size == dim else None

def uid_to_val(uid: str, ratio: float=VAL_RATIO):
    h = int(hashlib.md5(uid.encode()).hexdigest(), 16)
    return (h % 10000)/10000 < ratio

# ─────────── dataset ───────────
class Post2RWStream(IterableDataset):
    def __init__(self, split="train", max_posts=50):
        assert split in ("train","val")
        self.split = split
        self.rw = np.load(ACCS_NPY, allow_pickle=True).item()
        self.max_posts = max_posts
    def __iter__(self):
        cnt_uid = 0
        with open(POSTS_CSV, encoding='utf-8') as f:
            rdr=csv.reader(f); next(rdr)
            for uid, _, vec_s in rdr:
                if uid not in self.rw: continue
                if uid_to_val(uid) ^ (self.split=="val"): continue
                v = parse_vec(vec_s, POST_DIM)
                if v is None: continue
                yield torch.from_numpy(v), torch.from_numpy(self.rw[uid].astype(np.float32))

# ─────────── model ───────────
class Post2RW(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=HIDDEN_DIMS, drop=DROPOUT):
        super().__init__()
        layers=[]
        d=in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.ReLU(True), nn.Dropout(drop)]
            d=h
        layers.append(nn.Linear(d,out_dim))
        self.f = nn.Sequential(*layers)
    def forward(self,x): return self.f(x)

# ─────────── train loop ───────────
def train(resume=False):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[Device]", dev)

    tr_ds, va_ds = Post2RWStream("train"), Post2RWStream("val")
    dl_tr = DataLoader(tr_ds, batch_size=BATCH_SIZE)
    dl_va = DataLoader(va_ds, batch_size=BATCH_SIZE)

    rw_dim = next(iter(tr_ds.rw.values())).shape[0]
    model = Post2RW(POST_DIM, rw_dim).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    cos = nn.CosineEmbeddingLoss()

    best, wait, start = float("inf"), 0, 1
    if resume and os.path.isfile(CKPT):
        ck=torch.load(CKPT,map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best,wait,start = ck["best"], ck["wait"], ck["epoch"]+1
        print(f"[Resume] epoch={start-1} best={best:.4f}")

    for ep in range(start,EPOCHS+1):
        # ---- train ----
        model.train(); tl=n=0
        for x, y in tqdm(dl_tr, desc=f"Ep{ep}[train]", unit="batch", leave=False):
            x,y=x.to(dev),y.to(dev)
            y_hat=model(x)
            loss = (1-LAMBDA_COS)*mse(y_hat,y) + LAMBDA_COS*cos(y_hat,y,torch.ones(x.size(0),device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*x.size(0); n+=x.size(0)
        tl/=n
        # ---- val ----
        model.eval(); vl=n=0
        with torch.no_grad():
            for x,y in dl_va:
                x,y=x.to(dev),y.to(dev)
                y_hat=model(x)
                loss=(1-LAMBDA_COS)*mse(y_hat,y)+LAMBDA_COS*cos(y_hat,y,torch.ones(x.size(0),device=dev))
                vl+=loss.item()*x.size(0); n+=x.size(0)
        vl/=n
        print(f"Ep{ep:03d} train={tl:.4f} val={vl:.4f}")
        if vl<best-MIN_DELTA:
            best,wait=vl,0
            torch.save({"epoch":ep,"model":model.state_dict(),
                        "opt":opt.state_dict(),"best":best,"wait":wait}, CKPT)
            print("  ✔ checkpoint saved")
        else:
            wait+=1
            if wait>=PATIENCE: print("Early stopping"); break
    print(f"[Done] best_val={best:.4f}")

# ─────────── entry ───────────
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--resume",action="store_true")
    train(p.parse_args().resume)
