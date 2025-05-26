#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_compact_attn.py ― 投稿集合→アカウントベクトル回帰
次元圧縮＋Transformer(CLSプーリング)を１ステップで実装した
CompactAttnAggregator を用いた学習スクリプト。

* UID ごとに最新 N 投稿を (3072d) ベクトルとして収集
* UID 単位で 90/10 split → train / val
* CompactAttnAggregator (3072→d_model→CLSプーリング→128d)
* MSE + CosineEmbeddingLoss で rw_dim(128) 回帰
* --resume で refine_rw_agg.ckpt から続き学習
"""

import os
import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─────────── Paths & Hyperparams ───────────
BASE         = "/workspace/edit_agent"
VAST_DIR     = os.path.join(BASE, "vast")
POSTS_CSV    = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH    = os.path.join(VAST_DIR, "compact_attn.ckpt")

POST_DIM     = 3072
POSTS_PER_UID= 30
BATCH_SIZE   = 64
EPOCHS       = 500
LR           = 1e-4
WD           = 1e-5
VAL_RATIO    = 0.1
D_MODEL      = 128      # 中間次元を小さく絞る
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.1
LAMBDA_COS   = 0.3     # cosine 対 MSE
PATIENCE     = 15
MIN_DELTA    = 1e-4

# ─────────── Utils ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

# ─────────── Dataset ───────────
class PostsToRW(Dataset):
    def __init__(self, uid_list):
        # load account vectors
        self.acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
        # collect posts per uid
        user_posts = {uid: [] for uid in uid_list}
        with open(POSTS_CSV, encoding="utf-8") as f:
            rdr = csv.reader(f)
            next(rdr)
            for uid, _, vec_s in tqdm(rdr, desc="Collect posts", unit="line"):
                if uid not in user_posts: continue
                v = parse_vec(vec_s, POST_DIM)
                if v is None: continue
                user_posts[uid].append(v)
                if len(user_posts[uid]) > POSTS_PER_UID:
                    user_posts[uid].pop(0)
        # build samples
        self.samples = []
        for uid, vecs in user_posts.items():
            if not vecs: continue
            posts = np.stack(vecs, axis=0)                            # (S, POST_DIM)
            target = self.acc_dict[uid].astype(np.float32)           # (rw_dim,)
            self.samples.append((posts, target))
        if not self.samples:
            raise RuntimeError("No samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        posts, target = self.samples[idx]
        return posts, target

def collate_fn(batch):
    posts_list, targets_list = zip(*batch)
    lengths = [p.shape[0] for p in posts_list]
    max_len = max(lengths)
    B = len(posts_list)
    padded = torch.zeros(B, max_len, POST_DIM, dtype=torch.float32)
    mask   = torch.ones(B, max_len, dtype=torch.bool)
    for i, p in enumerate(posts_list):
        L = p.shape[0]
        padded[i, :L] = torch.from_numpy(p)
        mask[i, :L] = False
    targets = torch.stack([torch.from_numpy(t) for t in targets_list], dim=0)
    return padded, mask, targets

# ─────────── Model ───────────
class CompactAttnAggregator(nn.Module):
    def __init__(self, d_in=POST_DIM, d_out=128,
                 d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1,1,d_model))
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, d_out)

    def forward(self, x, mask):
        B, S, _ = x.size()
        h = self.input_proj(x)                           # (B,S,d_model)
        cls = self.cls_token.expand(B, -1, -1)            # (B,1,d_model)
        h   = torch.cat([cls, h], dim=1)                  # (B,S+1,d_model)
        if mask is not None:
            cls_mask = torch.zeros(B,1,device=mask.device, dtype=torch.bool)
            mask = torch.cat([cls_mask, mask], dim=1)     # (B,S+1)
        h_enc = self.encoder(h, src_key_padding_mask=mask)  # (B,S+1,d_model)
        cls_out = h_enc[:,0]                              # (B,d_model)
        return self.output_head(cls_out)                  # (B,d_out)

# ─────────── Training Loop ───────────
def train(resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # load uids and split
    acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
    all_uids = list(acc_dict.keys())
    random.shuffle(all_uids)
    n_val = int(len(all_uids) * VAL_RATIO)
    val_uids = set(all_uids[:n_val])
    tr_uids  = set(all_uids[n_val:])

    # datasets & loaders
    tr_ds = PostsToRW(tr_uids)
    va_ds = PostsToRW(val_uids)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE,
                       shuffle=False, collate_fn=collate_fn)

    # model & optimizer
    rw_dim = next(iter(acc_dict.values())).shape[0]
    model = CompactAttnAggregator(d_in=POST_DIM,
                                  d_out=rw_dim).to(device)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    mse   = nn.MSELoss()
    cos   = nn.CosineEmbeddingLoss()

    start_ep, best_val, patience = 1, float("inf"), 0
    if resume and os.path.isfile(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"] + 1
        best_val = ck["best_val"]
        patience = ck["patience"]
        print(f"[Resume] epoch={start_ep-1} best={best_val:.4f} pat={patience}")

    for ep in range(start_ep, EPOCHS+1):
        # train
        model.train()
        total_tr, n_tr = 0.0, 0
        for posts, mask, targets in tqdm(tr_ld, desc=f"Ep{ep} [train]"):
            posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
            preds = model(posts, mask)
            y = targets
            loss = (1-LAMBDA_COS)*mse(preds, y) + LAMBDA_COS*cos(preds, y, torch.ones(len(y),device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            total_tr += loss.item()*y.size(0); n_tr += y.size(0)
        tr_loss = total_tr / n_tr

        # val
        model.eval()
        total_va, n_va = 0.0, 0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_ld, desc=f"Ep{ep} [val]"):
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
                preds = model(posts, mask)
                y = targets
                loss = (1-LAMBDA_COS)*mse(preds, y) + LAMBDA_COS*cos(preds, y, torch.ones(len(y),device=device))
                total_va += loss.item()*y.size(0); n_va += y.size(0)
        va_loss = total_va / n_va

        print(f"Ep{ep}/{EPOCHS} train={tr_loss:.4f} val={va_loss:.4f}")

        # checkpoint & early stopping
        if va_loss < best_val - MIN_DELTA:
            best_val, patience = va_loss, 0
            torch.save({
                "epoch":     ep,
                "model":     model.state_dict(),
                "opt":       opt.state_dict(),
                "best_val":  best_val,
                "patience":  patience
            }, CKPT_PATH)
            print(f"  ✔ saved {CKPT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val = {best_val:.4f}")

# ─────────── Entry ───────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true", help="resume from checkpoint")
    args = p.parse_args()
    train(args.resume)