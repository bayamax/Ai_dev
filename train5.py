#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_contrastive.py ― CompactAttnAggregator を用いた InfoNCE コントラスト学習スクリプト

* 投稿集合 (最新 N 投稿) → CompactAttnAggregator → rw_dim 次元ベクトル
* InfoNCE loss：バッチ内の他サンプルをネガティブとみなして学習
* train/val を UID 単位で 90/10 split
* --resume でチェックポイント継続学習対応
"""

import os
import csv
import argparse
import random
import pickle

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
CKPT_PATH    = os.path.join(VAST_DIR, "contrastive.ckpt")
CACHE_PATH   = os.path.join(VAST_DIR, "contrastive_posts_cache.pkl")

POST_DIM       = 3072      # 投稿ベクトル次元
POSTS_PER_UID  = 30        # 平均に使う最新投稿数
D_MODEL        = 128       # CompactAttnAggregator の中間次元
RW_DIM         = None      # 後でロードから設定
N_HEADS        = 4
N_LAYERS       = 2
DROPOUT        = 0.3

BATCH_SIZE     = 128
EPOCHS         = 100
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
TEMPERATURE    = 0.1      # InfoNCE 温度
VAL_RATIO      = 0.1
PATIENCE       = 15
MIN_DELTA      = 1e-4

# ─────────── Utility ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

# ─────────── Dataset ───────────
class PostsToRW(Dataset):
    def __init__(self, uids, avg_dict):
        """
        uids: list of UID for this split
        avg_dict: {uid: np.ndarray(posts_per_uid, POST_DIM)}
        """
        self.samples = []
        for uid in uids:
            posts = avg_dict.get(uid)
            if posts is not None:
                self.samples.append((posts, uid))
        if not self.samples:
            raise RuntimeError("No samples for given UIDs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        posts, uid = self.samples[idx]
        return posts, uid

def collate_fn(batch):
    posts_list, uid_list = zip(*batch)
    B = len(posts_list)
    lengths = [p.shape[0] for p in posts_list]
    max_len = max(lengths)
    padded = torch.zeros(B, max_len, POST_DIM, dtype=torch.float32)
    mask   = torch.ones(B, max_len, dtype=torch.bool)
    for i, p in enumerate(posts_list):
        L = p.shape[0]
        padded[i, :L] = torch.from_numpy(p)
        mask[i, :L] = False
    return padded, mask, uid_list

# ─────────── Model ───────────
class CompactAttnAggregator(nn.Module):
    def __init__(self, d_in=POST_DIM, d_model=D_MODEL,
                 rw_dim=128, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1,1,d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, rw_dim)

    def forward(self, x, mask):
        B, S, _ = x.size()
        h = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        if mask is not None:
            cls_mask = torch.zeros(B,1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        h_enc = self.encoder(h, src_key_padding_mask=mask)
        cls_out = h_enc[:,0]
        return self.output_head(cls_out)

# ─────────── Main ───────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) Load or build posts per UID cache
    if os.path.exists(CACHE_PATH) and not args.force_recache:
        with open(CACHE_PATH, "rb") as f:
            uid_posts = pickle.load(f)
        print(f"[Cache] loaded posts for {len(uid_posts)} UIDs")
    else:
        acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
        uid_posts = {}
        # count lines
        total = sum(1 for _ in open(POSTS_CSV, encoding="utf-8")) - 1
        print("[Cache] building posts cache...")
        with open(POSTS_CSV, encoding="utf-8") as f:
            reader = csv.reader(f); next(reader)
            for uid, _, vec_s in tqdm(reader, total=total, desc="Reading CSV"):
                if uid not in acc_dict:
                    continue
                v = parse_vec(vec_s, POST_DIM)
                if v is None:
                    continue
                lst = uid_posts.setdefault(uid, [])
                lst.append(v)
                if len(lst) > POSTS_PER_UID:
                    lst.pop(0)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(uid_posts, f)
        print(f"[Cache] saved posts for {len(uid_posts)} UIDs")

    # 2) train/val split
    all_uids = list(uid_posts.keys())
    random.shuffle(all_uids)
    n_val = int(len(all_uids)*VAL_RATIO)
    val_uids = all_uids[:n_val]
    tr_uids  = all_uids[n_val:]
    print(f"[Split] train: {len(tr_uids)}  val: {len(val_uids)}")

    # 3) DataLoaders
    tr_ds = PostsToRW(tr_uids, uid_posts)
    va_ds = PostsToRW(val_uids, uid_posts)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn)

    # 4) Model & optimizer
    acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
    global RW_DIM
    RW_DIM = next(iter(acc_dict.values())).shape[0]
    model = CompactAttnAggregator(rw_dim=RW_DIM).to(device)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    start_ep, best_val, patience = 1, float("inf"), 0
    if args.resume and os.path.isfile(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"] + 1
        best_val = ck["best_val"]
        patience = ck["patience"]
        print(f"[Resume] from ep {start_ep-1}, best_val {best_val:.4f}")

    # 5) Training loop
    for ep in range(start_ep, EPOCHS+1):
        print(f"\n=== Epoch {ep}/{EPOCHS} ===")
        model.train()
        sum_tr, n_tr = 0.0, 0
        for posts, mask, uids in tqdm(tr_ld, desc="Train batches"):
            posts, mask = posts.to(device), mask.to(device)
            reps = model(posts, mask)                           # (B, RW_DIM)
            reps_norm = nn.functional.normalize(reps, dim=1)    # 正規化
            # build target matrix
            # for in-batch positives, match identical UID indices
            # get the true account vectors for uids
            true = torch.stack([torch.from_numpy(acc_dict[u])
                                for u in uids], dim=0).to(device)
            true_norm = nn.functional.normalize(true, dim=1)
            logits = reps_norm @ true_norm.T / TEMPERATURE      # (B, B)
            labels = torch.arange(logits.size(0), device=device)
            loss = nn.functional.cross_entropy(logits, labels)

            opt.zero_grad(); loss.backward(); opt.step()
            sum_tr += loss.item() * reps.size(0)
            n_tr   += reps.size(0)
        tr_loss = sum_tr / n_tr

        model.eval()
        sum_va, n_va = 0.0, 0
        with torch.no_grad():
            for posts, mask, uids in tqdm(va_ld, desc="Val batches"):
                posts, mask = posts.to(device), mask.to(device)
                reps = model(posts, mask)
                reps_norm = nn.functional.normalize(reps, dim=1)
                true = torch.stack([torch.from_numpy(acc_dict[u])
                                    for u in uids], dim=0).to(device)
                true_norm = nn.functional.normalize(true, dim=1)
                logits = reps_norm @ true_norm.T / TEMPERATURE
                labels = torch.arange(logits.size(0), device=device)
                loss = nn.functional.cross_entropy(logits, labels)
                sum_va += loss.item() * reps.size(0)
                n_va   += reps.size(0)
        va_loss = sum_va / n_va

        print(f"Epoch {ep}/{EPOCHS}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

        if va_loss < best_val - MIN_DELTA:
            best_val, patience = va_loss, 0
            torch.save({
                "epoch":    ep,
                "model":    model.state_dict(),
                "opt":      opt.state_dict(),
                "best_val": best_val,
                "patience": patience
            }, CKPT_PATH)
            print(f"  ✔ saved checkpoint to {CKPT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\n[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true",
                   help="resume from checkpoint")
    p.add_argument("--force-recache", action="store_true",
                   help="rebuild posts cache")
    args = p.parse_args()
    train(args)