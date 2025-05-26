#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_avg2rw.py ― 投稿ベクトルを単純平均 → アカウントRWベクトル回帰モデル

* 各 UID ごとに最新 N 投稿を平均した 3072 次元ベクトルを作成
* train/val を UID 単位で 90/10 split
* 2 層 MLP (3072→512→rw_dim) で回帰 (MSELoss)
* --resume でチェックポイント継続学習
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

# ─────────── Paths & Hyperparams ───────────
BASE         = "/workspace/edit_agent"
VAST_DIR     = os.path.join(BASE, "vast")
POSTS_CSV    = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH    = os.path.join(VAST_DIR, "avg2rw.ckpt")
CACHE_PATH   = os.path.join(VAST_DIR, "avg_posts_cache.pkl")

POST_DIM       = 3072      # 投稿ベクトル次元
POSTS_PER_UID  = 30        # 平均に使う最新投稿数
HIDDEN_DIM     = 512       # MLP 中間層サイズ
BATCH_SIZE     = 128
EPOCHS         = 100
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
VAL_RATIO      = 0.1
PATIENCE       = 10
MIN_DELTA      = 1e-4
DROPOUT        = 0.3

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
class AvgPostsDataset(Dataset):
    def __init__(self, uids, avg_dict, acc_dict):
        self.samples = []
        for uid in uids:
            if uid in avg_dict and uid in acc_dict:
                self.samples.append((
                    avg_dict[uid],               # np.ndarray (3072,)
                    acc_dict[uid].astype(np.float32)  # np.ndarray (rw_dim,)
                ))
        if not self.samples:
            raise RuntimeError("No samples for given UIDs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

# ─────────── Model ───────────
class Avg2RW(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# ─────────── Main ───────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) Load or build avg post vectors per UID
    if os.path.exists(CACHE_PATH) and not args.force_recache:
        with open(CACHE_PATH, "rb") as f:
            avg_posts = pickle.load(f)
        print(f"[Cache] loaded {len(avg_posts)} uid averages")
    else:
        acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
        buf = {}  # uid -> list of post vectors
        with open(POSTS_CSV, encoding="utf-8") as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_s in rdr:
                if uid not in acc_dict:
                    continue
                v = parse_vec(vec_s, POST_DIM)
                if v is None:
                    continue
                lst = buf.setdefault(uid, [])
                lst.append(v)
                if len(lst) > POSTS_PER_UID:
                    lst.pop(0)
        avg_posts = {
            uid: np.stack(vs,0).mean(axis=0)
            for uid, vs in buf.items() if vs
        }
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(avg_posts, f)
        print(f"[Cache] saved {len(avg_posts)} uid averages")

    # 2) Prepare train/val UID split
    all_uids = list(avg_posts.keys())
    random.shuffle(all_uids)
    n_val = int(len(all_uids)*VAL_RATIO)
    val_uids = all_uids[:n_val]
    tr_uids  = all_uids[n_val:]
    print(f"[Split] train: {len(tr_uids)}  val: {len(val_uids)}")

    # 3) Datasets & DataLoaders
    acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
    tr_ds = AvgPostsDataset(tr_uids, avg_posts, acc_dict)
    va_ds = AvgPostsDataset(val_uids, avg_posts, acc_dict)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4) Model / Optimizer / Loss
    rw_dim = next(iter(acc_dict.values())).shape[0]
    model = Avg2RW(POST_DIM, rw_dim).to(device)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

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
        model.train()
        sum_tr, n_tr = 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            sum_tr += loss.item()*x.size(0); n_tr += x.size(0)
        tr_loss = sum_tr / n_tr

        model.eval()
        sum_va, n_va = 0.0, 0
        with torch.no_grad():
            for x, y in va_ld:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                sum_va += criterion(pred, y).item()*x.size(0)
                n_va += x.size(0)
        va_loss = sum_va / n_va

        print(f"Ep{ep:03d} train={tr_loss:.4f} val={va_loss:.4f}")

        # checkpoint & early stop
        if va_loss < best_val - MIN_DELTA:
            best_val, patience = va_loss, 0
            torch.save({
                "epoch":    ep,
                "model":    model.state_dict(),
                "opt":      opt.state_dict(),
                "best_val": best_val,
                "patience": patience
            }, CKPT_PATH)
            print(f"  ✔ saved {CKPT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true",
                   help="resume from checkpoint")
    p.add_argument("--force-recache", action="store_true",
                   help="rebuild average cache")
    args = p.parse_args()
    train(args)