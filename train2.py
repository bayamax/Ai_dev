#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pair_classifier.py ― 投稿ベクトルとアカウントベクトルのペアを
正例(同一アカウント)=1／負例(異なるアカウント)=0 で学習するスクリプト

★ --resume を指定すると前回のチェックポイントから重み、オプティマイザ状態、
  早期停止情報を復元して続きから学習します。
"""

import os
import sys
import csv
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─────────── ハードコード済みパス ───────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CHECKPOINT_PATH = os.path.join(VAST_DIR, "pair_classifier_rw.ckpt")

# ──────────── ハイパラ ────────────
POST_DIM     = 3072
BATCH_SIZE   = 128
EPOCHS       = 500
LR           = 1e-4
WEIGHT_DECAY = 1e-5
NEG_RATIO    = 5   # 正例1に対して負例1つ
PATIENCE     = 15   # 早期停止 patience
MIN_DELTA    = 1e-4

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

class PairDataset(Dataset):
    def __init__(self, posts_csv, account_npy, neg_ratio=1):
        self.rw_dict = np.load(account_npy, allow_pickle=True).item()
        self.uids = list(self.rw_dict.keys())
        self.posts = []
        with open(posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_str in rdr:
                if uid not in self.rw_dict: continue
                vec = parse_vec(vec_str, POST_DIM)
                if vec is not None:
                    self.posts.append((uid, vec))
        if not self.posts:
            sys.exit("ERROR: No valid post vectors found.")
        self.neg_ratio = neg_ratio

    def __len__(self):
        return len(self.posts) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        N = len(self.posts)
        is_neg = idx // N
        i      = idx % N
        uid, post_vec = self.posts[i]

        if is_neg == 0:
            acc_vec = self.rw_dict[uid]; label = 1.0
        else:
            neg_uid = uid
            while neg_uid == uid:
                neg_uid = random.choice(self.uids)
            acc_vec = self.rw_dict[neg_uid]; label = 0.0

        post_t = torch.from_numpy(post_vec)
        acc_t  = torch.from_numpy(acc_vec.astype(np.float32))
        return post_t, acc_t, torch.tensor(label, dtype=torch.float32)

class PairClassifier(nn.Module):
    def __init__(self, post_dim, rw_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim + rw_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, post_vec, acc_vec):
        x = torch.cat([post_vec, acc_vec], dim=1)
        return self.net(x).squeeze(1)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = PairDataset(POSTS_CSV, ACCOUNT_NPY, neg_ratio=NEG_RATIO)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    sample_post, sample_acc, _ = ds[0]
    rw_dim = sample_acc.size(0)
    model  = PairClassifier(POST_DIM, rw_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 1
    best_loss   = float("inf")
    patience    = 0

    # resume
    if args.resume and os.path.isfile(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt["best_loss"]
        patience    = ckpt["patience"]
        print(f"[Resume] epoch={ckpt['epoch']}  best_loss={best_loss:.4f}  patience={patience}")

    for epoch in range(start_epoch, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for post, acc, label in loader:
            post, acc, label = post.to(device), acc.to(device), label.to(device)
            logits = model(post, acc)
            loss   = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * post.size(0)

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch:02d}  loss={avg_loss:.4f}")

        # early stopping & checkpoint
        if avg_loss < best_loss - MIN_DELTA:
            best_loss, patience = avg_loss, 0
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_loss":   best_loss,
                "patience":    patience
            }, CHECKPOINT_PATH)
            print(f"  ✔ saved checkpoint → {CHECKPOINT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_loss = {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="load checkpoint and continue training")
    args = parser.parse_args()
    train(args)