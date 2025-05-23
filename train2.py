#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pair_classifier.py ― 投稿ベクトルとアカウントベクトルのペアを
正例(同一アカウント)=1／負例(異なるアカウント)=0 で学習するスクリプト

★ --resume を指定すると、前回のチェックポイントからモデル重み、
  オプティマイザ状態、ベストバリデーション損失、patience を復元して続きから学習します。
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
from torch.utils.data import Dataset, DataLoader, random_split

# ─────────── ハードコード済みパス ───────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CHECKPOINT_PATH = os.path.join(VAST_DIR, "pair_classifier_rw.ckpt")

# ──────────── ハイパラ ────────────
POST_DIM      = 3072
BATCH_SIZE    = 128
EPOCHS        = 500
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.1
PATIENCE      = 15
MIN_DELTA     = 1e-4
NEG_RATIO     = 5  # 正例1に対して負例1つ

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

class PairDataset(Dataset):
    """投稿ベクトル⇔アカウントベクトルのペアを正例／負例で返すデータセット"""
    def __init__(self, posts_csv, account_npy, neg_ratio=1):
        # アカウント→RW埋め込み辞書読み込み
        self.rw_dict = np.load(account_npy, allow_pickle=True).item()
        self.uids    = list(self.rw_dict.keys())

        # 投稿ベクトルを一括読み込み
        self.posts = []
        with open(posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_str in rdr:
                if uid not in self.rw_dict:
                    continue
                vec = parse_vec(vec_str, POST_DIM)
                if vec is not None:
                    self.posts.append((uid, vec))
        if not self.posts:
            sys.exit("ERROR: No valid post vectors found.")
        self.neg_ratio = neg_ratio

    def __len__(self):
        # 正例 + 負例合わせた総数
        return len(self.posts) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        N      = len(self.posts)
        is_neg = idx // N
        i      = idx % N
        uid, post_vec = self.posts[i]

        if is_neg == 0:
            # 正例：同一UID
            acc_vec = self.rw_dict[uid]
            label   = 1.0
        else:
            # 負例：別UID
            neg_uid = uid
            while neg_uid == uid:
                neg_uid = random.choice(self.uids)
            acc_vec = self.rw_dict[neg_uid]
            label   = 0.0

        post_t = torch.from_numpy(post_vec)
        acc_t  = torch.from_numpy(acc_vec.astype(np.float32))
        return post_t, acc_t, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    posts, accs, labels = zip(*batch)
    return torch.stack(posts), torch.stack(accs), torch.stack(labels)

class PairClassifier(nn.Module):
    """投稿＋アカウント連結→MLP→バイナリ分類"""
    def __init__(self, post_dim, rw_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim + rw_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, post_vec, acc_vec):
        x = torch.cat([post_vec, acc_vec], dim=1)
        return self.net(x).squeeze(1)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # データセット & DataLoader (train/val split)
    ds       = PairDataset(POSTS_CSV, ACCOUNT_NPY, neg_ratio=NEG_RATIO)
    n_val    = int(len(ds) * VAL_SPLIT)
    n_train  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_train, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn)

    # モデル初期化
    sample_post, sample_acc, _ = ds[0]
    rw_dim = sample_acc.size(0)
    model  = PairClassifier(POST_DIM, rw_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 1
    best_val    = float("inf")
    patience    = 0

    # resume
    if args.resume and os.path.isfile(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        patience    = ckpt["patience"]
        print(f"[Resume] epoch={ckpt['epoch']}  best_val={best_val:.4f}  patience={patience}")

    # 学習ループ
    for epoch in range(start_epoch, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for posts, accs, labels in tr_loader:
            posts, accs, labels = posts.to(device), accs.to(device), labels.to(device)
            logits = model(posts, accs)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * posts.size(0)
        train_loss = total_loss / len(tr_loader.dataset)

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for posts, accs, labels in va_loader:
                posts, accs, labels = posts.to(device), accs.to(device), labels.to(device)
                logits = model(posts, accs)
                total_val += criterion(logits, labels).item() * posts.size(0)
        val_loss = total_val / len(va_loader.dataset)

        print(f"Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # Checkpoint & Early Stopping
        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val":    best_val,
                "patience":    patience
            }, CHECKPOINT_PATH)
            print(f"  ✔ saved checkpoint → {CHECKPOINT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="load checkpoint and continue training")
    args = parser.parse_args()
    train(args)