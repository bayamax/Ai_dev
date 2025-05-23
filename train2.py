#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pair_classifier_stream.py ― 投稿ベクトル ⇔ アカウントベクトル
ペアの正例(同一UID)=1／負例(異なるUID)=0 学習スクリプト
（ストリーミング読み込み＋ train/val split 対応版）
"""

import os, csv, random, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# ─────────── ハードコード済みパス ───────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CHECKPOINT_PATH = os.path.join(VAST_DIR, "pair_classifier_rw_stream.ckpt")

# ─────────── ハイパラ ────────────
POST_DIM     = 3072
BATCH_SIZE   = 128
EPOCHS       = 500
LR           = 1e-4
WEIGHT_DECAY = 1e-5
NEG_RATIO    = 5      # 正例１につき負例１
PATIENCE     = 15      # 早期停止
MIN_DELTA    = 1e-4
VAL_RATIO    = 0.1    # 全体の10%をバリデーションに割り当て

def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r: s = s[l+1:r]
    arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

def in_val_split(uid: str, ratio: float):
    """UID を md5 でハッシュして [0,1) に正規化、ratio 未満なら val に入れる"""
    h = int(hashlib.md5(uid.encode('utf-8')).hexdigest(), 16)
    return (h % 10000) / 10000.0 < ratio

class PairStreamDataset(IterableDataset):
    """
    train/val モードを選べるストリーミングデータセット
    split="train" or "val"
    """
    def __init__(self, posts_csv, account_npy, split="train"):
        assert split in ("train","val")
        self.posts_csv = posts_csv
        self.rw_dict   = np.load(account_npy, allow_pickle=True).item()
        self.uids      = list(self.rw_dict.keys())
        self.split     = split
        self.rw_dim    = next(iter(self.rw_dict.values())).shape[0]

    def __iter__(self):
        with open(self.posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f)
            next(rdr)
            for uid, _, vec_str in rdr:
                # まず split 判定
                is_val = in_val_split(uid, VAL_RATIO)
                if self.split=="train" and is_val: continue
                if self.split=="val"   and not is_val: continue

                vec = parse_vec(vec_str, POST_DIM)
                if vec is None or uid not in self.rw_dict:
                    continue

                post_t = torch.from_numpy(vec)
                # 正例
                acc_pos = torch.from_numpy(self.rw_dict[uid].astype(np.float32))
                yield post_t, acc_pos, torch.tensor(1.0)
                # 負例
                for _ in range(NEG_RATIO):
                    neg = uid
                    while neg == uid:
                        neg = random.choice(self.uids)
                    acc_neg = torch.from_numpy(self.rw_dict[neg].astype(np.float32))
                    yield post_t, acc_neg, torch.tensor(0.0)

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

    # train／val 用データセット
    train_ds = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, split="train")
    val_ds   = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = PairClassifier(POST_DIM, train_ds.rw_dim).to(device)
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

    for epoch in range(start_epoch, EPOCHS+1):
        # — train —
        model.train()
        total_train = 0.0; n_train = 0
        for post, acc, label in train_loader:
            post, acc, label = post.to(device), acc.to(device), label.to(device)
            logits = model(post, acc)
            loss   = criterion(logits, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            batch_n = post.size(0)
            total_train += loss.item() * batch_n
            n_train     += batch_n
        train_loss = total_train / n_train

        # — val —
        model.eval()
        total_val = 0.0; n_val = 0
        with torch.no_grad():
            for post, acc, label in val_loader:
                post, acc, label = post.to(device), acc.to(device), label.to(device)
                logits = model(post, acc)
                total_val += criterion(logits, label).item() * post.size(0)
                n_val     += post.size(0)
        val_loss = total_val / n_val

        print(f"Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # checkpoint & early stop
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