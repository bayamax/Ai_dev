#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― 投稿ベクトルからランダムウォーク埋め込みを予測する Transformer モデル学習
  ・投稿ベクトル CSV、ランダムウォーク NPY はハードコード
  ・途中再開用チェックポイントオプションあり
"""

import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from model import SetToVectorPredictor  # 同じディレクトリに model.py を置いてください

# ────────────────────────────
# ハードコードするパス
# ────────────────────────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACC_NPY         = os.path.join(VAST_DIR, "account_vectors.npy")
DEFAULT_CKPT    = os.path.join(VAST_DIR, "set_transformer_rwwalk_checkpoint.pt")

# ────────────────────────────
# モデル＆学習ハイパラ
# ────────────────────────────
POST_DIM        = 3072   # 投稿埋め込み次元
HID_DIM         = 512    # Transformer エンコーダの内部次元
OUT_DIM         = 128    # ランダムウォーク埋め込み次元
N_HEADS         = 4
N_LAYERS        = 2
DROPOUT         = 0.1

LR              = 1e-4
WD              = 1e-5
BATCH_SIZE      = 64
EPOCHS          = 500
VAL_SPLIT       = 0.1
PATIENCE        = 15
MIN_DELTA       = 1e-4
MAX_POSTS       = 50     # 最大投稿数

# ────────────────────────────
# データセット定義
# ────────────────────────────
class RWDataset(Dataset):
    def __init__(self, posts_csv: str, acc_npy: str, max_posts: int):
        # ランダムウォーク埋め込み辞書を読み込み
        rw_dict = np.load(acc_npy, allow_pickle=True).item()
        self.acc_list = sorted(rw_dict.keys())
        self.acc_to_idx = {a:i for i,a in enumerate(self.acc_list)}
        self.out_dim = len(next(iter(rw_dict.values())))  # NPY 内のベクトル長

        # 投稿ベクトル読み込み
        self.data = []
        with open(posts_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            buf = {}
            for uid, _, vec_str in reader:
                if uid not in self.acc_to_idx:
                    continue
                # 文字列を NumPy 配列に変換
                arr = np.fromstring(
                    re.sub(r'[\s,]+',' ', vec_str.strip().strip('[]"')),
                    dtype=np.float32, sep=' '
                )
                if arr.shape[0] != POST_DIM:
                    continue
                buf.setdefault(uid, []).append(arr)
            # 各ユーザーごとに最大 MAX_POSTS 最新分だけを保持
            for uid, vecs in buf.items():
                if len(vecs) >= 1:
                    vecs = vecs[-max_posts:]
                    posts_tensor = torch.tensor(np.stack(vecs), dtype=torch.float32)
                    rw_vec = torch.tensor(rw_dict[uid], dtype=torch.float32)
                    self.data.append((posts_tensor, rw_vec))
        if not self.data:
            sys.exit("ERROR: データが一件も読み込めませんでした。")

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate_fn(batch):
    posts, targets = zip(*batch)
    lengths = torch.tensor([p.size(0) for p in posts])
    max_len = lengths.max().item()
    padded = torch.nn.utils.rnn.pad_sequence(posts, batch_first=True)
    padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets)
    return padded, padding_mask, targets

# ────────────────────────────
# 学習ルーチン
# ────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = RWDataset(POSTS_CSV, ACC_NPY, MAX_POSTS)
    n_val = int(len(ds) * VAL_SPLIT)
    n_tr  = len(ds) - n_val
    tr_set, va_set = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SetToVectorPredictor(
        post_embedding_dim=POST_DIM,
        encoder_output_dim=HID_DIM,
        num_all_accounts=None,        # デコーダ部分は使用せずに pooled のみを使う
        num_attention_heads=N_HEADS,
        num_encoder_layers=N_LAYERS,
        dropout_rate=DROPOUT
    ).to(device)

    # チェックポイントから再開
    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"[Resume] loading {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience = 0

    for ep in range(1, EPOCHS+1):
        model.train()
        sum_loss = 0.0
        for posts, mask, targets in tqdm(tr_loader, desc=f"Ep{ep}[train]"):
            posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad()
            _, pooled = model(posts, mask)
            pred = pooled  # pooled → (B, HID_DIM)
            # ランダムウォーク埋め込み（OUT_DIM）に変換する MLP を追加で定義している場合はここで呼び出し
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += loss.item() * posts.size(0)
        train_loss = sum_loss / len(tr_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for posts, mask, targets in va_loader:
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
                _, pooled = model(posts, mask)
                val_sum += criterion(pooled, targets).item() * posts.size(0)
        val_loss = val_sum / len(va_loader.dataset)

        print(f"Epoch {ep}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")
        scheduler.step()

        if val_loss + MIN_DELTA < best_val:
            best_val = val_loss
            patience = 0
            os.makedirs(os.path.dirname(DEFAULT_CKPT), exist_ok=True)
            torch.save(model.state_dict(), DEFAULT_CKPT)
            print(f"  ✔ saved checkpoint → {DEFAULT_CKPT}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"[Done] best_val = {best_val:.4f}")

# ────────────────────────────
# エントリポイント
# ────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train RW-vector predictor")
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                    help="checkpoint to load/save")
    args = ap.parse_args()
    train(args)