#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― 投稿セットからランダムウォーク埋め込みを予測する Set‐Transformer 学習スクリプト
  • CSV はチャンク単位で読み込み、Pandas の itertuples() で高速かつ警告なしにアクセス
  • ランダムウォーク埋め込みは account_vectors.npy からロード
  • 欠損投稿ユーザはスキップ、最大 max_posts 件まで最新投稿を使用
  • Set‐Transformer による平均プーリング＋全結合デコーダ
  • MSELoss、AdamW、クリップ＆早期停止付き
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ───────── 既定ディレクトリ ─────────
VAST_DIR            = "/workspace/edit_agent/vast"
DEFAULT_POSTS_CSV   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
DEFAULT_ACCOUNT_NPY = os.path.join(VAST_DIR, "account_vectors.npy")
DEFAULT_SAVE_PATH   = os.path.join(VAST_DIR, "set_transformer_rw.pt")

# ───────── ハイパラ ─────────
POST_DIM     = 3072
ENC_DIM      = 512
MAX_POSTS    = 50
CHUNK_SIZE   = 1000  # CSV 読み込み時のチャンクサイズ（デフォルト）
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT    = 0.1
PATIENCE     = 10
MIN_DELTA    = 1e-4
CLIP_NORM    = 1.0

# ───────── データセットクラス ─────────
class RWDataset(Dataset):
    def __init__(self, posts_csv, account_npy, max_posts, chunk_size):
        # ランダムウォーク埋め込み読み込み
        try:
            rw_dict = np.load(account_npy, allow_pickle=True).item()
        except Exception as e:
            sys.exit(f"ERROR: account_vectors.npy の読み込みに失敗: {e}")
        # 埋め込み次元を推定
        first = next(iter(rw_dict.values()))
        self.rw_dim = first.shape[0]

        # 投稿埋め込みをチャンク単位で読み込み
        user_posts = defaultdict(list)
        for chunk in pd.read_csv(
            posts_csv,
            usecols=[0, 2],
            names=["uid", "_dummy", "vec"],
            header=0,
            chunksize=chunk_size,
            dtype={"uid": str, "vec": str},
        ):
            # itertuples() なら Series 警告なし
            for uid, vec_str in chunk.itertuples(index=False):
                if uid not in rw_dict:
                    continue
                s = vec_str.strip()
                if s.startswith('"[') and s.endswith(']"'):
                    s = s[1:-1]
                # カンマ／スペースどちらも sep として扱う
                arr = np.fromstring(s.replace(",", " "), dtype=np.float32, sep=" ")
                if arr.size != POST_DIM:
                    continue
                user_posts[uid].append(arr)

        # Tensor 化＆フィルタ
        self.samples = []
        for uid, vecs in user_posts.items():
            if not vecs:
                continue
            vecs = vecs[-max_posts:]
            posts_tensor = torch.tensor(
                np.stack(vecs, axis=0), dtype=torch.float32
            )  # (S, POST_DIM)
            target = torch.tensor(rw_dict[uid], dtype=torch.float32)  # (rw_dim,)
            self.samples.append((posts_tensor, target))

        if not self.samples:
            sys.exit("ERROR: 適切なサンプルが一件もロードできませんでした。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    posts_list, targets = zip(*batch)
    lengths = torch.tensor([p.size(0) for p in posts_list], dtype=torch.long)
    max_len = lengths.max().item()

    # (B, S, POST_DIM)
    padded = torch.nn.utils.rnn.pad_sequence(
        posts_list, batch_first=True, padding_value=0.0
    )
    # True = パディング位置
    padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets)  # (B, rw_dim)
    return padded, padding_mask, targets


# ───────── モデル定義 ─────────
class SetToRW(nn.Module):
    def __init__(self, post_dim, enc_dim, rw_dim, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(post_dim, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=n_heads,
            dim_feedforward=enc_dim * 4,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = nn.Linear(enc_dim, rw_dim)

    def forward(self, posts, pad_mask):
        # posts: (B, S, POST_DIM)
        x = self.proj(posts)  # (B, S, ENC_DIM)
        x = x.permute(1, 0, 2)  # (S, B, ENC_DIM)
        x = self.encoder(
            x, src_key_padding_mask=pad_mask
        )  # (S, B, ENC_DIM)
        x = x.permute(1, 0, 2)  # (B, S, ENC_DIM)

        # 平均プーリング（パディング無視）
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, S, 1)
        summed = (x * valid).sum(dim=1)  # (B, ENC_DIM)
        lengths = valid.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = summed / lengths  # (B, ENC_DIM)

        return self.decoder(pooled)  # (B, rw_dim)


# ───────── 学習ルーチン ─────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    ds = RWDataset(
        args.posts_csv, args.account_npy, args.max_posts, args.chunk_size
    )
    n_val = int(len(ds) * args.val_split)
    n_tr = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = SetToRW(
        post_dim=POST_DIM,
        enc_dim=args.enc_dim,
        rw_dim=ds.rw_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        # — train —
        model.train()
        sum_tr = 0.0
        for posts, mask, targets in tqdm(tr_loader, desc=f"Epoch {epoch} [train]"):
            posts, mask, targets = (
                posts.to(device),
                mask.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            preds = model(posts, mask)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            sum_tr += loss.item() * posts.size(0)
        tr_loss = sum_tr / len(tr_loader.dataset)

        # — validation —
        model.eval()
        sum_va = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_loader, desc=f"Epoch {epoch} [val]"):
                posts, mask, targets = (
                    posts.to(device),
                    mask.to(device),
                    targets.to(device),
                )
                sum_va += criterion(model(posts, mask), targets).item() * posts.size(0)
        va_loss = sum_va / len(va_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}")

        # 早期停止＆モデル保存
        if va_loss < best_val - args.min_delta:
            best_val = va_loss
            patience = 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✔ saved best → {args.save_path}")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping.")
                break

    print(f"[Done] best_val_loss = {best_val:.4f}")


# ───────── CLI ─────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Set‐Transformer RW 埋め込み予測 Trainer")
    p.add_argument(
        "--posts_csv",
        default=DEFAULT_POSTS_CSV,
        help="投稿ベクトル CSV ファイル (user_id, post_id, embedding_str)",
    )
    p.add_argument(
        "--account_npy",
        default=DEFAULT_ACCOUNT_NPY,
        help="ランダムウォーク埋め込み npy ファイル (dict)",
    )
    p.add_argument(
        "--save_path",
        default=DEFAULT_SAVE_PATH,
        help="学習済みモデル保存パス",
    )
    p.add_argument(
        "--max_posts",
        type=int,
        default=MAX_POSTS,
        help="ユーザあたり最大投稿数",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help="CSV 読み込みチャンクサイズ",
    )
    p.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="バッチサイズ"
    )
    p.add_argument("--epochs", type=int, default=EPOCHS, help="エポック数")
    p.add_argument("--lr", type=float, default=LR, help="学習率")
    p.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay (AdamW)",
    )
    p.add_argument(
        "--val_split",
        type=float,
        default=VAL_SPLIT,
        help="検証データ割合",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="早期停止 patience",
    )
    p.add_argument(
        "--min_delta",
        type=float,
        default=MIN_DELTA,
        help="ベスト更新判定 min_delta",
    )
    p.add_argument(
        "--enc_dim",
        type=int,
        default=ENC_DIM,
        help="Set‐Transformer Encoder 出力次元",
    )
    p.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Multi‐Head Attention ヘッド数",
    )
    p.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Transformer Encoder Layer 数",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="ドロップアウト率",
    )
    args = p.parse_args()
    train(args)