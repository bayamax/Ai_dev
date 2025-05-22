#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_post_to_account.py ― 単一投稿ベクトルからアカウントベクトルを予測する Transformer 学習スクリプト

• 入力ファイル（ハードコード／CLIで上書き可）:
    - aggregated_posting_vectors.csv  (投稿ベクトル: uid, timestamp, "[v0, v1, …]" )
    - account_vectors.npy             (uid→ランダムウォーク埋め込み dict)
• モデル出力:
    - set_transformer_post2acc.pt
• 損失: CosineEmbeddingLoss（コサイン類似度最大化）
• 早期停止／モデル保存対応
"""

import os, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm

# ─────────── 既定パス ───────────
VAST_DIR       = "/workspace/edit_agent/vast"
DEFAULT_POSTS  = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
DEFAULT_ACCNPY = os.path.join(VAST_DIR, "account_vectors.npy")
DEFAULT_CKPT   = os.path.join(VAST_DIR, "set_transformer_post2acc.pt")

# ─────────── ハイパラ ───────────
POST_DIM     = 3072
MODEL_DIM    = 256
NUM_HEADS    = 4
NUM_LAYERS   = 2
DROPOUT      = 0.1
LR           = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 64
EPOCHS       = 50
VAL_SPLIT    = 0.1
PATIENCE     = 5
MIN_DELTA    = 1e-4

# ─────────── 文字列→ベクトルパース ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if not (0 <= l < r):
        return None
    s = s[l+1:r].replace(',', ' ')
    arr = np.fromstring(s, dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

# ─────────── Dataset ───────────
class Post2AccDataset(Dataset):
    def __init__(self, posts_csv, acc_npy):
        # ランダムウォーク埋め込み読み込み
        self.rw_dict = np.load(acc_npy, allow_pickle=True).item()
        self.rw_dim = next(iter(self.rw_dict.values())).shape[0]
        self.samples = []

        with open(posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid, *_ , vec_str in tqdm(rdr, desc="Loading posts"):
                if uid not in self.rw_dict:
                    continue
                v = parse_vec(vec_str, POST_DIM)
                if v is None:
                    continue
                self.samples.append((v, self.rw_dict[uid]))

        if not self.samples:
            raise RuntimeError("No valid samples loaded.")
        print(f"[Dataset] {len(self.samples)} samples, target dim = {self.rw_dim}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        v, rw = self.samples[i]
        return torch.from_numpy(v), torch.from_numpy(rw)

# ─────────── Model ───────────
class FeatureTransformer(nn.Module):
    def __init__(self, in_dim, model_dim, out_dim, n_heads, n_layers, dropout):
        super().__init__()
        # 特徴次元を「トークン系列」とみなす
        self.input_proj = nn.Linear(1, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim*4,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(model_dim, out_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, POST_DIM)
        B, D = x.shape
        y = x.unsqueeze(-1).permute(1,0,2)         # (D, B, 1)
        y = self.input_proj(y)                     # (D, B, model_dim)
        y = self.encoder(y)                        # (D, B, model_dim)
        y = y.permute(1,0,2).mean(dim=1)           # (B, model_dim)
        return self.head(y)                        # (B, out_dim)

# ─────────── Training ───────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = Post2AccDataset(args.posts_csv, args.acc_npy)
    n_val = int(len(ds)*VAL_SPLIT)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FeatureTransformer(
        in_dim=POST_DIM,
        model_dim=MODEL_DIM,
        out_dim=ds.rw_dim,
        n_heads=NUM_HEADS,
        n_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CosineEmbeddingLoss(margin=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, patience = float("inf"), 0
    for epoch in range(1, EPOCHS+1):
        # --- train ---
        model.train()
        total_loss = 0.0
        for x, y in tqdm(tr_loader, desc=f"Epoch {epoch} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            # CosineEmbeddingLoss requires label +1 for similar pairs
            target = torch.ones(x.size(0), device=device)
            loss = criterion(pred, y, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(tr_ds)

        # --- validation ---
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                tgt = torch.ones(x.size(0), device=device)
                total_val += criterion(pred, y, tgt).item() * x.size(0)
        val_loss = total_val / len(va_ds)

        print(f"Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # early stopping
        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save(model.state_dict(), args.ckpt)
            print("  ✔ saved best →", args.ckpt)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"[Done] best_val = {best_val:.4f}")

# ─────────── CLI ───────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--posts_csv", default=DEFAULT_POSTS)
    p.add_argument("--acc_npy",   default=DEFAULT_ACCNPY)
    p.add_argument("--ckpt",      default=DEFAULT_CKPT)
    args = p.parse_args()
    train(args)