#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_feature_chunk.py ― 投稿ベクトルを特徴次元方向にチャンク化して
Set-Transformer でランダムウォーク埋め込みを予測する学習スクリプト
"""

import os, csv, argparse, math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, random_split
from tqdm import tqdm

# デフォルトパス（必要なら CLI で上書き）
POSTS_CSV   = "/workspace/edit_agent/vast/aggregated_posting_vectors.csv"
ACCOUNT_NPY = "/workspace/edit_agent/vast/account_vectors.npy"
CHECKPT     = "/workspace/edit_agent/vast/set_transformer_feature.pt"

# ハイパラ（CLI で上書き可）
POST_DIM     = 3072
ENC_DIM      = 512
RW_DIM       = 128    # ランダムウォーク埋め込み元の次元
N_HEADS      = 4
N_LAYERS     = 16
DROPOUT      = 0.3
LR           = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS       = 500
BATCH_SIZE   = 128
VAL_SPLIT    = 0.1
PATIENCE     = 15
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

class FeatureChunkDataset(IterableDataset):
    def __init__(self, posts_csv, account_npy, feature_chunk):
        self.posts_csv     = posts_csv
        self.feature_chunk = feature_chunk
        # ランダムウォーク埋め込み辞書
        self.rw_dict = np.load(account_npy, allow_pickle=True).item()
        # シーケンス長（チャンク数）
        self.seq_len = math.ceil(POST_DIM / feature_chunk)

    def __iter__(self):
        f = open(self.posts_csv, encoding='utf-8')
        rdr = csv.reader(f)
        next(rdr)  # header
        for row in rdr:
            uid = row[0]
            if uid not in self.rw_dict: 
                continue
            vec = parse_vec(row[2], POST_DIM)
            if vec is None: 
                continue
            # 特徴次元チャンク化
            chunks = []
            for i in range(self.seq_len):
                start = i * self.feature_chunk
                end   = min(start + self.feature_chunk, POST_DIM)
                sub = vec[start:end]
                if end - start < self.feature_chunk:
                    pad = np.zeros(self.feature_chunk, dtype=np.float32)
                    pad[: sub.shape[0]] = sub
                    sub = pad
                chunks.append(sub)
            tokens = torch.tensor(np.stack(chunks, 0), dtype=torch.float32)  # (seq_len, feature_chunk)
            target = torch.tensor(self.rw_dict[uid], dtype=torch.float32)    # (RW_DIM,)
            yield tokens, target
        f.close()

def collate_fn(batch):
    toks, tgts = zip(*batch)
    return torch.stack(toks, 0), torch.stack(tgts, 0)

class SetTransformerFeature(nn.Module):
    def __init__(self, feature_dim, enc_dim, rw_dim, n_heads, n_layers, dropout):
        super().__init__()
        # チャンクごとに埋め込み
        self.initial_projection = nn.Linear(feature_dim, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim, nhead=n_heads,
            dim_feedforward=enc_dim*4, dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = nn.Linear(enc_dim, rw_dim)

    def forward(self, tokens):
        # tokens: (B, S, F)
        x = self.initial_projection(tokens)     # (B, S, enc)
        x = x.permute(1,0,2)                    # (S, B, enc)
        x = self.encoder(x)                     # (S, B, enc)
        x = x.permute(1,0,2)                    # (B, S, enc)
        # 平均プーリング（全チャンク平均）
        pooled = x.mean(dim=1)                  # (B, enc)
        return self.decoder(pooled)             # (B, RW_DIM)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    ds = FeatureChunkDataset(
        args.posts_csv, args.account_npy, args.feature_chunk
    )
    # train/val split にするため一度全件リスト化
    all_samples = list(ds)
    n_val  = int(len(all_samples) * VAL_SPLIT)
    n_tr   = len(all_samples) - n_val
    tr_list, va_list = random_split(all_samples, [n_tr, n_val])
    tr_loader = DataLoader(tr_list, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(va_list, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SetTransformerFeature(
        feature_dim=args.feature_chunk,
        enc_dim=ENC_DIM,
        rw_dim=RW_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.MSELoss()

    start_ep = 1
    best_val = float("inf")
    patience = 0

    # resume
    if args.resume and os.path.isfile(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        start_ep = ck["epoch"] + 1
        print(f"[Resume] from epoch {start_ep}")

    for ep in range(start_ep, args.epochs+1):
        # train
        model.train()
        tot = 0.0
        for toks, tgt in tqdm(tr_loader, desc=f"Epoch {ep} [train]"):
            toks, tgt = toks.to(device), tgt.to(device)
            opt.zero_grad()
            pred = model(toks)
            loss = crit(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * toks.size(0)
        train_loss = tot / len(tr_loader.dataset)
        sch.step()

        # val
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for toks, tgt in tqdm(va_loader, desc=f"Epoch {ep} [val]"):
                toks, tgt = toks.to(device), tgt.to(device)
                tot += crit(model(toks), tgt).item() * toks.size(0)
        val_loss = tot / len(va_loader.dataset)

        print(f"Epoch {ep}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            ck = {
                "epoch": ep,
                "model": model.state_dict(),
                "opt":   opt.state_dict(),
                "sch":   sch.state_dict(),
            }
            torch.save(ck, args.checkpoint)
            print("  ✔ saved best →", args.checkpoint)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--posts_csv",       default=POSTS_CSV)
    p.add_argument("--account_npy",     default=ACCOUNT_NPY)
    p.add_argument("--checkpoint",      default=CHECKPT)
    p.add_argument("--feature_chunk",   type=int,   default=32,
                   help="3072次元を何次元ずつトークン化するか")
    p.add_argument("--batch_size",      type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",          type=int,   default=EPOCHS)
    p.add_argument("--resume",          action="store_true")
    args = p.parse_args()
    train(args)