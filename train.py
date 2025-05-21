#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― ランダムウォークベクトル予測用 Transformer 学習スクリプト
  • 投稿ベクトルセット → ランダムウォークベクトル予測タスク
  • cosine 損失 (1 - cos) を使用
  • /workspace/edit_agent/vast 配下のファイルを参照
"""

import os
import sys
import csv
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from model import SetToVectorPredictor  # 同じフォルダに置いた model.py

# ───────── ハードコード定数 ─────────
VAST_DIR      = "/workspace/edit_agent/vast"
POSTS_CSV     = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACC_NPY       = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH     = os.path.join(VAST_DIR, "set_transformer_rw_predictor.pt")

POST_DIM      = 3072
ENC_DIM       = 128   # ランダムウォークベクトルの次元に合わせて
NUM_HEADS     = 4
NUM_LAYERS    = 2
DROPOUT       = 0.1

MAX_POSTS     = 50
BATCH_SIZE    = 64
EPOCHS        = 500
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.1
PATIENCE      = 15
MIN_DELTA     = 1e-4

# ───────── ユーティリティ ─────────
def parse_vector_string(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'): s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r: s = s[l+1:r]
    s = re.sub(r'[\s,]+', ' ', s).strip()
    arr = np.fromstring(s, dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

# ───────── データセット ─────────
class RWDataset(Dataset):
    def __init__(self, posts_csv, acc_npy, max_posts):
        # ランダムウォーク埋め込み読み込み
        data = np.load(acc_npy, allow_pickle=True).item()
        self.rw_dict = {k: v.astype(np.float32) for k,v in data.items()}
        # 投稿ベクトル読み込み
        tmp = defaultdict(list)
        with open(posts_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f); next(reader, None)
            for uid,_,vec_str in reader:
                vec = parse_vector_string(vec_str, POST_DIM)
                if vec is not None: tmp[uid].append(vec)
        self.samples = []
        for uid, vecs in tmp.items():
            if uid not in self.rw_dict: continue
            if not vecs: continue
            vecs = vecs[-max_posts:]
            post_tensor = torch.tensor(vecs, dtype=torch.float32)
            rw_tensor   = torch.tensor(self.rw_dict[uid], dtype=torch.float32)
            self.samples.append((post_tensor, rw_tensor))
        if not self.samples:
            sys.exit("ERROR: 有効なサンプルがありません。")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_fn(batch):
    posts, rws = zip(*batch)
    lengths = torch.tensor([p.size(0) for p in posts])
    M = lengths.max().item()
    padded = torch.nn.utils.rnn.pad_sequence(posts, batch_first=True, padding_value=0.0)
    mask   = torch.arange(M).unsqueeze(0) >= lengths.unsqueeze(1)
    targets= torch.stack(rws)
    return padded, mask, targets

# ───────── トレーニング ─────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = RWDataset(POSTS_CSV, ACC_NPY, MAX_POSTS)
    n_val = int(len(ds)*VAL_SPLIT); n_tr = len(ds)-n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SetToVectorPredictor(
        post_embedding_dim=POST_DIM,
        encoder_output_dim=ENC_DIM,
        num_all_accounts=ENC_DIM,    # 出力次元を ENC_DIM に見立て
        num_attention_heads=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        dropout_rate=DROPOUT
    ).to(device)
    # decoder を直接ベクトル出力に置き換え
    model.decoder = nn.Linear(ENC_DIM, ENC_DIM).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float('inf'); patience = 0

    for ep in range(1, EPOCHS+1):
        # train
        model.train(); sum_l=0.0
        for posts, mask, targets in tqdm(tr_ld, desc=f"Ep{ep}[train]"):
            posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(posts, mask)  # (B, ENC_DIM)
            # cosine loss = mean(1 - cos(u,v))
            cos = nn.functional.cosine_similarity(logits, targets, dim=1)
            loss = (1.0 - cos).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            sum_l += loss.item() * posts.size(0)
        tr_loss = sum_l / len(tr_ld.dataset)

        # val
        model.eval(); sum_v=0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_ld, desc=f"Ep{ep}[val]"):
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
                logits, _ = model(posts, mask)
                cos = nn.functional.cosine_similarity(logits, targets, dim=1)
                sum_v += ((1.0 - cos).sum().item())
        val_loss = sum_v / len(va_ld.dataset)

        print(f"Ep {ep}/{EPOCHS}  train={tr_loss:.6f}  val={val_loss:.6f}")
        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            torch.save(model.state_dict(), CKPT_PATH)
            print("  ✔ saved →", CKPT_PATH)
        else:
            patience += 1
            print(f"  (no improvement {patience}/{PATIENCE})")
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val = {best_val:.6f}")

if __name__=="__main__":
    train()