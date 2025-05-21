#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― マルチビュー・コントラスト学習（バッチ処理対応版）
投稿セットとグラフ構造を同一空間にマッピングします
"""

import os
import sys
import csv
import argparse
from collections import defaultdict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import logsumexp
from tqdm import tqdm

# ─────────── ハードコード済みパス ───────────
VAST_DIR    = "/workspace/edit_agent/vast"
POSTS_CSV   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
EDGES_CSV   = os.path.join(VAST_DIR, "edges.csv")
ACCOUNT_NPY = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH   = os.path.join(VAST_DIR, "dualview_contrastive.pt")

# ─────────── ハイパラ ───────────
POST_DIM   = 3072
MAX_POST   = 50
ENC_DIM    = 512
GNN_DIM    = 512
N_HEADS    = 4
N_LAYERS   = 2
DROPOUT    = 0.1

LR         = 1e-4
EPOCHS     = 100
TAU        = 0.1
PATIENCE   = 10
MIN_DELTA  = 1e-4

# アカウントごとのコントラスト学習ミニバッチサイズ
ACCOUNT_BATCH_SIZE = 128

# ─────────── モデル定義 ───────────
class SetEncoder(nn.Module):
    def __init__(self, post_dim, enc_dim, n_heads, n_layers, dropout):
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

    def forward(self, posts, pad_mask):
        # posts: (B, S, D)   pad_mask: (B, S)
        x = self.proj(posts)                         # (B, S, D)
        x = x.permute(1, 0, 2)                       # (S, B, D)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (S, B, D)
        x = x.permute(1, 0, 2)                       # (B, S, D)
        valid = (~pad_mask).unsqueeze(2).float()     # (B, S, 1)
        summed  = (x * valid).sum(dim=1)             # (B, D)
        lengths = valid.sum(dim=1).clamp(min=1.0)    # (B, 1)
        return summed / lengths                     # (B, D)

class DualViewEncoder(nn.Module):
    def __init__(self, post_dim, enc_dim, gnn_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.set_enc = SetEncoder(post_dim, enc_dim, n_heads, n_layers, dropout)
        self.gnn_lin = nn.Linear(enc_dim * 2, gnn_dim)

    def forward(self, posts, pad_mask, neighbor_list):
        # posts: (B, S, D)  pad_mask: (B, S)  neighbor_list: list[list[int]]
        c = self.set_enc(posts, pad_mask)        # (B, D)
        neigh_means = []
        for i, nbrs in enumerate(neighbor_list):
            if nbrs:
                neigh_means.append(c[nbrs].mean(dim=0))
            else:
                neigh_means.append(torch.zeros_like(c[i]))
        nbr   = torch.stack(neigh_means, dim=0)  # (B, D)
        combo = torch.cat([c, nbr], dim=1)       # (B, 2D)
        s     = F.relu(self.gnn_lin(combo))      # (B, gnn_dim)
        return c, s

def load_data(posts_csv, edges_csv, acc_npy, max_posts, post_dim):
    """CPU上に全データをロードし、Tensorやリストで返す"""
    rw_dict      = np.load(acc_npy, allow_pickle=True).item()
    account_list = sorted(rw_dict.keys())
    acc2idx      = {a:i for i,a in enumerate(account_list)}
    N            = len(account_list)

    # 投稿セット読み込み
    user_posts = defaultdict(list)
    with open(posts_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr)
        for uid, _, vec_str in tqdm(rdr, desc="Loading posts"):
            if uid not in acc2idx: continue
            s = vec_str.strip()
            if s.startswith('"[') and s.endswith(']"'):
                s = s[1:-1]
            l, r = s.find('['), s.rfind(']')
            if 0<=l<r: s = s[l+1:r]
            arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
            if arr.size != post_dim: continue
            user_posts[uid].append(arr)

    posts_list, pad_masks = [], []
    for uid in account_list:
        vecs = user_posts.get(uid, [])
        if not vecs:
            tensor = torch.zeros((1, post_dim), dtype=torch.float32)
            mask   = torch.tensor([[False]], dtype=torch.bool)
        else:
            vecs   = vecs[-max_posts:]
            tensor = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)
            mask   = torch.zeros((tensor.size(0),), dtype=torch.bool)
        posts_list.append(tensor)
        pad_masks.append(mask)

    lengths = torch.tensor([p.size(0) for p in posts_list])
    Smax    = lengths.max().item()
    padded  = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True)  # (N, Smax, D)
    masks   = torch.arange(Smax).unsqueeze(0) >= lengths.unsqueeze(1)         # (N, Smax)

    # 隣接リスト読み込み
    neighbors = [[] for _ in range(N)]
    with open(edges_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr)
        for src, dst in tqdm(rdr, desc="Loading edges"):
            if src in acc2idx and dst in acc2idx:
                i, j = acc2idx[src], acc2idx[dst]
                neighbors[i].append(j)
                neighbors[j].append(i)

    return account_list, padded, masks, neighbors

def infonce_loss(c, s, tau):
    c_norm = F.normalize(c, dim=1)
    s_norm = F.normalize(s, dim=1)
    sim    = (c_norm @ s_norm.T) / tau    # (B, B)
    row_max, _ = sim.max(dim=1, keepdim=True)
    sim_adj    = sim - row_max
    logsum     = logsumexp(sim_adj, dim=1) + row_max.squeeze(1)
    loss       = (logsum - torch.diag(sim))  # (B,)
    return loss.mean()

def train_contrastive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) データを CPU 上にまとめて読み込む
    account_list, posts_cpu, mask_cpu, neighbors = load_data(
        POSTS_CSV, EDGES_CSV, ACCOUNT_NPY, MAX_POST, POST_DIM
    )
    N = len(account_list)

    model = DualViewEncoder(POST_DIM, ENC_DIM, GNN_DIM, N_HEADS, N_LAYERS, DROPOUT).to(device)
    opt   = optim.AdamW(model.parameters(), lr=LR)

    best_val, patience = float("inf"), 0
    indices = list(range(N))

    for ep in range(1, EPOCHS+1):
        # シャッフルしてミニバッチ単位で学習
        random.shuffle(indices)
        total_loss = 0.0
        steps = 0

        for start in range(0, N, ACCOUNT_BATCH_SIZE):
            batch_idx = indices[start:start+ACCOUNT_BATCH_SIZE]

            # バッチ用テンソルを GPU に転送
            posts_batch    = posts_cpu[batch_idx].to(device)
            pad_mask_batch = mask_cpu[batch_idx].to(device)

            # 隣接リストも「バッチ内のみ」で再マッピング
            local_map = {g: i for i, g in enumerate(batch_idx)}
            nbrs_batch = []
            for g in batch_idx:
                # 同じバッチにいる neighbor だけ残す
                nbrs_batch.append([local_map[n] for n in neighbors[g] if n in local_map])

            model.train()
            opt.zero_grad()
            c_b, s_b = model(posts_batch, pad_mask_batch, nbrs_batch)
            loss     = infonce_loss(c_b, s_b, TAU)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            steps += 1

            # メモリをクリア
            torch.cuda.empty_cache()

        avg_train = total_loss / steps
        # 簡易バリデーション：最初のミニバッチだけで評価
        with torch.no_grad():
            val_idx = indices[:ACCOUNT_BATCH_SIZE]
            p_val   = posts_cpu[val_idx].to(device)
            m_val   = mask_cpu[val_idx].to(device)
            lm      = {g:i for i,g in enumerate(val_idx)}
            nbrs_val = [[lm[n] for n in neighbors[g] if n in lm] for g in val_idx]
            model.eval()
            c_v, s_v = model(p_val, m_val, nbrs_val)
            val_loss = infonce_loss(c_v, s_v, TAU).item()

        print(f"Epoch {ep}/{EPOCHS} — train={avg_train:.4f}  val={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            torch.save(model.state_dict(), CKPT_PATH)
            print("  ✔ saved best →", CKPT_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual‐View Contrastive Training (Batch)")
    _ = parser.parse_args()
    train_contrastive()