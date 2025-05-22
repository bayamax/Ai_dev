#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― マルチビュー・コントラスト学習（正則化強化版＋デバッグログ＋再開機能）
  • SetEncoder + GNNビュー を MLPプロジェクションヘッド上で対比学習
  • 入力・隣接・ヘッドにドロップアウト
  • CosineAnnealingLR ＋ weight_decay
  • 各バッチごとに類似度統計・勾配ノルム・学習率を出力
  • --resume で前回チェックポイントから再開（patience はリセット）
"""

import os
import csv
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# ─────────── ハードコード済みパス ───────────
VAST_DIR    = "/workspace/edit_agent/vast"
POSTS_CSV   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
EDGES_CSV   = os.path.join(VAST_DIR, "edges.csv")
ACCOUNT_NPY = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH   = os.path.join(VAST_DIR, "dualview_contrastive.ckpt")

# ─────────── ハイパラ ───────────
POST_DIM      = 3072
MAX_POST      = 50
ENC_DIM       = 512
GNN_DIM       = 512
PROJ_DIM      = 256
N_HEADS       = 4
N_LAYERS      = 2
DROPOUT       = 0.1
NEIGH_DROPOUT = 0.2

LR            = 5e-5
WEIGHT_DECAY  = 5e-4
EPOCHS        = 100
TAU           = 0.1
PATIENCE      = 10
MIN_DELTA     = 1e-4

ACCOUNT_BATCH_SIZE = 128
EPS_NORMALIZE      = 1e-6

def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'): s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r: s = s[l+1:r]
    arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

class SetEncoder(nn.Module):
    def __init__(self, post_dim, enc_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(post_dim, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim, nhead=n_heads,
            dim_feedforward=enc_dim*4, dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, posts, pad_mask):
        x = self.input_dropout(posts)
        x = self.proj(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x.permute(1, 0, 2)
        valid = (~pad_mask).unsqueeze(2).float()
        summed = (x * valid).sum(dim=1)
        lengths = valid.sum(dim=1).clamp(min=1.0)
        return summed / lengths

class DualViewEncoder(nn.Module):
    def __init__(self, post_dim, enc_dim, gnn_dim, n_heads, n_layers, dropout, proj_dim):
        super().__init__()
        self.set_enc   = SetEncoder(post_dim, enc_dim, n_heads, n_layers, dropout)
        self.gnn_lin   = nn.Linear(enc_dim*2, gnn_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(gnn_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, gnn_dim)
        )

    def forward(self, posts, pad_mask, neighbor_list):
        c = self.set_enc(posts, pad_mask)
        neigh_means = []
        for i, nbrs in enumerate(neighbor_list):
            if nbrs:
                neigh_means.append(c[nbrs].mean(dim=0))
            else:
                neigh_means.append(torch.zeros_like(c[i]))
        n     = torch.stack(neigh_means, dim=0)
        combo = torch.cat([c, n], dim=1)
        s_raw = F.relu(self.gnn_lin(combo))
        z     = self.proj_head(s_raw)
        return c, z

def load_data(posts_csv, edges_csv, acc_npy, max_posts, post_dim):
    rw_dict   = np.load(acc_npy, allow_pickle=True).item()
    posts_map = defaultdict(list)
    with open(posts_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr)
        for uid, _, vec_str in tqdm(rdr, desc="Loading posts"):
            if uid not in rw_dict: continue
            vec = parse_vec(vec_str, post_dim)
            if vec is not None: posts_map[uid].append(vec)
    account_list = sorted(uid for uid, vecs in posts_map.items() if vecs)
    acc2idx      = {a:i for i,a in enumerate(account_list)}

    posts_list, pad_masks = [], []
    for uid in account_list:
        vecs   = posts_map[uid][-max_posts:]
        tensor = torch.tensor(np.stack(vecs,0), dtype=torch.float32)
        mask   = torch.zeros((tensor.size(0),), dtype=torch.bool)
        posts_list.append(tensor)
        pad_masks.append(mask)

    lengths = torch.tensor([p.size(0) for p in posts_list])
    Smax    = lengths.max().item()
    padded  = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True)
    masks   = torch.arange(Smax).unsqueeze(0) >= lengths.unsqueeze(1)

    neighbors = [[] for _ in range(len(account_list))]
    with open(edges_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr)
        for src, dst in tqdm(rdr, desc="Loading edges"):
            if src in acc2idx and dst in acc2idx:
                i, j = acc2idx[src], acc2idx[dst]
                neighbors[i].append(j)
                neighbors[j].append(i)

    return account_list, padded, masks, neighbors

def infonce_loss(c, z, tau, eps=EPS_NORMALIZE):
    c_norm = F.normalize(c, dim=1, eps=eps)
    z_norm = F.normalize(z, dim=1, eps=eps)
    sim     = (c_norm @ z_norm.T) / tau
    sim_max = sim.max(dim=1, keepdim=True).values
    sim_adj = sim - sim_max.detach()
    logsum  = torch.logsumexp(sim_adj, dim=1) + sim_max.squeeze(1)
    loss    = (logsum - torch.diag(sim))
    return loss.mean()

def train_contrastive(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ac_list, posts_cpu, mask_cpu, neighbors = load_data(
        POSTS_CSV, EDGES_CSV, ACCOUNT_NPY, MAX_POST, POST_DIM
    )
    N = len(ac_list)

    model     = DualViewEncoder(POST_DIM, ENC_DIM, GNN_DIM, N_HEADS, N_LAYERS, DROPOUT, PROJ_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 1
    best_val    = float("inf")
    patience    = 0

    if args.resume and os.path.isfile(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        # patience はリセットする
        patience = 0
        print(f"[Resume] epoch={ckpt['epoch']}  best_val={best_val:.4f}  patience reset to 0")

    idxs = list(range(N))
    for ep in range(start_epoch, EPOCHS+1):
        random.shuffle(idxs)
        total_loss, steps = 0.0, 0

        for st in range(0, N, ACCOUNT_BATCH_SIZE):
            batch_idx = idxs[st:st+ACCOUNT_BATCH_SIZE]
            posts_b   = posts_cpu[batch_idx].to(device)
            mask_b    = mask_cpu[batch_idx].to(device)
            local_map = {g:i for i,g in enumerate(batch_idx)}
            nbrs_b    = [
                [local_map[n] for n in neighbors[g] if n in local_map and random.random()>NEIGH_DROPOUT]
                for g in batch_idx
            ]

            model.train(); optimizer.zero_grad()
            c_b, z_b = model(posts_b, mask_b, nbrs_b)
            loss     = infonce_loss(c_b, z_b, TAU)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps      += 1
            torch.cuda.empty_cache()

        train_loss = total_loss / steps
        scheduler.step()

        # バリデーション
        with torch.no_grad():
            val_idx = idxs[:ACCOUNT_BATCH_SIZE]
            p_v     = posts_cpu[val_idx].to(device)
            m_v     = mask_cpu[val_idx].to(device)
            local_map = {g:i for i,g in enumerate(val_idx)}
            nbrs_v    = [
                [local_map[n] for n in neighbors[g] if n in local_map and random.random()>NEIGH_DROPOUT]
                for g in val_idx
            ]
            model.eval()
            c_v, z_v = model(p_v, m_v, nbrs_v)
            val_loss = infonce_loss(c_v, z_v, TAU).item()

        print(f"Ep {ep}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            torch.save({
                "epoch":     ep,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val":  best_val,
                "patience":  patience,
            }, CKPT_PATH)
            print(f"  ✔ saved ckpt → {CKPT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual‐View Contrastive Training")
    parser.add_argument("--resume", action="store_true", help="load checkpoint and resume training")
    args = parser.parse_args()
    train_contrastive(args)