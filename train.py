#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― マルチビュー・コントラスト学習（正則化強化版＋デバッグログ）
  • SetEncoder + GNNビュー を MLPプロジェクションヘッド上で対比学習
  • 入力・隣接・ヘッドにドロップアウト
  • CosineAnnealingLR ＋ weight_decay
  • 各バッチごとに類似度統計・勾配ノルム・学習率を出力
"""

import os
import sys
import csv
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ─────────── ハードコード済みパス ───────────
VAST_DIR    = "/workspace/edit_agent/vast"
POSTS_CSV   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
EDGES_CSV   = os.path.join(VAST_DIR, "edges.csv")
ACCOUNT_NPY = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH   = os.path.join(VAST_DIR, "dualview_contrastive.pt")

# ─────────── ハイパラ ───────────
POST_DIM      = 3072
MAX_POST      = 50
ENC_DIM       = 512
GNN_DIM       = 512
PROJ_DIM      = 256    # プロジェクションヘッド中間次元
N_HEADS       = 4
N_LAYERS      = 2
DROPOUT       = 0.1
NEIGH_DROPOUT = 0.2    # 隣接ドロップアウト率

LR            = 5e-5   # 小さめ
WEIGHT_DECAY  = 5e-4   # 強め
EPOCHS        = 100
TAU           = 0.1
PATIENCE      = 10
MIN_DELTA     = 1e-4

ACCOUNT_BATCH_SIZE = 128

# ─────────── モデル定義 ───────────
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
        x = self.proj(x)                           # (B, S, D_enc)
        x = x.permute(1, 0, 2)                     # (S, B, D_enc)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x.permute(1, 0, 2)                     # (B, S, D_enc)
        valid = (~pad_mask).unsqueeze(2).float()   # (B, S, 1)
        summed = (x * valid).sum(dim=1)            # (B, D_enc)
        lengths = valid.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return summed / lengths                    # (B, D_enc)

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
        c = self.set_enc(posts, pad_mask)          # (B, D_enc)
        neigh_means = []
        for i, nbrs in enumerate(neighbor_list):
            if nbrs:
                neigh_means.append(c[nbrs].mean(dim=0))
            else:
                neigh_means.append(torch.zeros_like(c[i]))
        n = torch.stack(neigh_means, dim=0)        # (B, D_enc)
        combo = torch.cat([c, n], dim=1)           # (B, 2D_enc)
        s_raw = F.relu(self.gnn_lin(combo))        # (B, D_gnn)
        z = self.proj_head(s_raw)                  # (B, D_gnn)
        return c, z

# ────────── データ準備 ──────────
def load_data(posts_csv, edges_csv, acc_npy, max_posts, post_dim):
    rw_dict      = np.load(acc_npy, allow_pickle=True).item()
    account_list = sorted(rw_dict.keys())
    acc2idx      = {a:i for i,a in enumerate(account_list)}
    N            = len(account_list)

    user_posts = defaultdict(list)
    with open(posts_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr)
        for uid, _, vec_str in tqdm(rdr, desc="Loading posts"):
            if uid not in acc2idx: continue
            s = vec_str.strip()
            if s.startswith('"[') and s.endswith(']"'): s = s[1:-1]
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
    padded  = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True)
    masks   = torch.arange(Smax).unsqueeze(0) >= lengths.unsqueeze(1)

    neighbors = [[] for _ in range(N)]
    with open(edges_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr)
        for src, dst in tqdm(rdr, desc="Loading edges"):
            if src in acc2idx and dst in acc2idx:
                i, j = acc2idx[src], acc2idx[dst]
                neighbors[i].append(j)
                neighbors[j].append(i)

    return account_list, padded, masks, neighbors

# ────────── InfoNCE loss ──────────
def infonce_loss(c, z, tau):
    c_norm = F.normalize(c, dim=1)
    z_norm = F.normalize(z, dim=1)
    sim    = (c_norm @ z_norm.T) / tau       # (B, B)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_adj    = sim - sim_max.detach()
    logsum     = torch.logsumexp(sim_adj, dim=1) + sim_max.squeeze(1)
    loss       = (logsum - torch.diag(sim))   # (B,)
    return loss.mean()

# ─────────── 訓練ループ ───────────
def train_contrastive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ac_list, posts_cpu, mask_cpu, neighbors = load_data(
        POSTS_CSV, EDGES_CSV, ACCOUNT_NPY, MAX_POST, POST_DIM
    )
    N = len(ac_list)
    model = DualViewEncoder(
        POST_DIM, ENC_DIM, GNN_DIM, N_HEADS, N_LAYERS, DROPOUT, PROJ_DIM
    ).to(device)

    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val, patience = float("inf"), 0
    idxs = list(range(N))

    for ep in range(1, EPOCHS + 1):
        random.shuffle(idxs)
        total_loss, steps = 0.0, 0

        for st in range(0, N, ACCOUNT_BATCH_SIZE):
            batch_idx = idxs[st:st + ACCOUNT_BATCH_SIZE]
            posts_b   = posts_cpu[batch_idx].to(device)
            mask_b    = mask_cpu[batch_idx].to(device)

            # 隣接ドロップアウト
            local_map = {g: i for i, g in enumerate(batch_idx)}
            nbrs_b = []
            for g in batch_idx:
                nbrs = [local_map[n] for n in neighbors[g] if n in local_map]
                nbrs = [n for n in nbrs if random.random() > NEIGH_DROPOUT]
                nbrs_b.append(nbrs)

            model.train()
            opt.zero_grad()
            c_b, z_b = model(posts_b, mask_b, nbrs_b)

            # ── デバッグ: 類似度統計 ──
            with torch.no_grad():
                c_n = F.normalize(c_b, dim=1)
                z_n = F.normalize(z_b, dim=1)
                sim_mat = (c_n @ z_n.T) / TAU
                pos_sims = sim_mat.diag()
                neg_sims = sim_mat[~torch.eye(sim_mat.size(0), dtype=bool, device=device)].view(sim_mat.size(0), -1)
                print(f"[DEBUG][Ep{ep} B{steps+1}] pos μ={pos_sims.mean():.4f} σ={pos_sims.std():.4f}  "
                      f"neg μ={neg_sims.mean():.4f} σ={neg_sims.std():.4f}")

            loss = infonce_loss(c_b, z_b, TAU)
            loss.backward()

            # ── デバッグ: 勾配ノルム ──
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"[DEBUG][Ep{ep} B{steps+1}] grad_norm={total_norm:.4f}")

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # ── デバッグ: 学習率 ──
            lr = opt.param_groups[0]['lr']
            print(f"[DEBUG][Ep{ep} B{steps+1}] lr={lr:.2e}")

            total_loss += loss.item()
            steps += 1
            torch.cuda.empty_cache()

        train_loss = total_loss / steps
        sched.step()

        # 簡易バリデーション（最初のバッチ）
        with torch.no_grad():
            val_idx = idxs[:ACCOUNT_BATCH_SIZE]
            p_v     = posts_cpu[val_idx].to(device)
            m_v     = mask_cpu[val_idx].to(device)
            local_map = {g: i for i, g in enumerate(val_idx)}
            nbrs_v = []
            for g in val_idx:
                nbrs = [local_map[n] for n in neighbors[g] if n in local_map]
                nbrs = [n for n in nbrs if random.random() > NEIGH_DROPOUT]
                nbrs_v.append(nbrs)
            model.eval()
            c_v, z_v = model(p_v, m_v, nbrs_v)
            val_loss = infonce_loss(c_v, z_v, TAU).item()

        print(f"Ep {ep}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

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
    train_contrastive()