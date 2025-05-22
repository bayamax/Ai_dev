#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― マルチビュー・コントラスト学習（投稿ゼロアカウント除外＋eps 正規化）
  ※ CSV読み込みはあくまでCPU上で for ループにより逐次的に行います
"""

import os
import csv
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
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
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    s = s.replace(',', ' ')
    arr = np.fromstring(s, dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

class SetEncoder(nn.Module):
    def __init__(self, post_dim, enc_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(post_dim, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim, nhead=n_heads,
            dim_feedforward=enc_dim*4, dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, posts, pad_mask):
        x = self.proj(posts)                           
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
            neigh_means.append(
                c[nbrs].mean(dim=0) if nbrs else torch.zeros_like(c[i])
            )
        n     = torch.stack(neigh_means, dim=0)    
        combo = torch.cat([c, n], dim=1)           
        s_raw = F.relu(self.gnn_lin(combo))        
        z     = self.proj_head(s_raw)              
        return c, z

# ─────────── データ準備（元のまま CPU 上で逐次読み込み） ───────────
def load_data(posts_csv, edges_csv, acc_npy, max_posts, post_dim):
    # 1) ランダムウォーク埋め込み読み込み
    rw_dict   = np.load(acc_npy, allow_pickle=True).item()

    # 2) 投稿ベクトルを一行ずつ CPU で読み込む
    posts_map = defaultdict(list)
    with open(posts_csv, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for uid, _, vec_str in tqdm(reader, desc="Loading posts"):
            if uid not in rw_dict:
                continue
            vec = parse_vec(vec_str, post_dim)
            if vec is not None:
                posts_map[uid].append(vec)

    # 3) 投稿ゼロアカウントを除外
    account_list = sorted([u for u, vs in posts_map.items() if vs])
    acc2idx      = {a:i for i,a in enumerate(account_list)}
    N            = len(account_list)

    # 4) Tensor化 & マスク
    posts_list, pad_masks = [], []
    for uid in account_list:
        vecs   = posts_map[uid][-max_posts:]
        tensor = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)
        mask   = torch.zeros((tensor.size(0),), dtype=torch.bool)
        posts_list.append(tensor)
        pad_masks.append(mask)
    lengths = torch.tensor([p.size(0) for p in posts_list])
    Smax    = lengths.max().item()
    padded  = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True)
    masks   = torch.arange(Smax).unsqueeze(0) >= lengths.unsqueeze(1)

    # 5) 隣接リスト読み込み
    neighbors = [[] for _ in range(N)]
    with open(edges_csv, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for src, dst in tqdm(reader, desc="Loading edges"):
            if src in acc2idx and dst in acc2idx:
                i, j = acc2idx[src], acc2idx[dst]
                neighbors[i].append(j)
                neighbors[j].append(i)

    return account_list, padded, masks, neighbors

def infonce_loss(c, z, tau):
    c_norm = F.normalize(c, dim=1, eps=EPS_NORMALIZE)
    z_norm = F.normalize(z, dim=1, eps=EPS_NORMALIZE)
    sim     = (c_norm @ z_norm.T) / tau       
    sim_max = sim.max(dim=1, keepdim=True).values
    sim_adj = sim - sim_max.detach()
    logsum  = torch.logsumexp(sim_adj, dim=1) + sim_max.squeeze(1)
    loss    = (logsum - torch.diag(sim))
    return loss.mean()

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

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val, patience = float("inf"), 0
    idxs = list(range(N))

    for ep in range(1, EPOCHS+1):
        random.shuffle(idxs)
        total_loss, steps = 0.0, 0

        for st in range(0, N, ACCOUNT_BATCH_SIZE):
            batch_idx = idxs[st:st+ACCOUNT_BATCH_SIZE]
            posts_b   = posts_cpu[batch_idx].to(device)
            mask_b    = mask_cpu[batch_idx].to(device)
            # …（以下、元の学習ループ処理をそのまま）
            # infoNCE → backward → clip → step → LR scheduler…
            # ロギング、early stopping も変更なし
            …
        # 省略

    # 省略

if __name__ == "__main__":
    train_contrastive()