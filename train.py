#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― マルチビュー・コントラスト学習（軽量＋Augmentation 強化版）
  • 欠損投稿マスク & 隣接ドロップアウト
  • モデル容量削減（ENC_DIM=256, GNN_DIM=256, LAYERS=1）
  • TAU=0.3, WD=1e-3, batch_size=256, patience=5
"""

import os, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

# ─────────── ハードコード済みパス ───────────
VAST_DIR           = "/workspace/edit_agent/vast"
POSTS_CSV          = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
EDGES_CSV          = os.path.join(VAST_DIR, "edges.csv")
ACCOUNT_NPY        = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH          = os.path.join(VAST_DIR, "dualview_contrastive_aug.pt")

# ─────────── ハイパラ ───────────
POST_DIM           = 3072
MAX_POST           = 50
ENC_DIM            = 256
GNN_DIM            = 256
PROJ_DIM           = 128
N_HEADS            = 4
N_LAYERS           = 1
DROPOUT            = 0.1
POST_MASK_DROPOUT  = 0.2   # 投稿マスク率
NEIGH_DROPOUT      = 0.3   # 隣接ドロップ率

LR                 = 5e-5
WEIGHT_DECAY       = 1e-3
EPOCHS             = 100
TAU                = 0.3
PATIENCE           = 5
MIN_DELTA          = 1e-4

ACCOUNT_BATCH_SIZE = 256
EPS_NORMALIZE      = 1e-6

# ─────────── 文字列→ベクトルパース ───────────
def parse_vec(s, dim):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'): s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r: s = s[l+1:r]
    v = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return v if v.size == dim else None

# ─────────── モデル定義 ───────────
class SetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dropout = nn.Dropout(DROPOUT)
        self.proj = nn.Linear(POST_DIM, ENC_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=ENC_DIM, nhead=N_HEADS,
            dim_feedforward=ENC_DIM*4, dropout=DROPOUT,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)

    def forward(self, posts, pad_mask):
        x = self.input_dropout(posts)     # (B,S,POST_DIM)
        x = self.proj(x)                  # (B,S,ENC_DIM)
        x = x.permute(1, 0, 2)            # (S,B,ENC_DIM)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (S,B,ENC_DIM)
        x = x.permute(1, 0, 2)            # (B,S,ENC_DIM)
        valid = (~pad_mask).unsqueeze(-1).float()  # (B,S,1)
        sum_vec = (x * valid).sum(dim=1)          # (B,ENC_DIM)
        lengths = valid.sum(dim=1).clamp(min=1.0) # (B,1)
        return sum_vec / lengths                  # (B,ENC_DIM)

class DualViewEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.set_enc   = SetEncoder()
        self.gnn_lin   = nn.Linear(ENC_DIM*2, GNN_DIM)
        self.proj_head = nn.Sequential(
            nn.Linear(GNN_DIM, PROJ_DIM),
            nn.BatchNorm1d(PROJ_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(PROJ_DIM, GNN_DIM),
        )

    def forward(self, posts, pad_mask, neighbor_list):
        c = self.set_enc(posts, pad_mask)  # (B,ENC_DIM)
        neigh_means = []
        for i, nbrs in enumerate(neighbor_list):
            if nbrs:
                neigh_means.append(c[nbrs].mean(dim=0))
            else:
                neigh_means.append(torch.zeros_like(c[i]))
        n = torch.stack(neigh_means, dim=0)      # (B,ENC_DIM)
        combo = torch.cat([c, n], dim=1)         # (B,2*ENC_DIM)
        s_raw = F.relu(self.gnn_lin(combo))      # (B,GNN_DIM)
        z = self.proj_head(s_raw)                # (B,GNN_DIM)
        return c, z

# ─────────── データ準備 ───────────
def load_data():
    rw_dict = np.load(ACCOUNT_NPY, allow_pickle=True).item()
    posts_map = defaultdict(list)
    # 投稿ベクトル全ロード
    with open(POSTS_CSV, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr)
        for uid, _, vec_str in tqdm(rdr, desc="Loading posts"):
            if uid not in rw_dict: continue
            v = parse_vec(vec_str, POST_DIM)
            if v is not None:
                posts_map[uid].append(v)
    # 投稿ゼロ除外
    account_list = sorted([u for u, vs in posts_map.items() if vs])
    acc2idx = {u:i for i,u in enumerate(account_list)}
    N = len(account_list)
    # テンソル化＋パッディングマスク
    posts_list, pad_masks = [], []
    for uid in account_list:
        vecs = posts_map[uid][-MAX_POST:]
        t = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)
        m = torch.zeros((t.size(0),), dtype=torch.bool)
        posts_list.append(t)
        pad_masks.append(m)
    lengths = torch.tensor([p.size(0) for p in posts_list])
    Smax = lengths.max().item()
    padded = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True)   # (N,Smax,POST_DIM)
    masks  = torch.arange(Smax).unsqueeze(0) >= lengths.unsqueeze(1)          # (N,Smax)
    # 隣接リストロード
    neighbors = [[] for _ in range(N)]
    with open(EDGES_CSV, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr)
        for src, dst in tqdm(rdr, desc="Loading edges"):
            if src in acc2idx and dst in acc2idx:
                i, j = acc2idx[src], acc2idx[dst]
                neighbors[i].append(j)
                neighbors[j].append(i)
    return account_list, padded, masks, neighbors

# ─────────── InfoNCE Loss ───────────
def infonce_loss(c, z):
    c_n = F.normalize(c, dim=1, eps=EPS_NORMALIZE)
    z_n = F.normalize(z, dim=1, eps=EPS_NORMALIZE)
    sim = (c_n @ z_n.T) / TAU                  # (B,B)
    sim_max = sim.max(dim=1, keepdim=True).values
    sim_adj = sim - sim_max.detach()
    lsum = torch.logsumexp(sim_adj, dim=1) + sim_max.squeeze(1)
    loss = (lsum - torch.diag(sim))            # (B,)
    return loss.mean()

# ─────────── 訓練ループ ───────────
def train_contrastive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    accounts, posts_cpu, masks_cpu, neighbors = load_data()
    N = len(accounts)
    model = DualViewEncoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val, patience_cnt = float("inf"), 0
    indices = list(range(N))

    for ep in range(1, EPOCHS+1):
        random.shuffle(indices)
        total_loss, steps = 0.0, 0

        for st in range(0, N, ACCOUNT_BATCH_SIZE):
            batch = indices[st:st+ACCOUNT_BATCH_SIZE]
            posts_b = posts_cpu[batch].to(device)      # (B,Smax,POST_DIM)
            mask_b  = masks_cpu[batch].to(device)     # (B,Smax)
            # —— 投稿マスクドロップアウト —— 
            keep = (torch.rand_like(mask_b.float()) > POST_MASK_DROPOUT)
            mask_aug = mask_b | (~keep)
            # —— 隣接ドロップアウト —— 
            local_map = {g:i for i,g in enumerate(batch)}
            nbrs_b = []
            for g in batch:
                nbrs = [local_map[n] for n in neighbors[g] if n in local_map]
                nbrs = [n for n in nbrs if random.random() > NEIGH_DROPOUT]
                nbrs_b.append(nbrs)
            # forward / backward
            model.train(); optimizer.zero_grad()
            c_b, z_b = model(posts_b, mask_aug, nbrs_b)
            loss = infonce_loss(c_b, z_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            torch.cuda.empty_cache()

        train_loss = total_loss / steps
        scheduler.step()

        # —— 簡易バリデーション（バッチ頭） —— 
        with torch.no_grad():
            vb = indices[:ACCOUNT_BATCH_SIZE]
            p_v = posts_cpu[vb].to(device); m_v = masks_cpu[vb].to(device)
            keep_v = (torch.rand_like(m_v.float()) > POST_MASK_DROPOUT)
            m_aug_v = m_v | (~keep_v)
            local_v = {g:i for i,g in enumerate(vb)}
            nbrs_v = []
            for g in vb:
                ns = [local_v[n] for n in neighbors[g] if n in local_v]
                ns = [n for n in ns if random.random() > NEIGH_DROPOUT]
                nbrs_v.append(ns)
            model.eval()
            c_v, z_v = model(p_v, m_aug_v, nbrs_v)
            val_loss = infonce_loss(c_v, z_v).item()

        print(f"Ep {ep}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val - MIN_DELTA:
            best_val, patience_cnt = val_loss, 0
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            torch.save(model.state_dict(), CKPT_PATH)
            print("  ✔ saved best →", CKPT_PATH)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    train_contrastive()