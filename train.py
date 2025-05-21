#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_graph_autoencoder.py ― 投稿セット特徴 + GNN を用いたグラフオートエンコーダ
  • ノード特徴: Set-Transformer で投稿埋め込みセットをエンコード
  • GNN エンコーダ: GraphSAGE ライクな隣接平均メッセージパッシング
  • デコーダ: 内積によるリンク再構築
  • 損失: BCEWithLogitsLoss（正例×負例サンプリング）
  • 早期停止・モデル保存機能
"""

import os
import sys
import csv
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from tqdm import tqdm

# ────────── ハードコード済みパス ──────────
VAST_DIR      = "/workspace/edit_agent/vast"
POSTS_CSV     = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
EDGES_CSV     = os.path.join(VAST_DIR, "edges.csv")
MODEL_SAVE    = os.path.join(VAST_DIR, "gae_follow_predictor.pt")

# ────────── ハイパラ ──────────
POST_DIM      = 3072    # 投稿埋め込み次元
ENC_DIM       = 512     # Set-Transformer 及び GNN 中間次元
GNN_LAYERS    = 2       # メッセージパッシング層数
MAX_POSTS     = 50      # 投稿セット最大数
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
EPOCHS        = 100
VAL_RATIO     = 0.1
PATIENCE      = 10
MIN_DELTA     = 1e-4
NEG_SAMPLE_RATIO = 1.0  # 正例数に対する負例比率

# ────────── ユーティリティ: 埋め込み文字列→ベクトル ──────────
def parse_vec(s: str, dim: int) -> np.ndarray:
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l = s.find('['); r = s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    s = " ".join(s.replace(",", " ").split())
    v = np.fromstring(s, dtype=np.float32, sep=' ')
    return v if v.size == dim else None

# ────────── データ読み込み ──────────
def load_data(posts_csv, edges_csv, post_dim, max_posts):
    # 1) 投稿セット読み込み
    user_posts = defaultdict(list)
    with open(posts_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr, None)
        for uid, _, vec_str in tqdm(rdr, desc="Loading posts"):
            vec = parse_vec(vec_str, post_dim)
            if vec is not None:
                user_posts[uid].append(vec)
    # ノードリスト
    all_uids = sorted(user_posts.keys())
    uid2idx  = {u:i for i,u in enumerate(all_uids)}
    N = len(all_uids)
    # パディングしてテンソル化
    posts_list = []
    pad_masks   = []
    for uid in all_uids:
        vecs = user_posts[uid][-max_posts:]
        t = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)  # (S, D)
        L = t.size(0)
        posts_list.append(t)
        pad_masks.append(torch.tensor([False]*L, dtype=torch.bool))
    lengths = torch.tensor([p.size(0) for p in posts_list])
    Smax = int(lengths.max().item())
    # pad_sequence で (N, Smax, D)
    posts_padded = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True)
    pad_mask     = torch.arange(Smax).unsqueeze(0) >= lengths.unsqueeze(1)  # True=pad

    # 2) エッジ読み込み
    edges = []
    with open(edges_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr, None)
        for src, dst in tqdm(rdr, desc="Loading edges"):
            if src in uid2idx and dst in uid2idx:
                u, v = uid2idx[src], uid2idx[dst]
                edges.append((u, v))
    # 3) train/val split for positive edges
    random.shuffle(edges)
    n_val = int(len(edges) * VAL_RATIO)
    val_edges = edges[:n_val]
    train_edges = edges[n_val:]
    pos_set = set(train_edges)

    # 4) 隣接リスト（全ノード）
    neighbors = [[] for _ in range(N)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    return all_uids, posts_padded, pad_mask, neighbors, train_edges, val_edges, pos_set

# ────────── モデル定義 ──────────
class SetEncoder(nn.Module):
    def __init__(self, post_dim, enc_dim, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(post_dim, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim, nhead=n_heads,
            dim_feedforward=enc_dim*4, dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, posts, pad_mask):
        # posts: (N, S, D), pad_mask: (N, S)
        x = self.proj(posts)           # (N, S, E)
        x = x.permute(1,0,2)           # (S, N, E)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (S, N, E)
        x = x.permute(1,0,2)           # (N, S, E)
        valid = (~pad_mask).unsqueeze(2).float()
        summed = (x * valid).sum(dim=1)        # (N, E)
        lengths = valid.sum(dim=1).clamp(min=1.0)
        return summed / lengths               # (N, E)

class GraphSAGEEncoder(nn.Module):
    def __init__(self, enc_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(enc_dim*2, enc_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, neighbors):
        # h: (N, E), neighbors: List[List[int]]
        for lin in self.layers:
            agg = []
            for i, nbrs in enumerate(neighbors):
                if nbrs:
                    m = h[nbrs].mean(dim=0)
                else:
                    m = torch.zeros_like(h[i])
                agg.append(m)
            m = torch.stack(agg, dim=0)      # (N, E)
            h = torch.cat([h, m], dim=1)     # (N, 2E)
            h = lin(self.dropout(h))
            h = F.relu(h)
        return h  # (N, E)

class GraphAutoEncoder(nn.Module):
    def __init__(self, post_dim, enc_dim, n_heads, st_layers, gnn_layers, dropout):
        super().__init__()
        self.set_enc   = SetEncoder(post_dim, enc_dim, n_heads, st_layers, dropout)
        self.gnn_enc   = GraphSAGEEncoder(enc_dim, gnn_layers, dropout)

    def forward(self, posts, pad_mask, neighbors):
        h0 = self.set_enc(posts, pad_mask)          # (N, E)
        h  = self.gnn_enc(h0, neighbors)            # (N, E)
        return h

# ────────── リンク再構築損失 ──────────
def reconstruction_loss(h, pos_edges, neg_edges, device):
    # pos
    ui = torch.tensor([u for u,v in pos_edges], dtype=torch.long, device=device)
    vi = torch.tensor([v for u,v in pos_edges], dtype=torch.long, device=device)
    pos_score = (h[ui] * h[vi]).sum(dim=1)
    # neg
    uj = torch.tensor([u for u,v in neg_edges], dtype=torch.long, device=device)
    vj = torch.tensor([v for u,v in neg_edges], dtype=torch.long, device=device)
    neg_score = (h[uj] * h[vj]).sum(dim=1)
    logits = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([
        torch.ones_like(pos_score),
        torch.zeros_like(neg_score)
    ], dim=0)
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(logits, labels)

# ────────── 学習ループ ──────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # データロード
    (uids, posts, pad_mask, neighbors,
     train_pos, val_pos, pos_set) = load_data(
        POSTS_CSV, EDGES_CSV, POST_DIM, MAX_POSTS
    )
    N = len(uids)
    posts = posts.to(device)
    pad_mask = pad_mask.to(device)

    model = GraphAutoEncoder(
        post_dim=POST_DIM,
        enc_dim=ENC_DIM,
        n_heads=4,
        st_layers=1,
        gnn_layers=GNN_LAYERS,
        dropout=0.1
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    patience = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        h = model(posts, pad_mask, neighbors)  # (N, E)

        # ネガティブサンプリング
        num_pos = len(train_pos)
        neg_needed = int(num_pos * NEG_SAMPLE_RATIO)
        neg_edges = []
        while len(neg_edges) < neg_needed:
            u = random.randrange(N)
            v = random.randrange(N)
            if u==v or (u,v) in pos_set:
                continue
            neg_edges.append((u,v))

        loss = reconstruction_loss(h, train_pos, neg_edges, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # バリデーション
        model.eval()
        with torch.no_grad():
            h_val = model(posts, pad_mask, neighbors)
            # 同数の neg for val
            num_val = len(val_pos)
            neg_val = random.sample(
                [(u,v) for u in range(N) for v in range(N)
                 if u!=v and (u,v) not in pos_set],
                num_val
            )
            val_loss = reconstruction_loss(h_val, val_pos, neg_val, device)

        print(f"Epoch {epoch}/{EPOCHS}  train={loss.item():.4f}  val={val_loss.item():.4f}")

        # 早期停止
        if val_loss.item() < best_val - MIN_DELTA:
            best_val = val_loss.item()
            patience = 0
            os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE)
            print("  ✔ saved best →", MODEL_SAVE)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    train()