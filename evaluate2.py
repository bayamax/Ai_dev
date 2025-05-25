#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate2.py   ← 評価専用
投稿集合 → Post2RW モデル → RefineNet (Attention-Pooling+MLP)
                    ↓
             推定アカウント RW
と真のアカウント RW を比較し、Recall@K (K=1,5,10,20,50,100,200,500) を出力
"""

import os
import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import importlib.util

# ───────── 固定パス ─────────
BASE       = "/workspace/edit_agent"
TRAIN_DIR  = os.path.join(BASE, "train")
VAST       = os.path.join(BASE, "vast")

POSTS_CSV  = os.path.join(VAST, "aggregated_posting_vectors.csv")
ACCS_NPY   = os.path.join(VAST, "account_vectors.npy")

POST2RW_PY = os.path.join(TRAIN_DIR, "train4.py")
POST2RW_CK = os.path.join(VAST, "post2rw.ckpt")
REFINE_CK  = os.path.join(VAST, "refine_rw_agg.ckpt")

# ───────── CLI 引数 ─────────
parser = argparse.ArgumentParser()
parser.add_argument("--uid_samples",   type=int, default=1000,
                    help="評価する UID 数 (0=全 UID)")
parser.add_argument("--posts_per_uid", type=int, default=30,
                    help="各 UID で使う投稿の最大本数")
args = parser.parse_args()

K_LIST = [1, 5, 10, 20, 50, 100, 200, 500]
DEV    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ───────── Post2RW 定義を import ─────────
spec = importlib.util.spec_from_file_location("p2r", POST2RW_PY)
p2r  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2r)

Post2RW    = p2r.Post2RW
parse_vec  = p2r.parse_vec
POST_DIM   = p2r.POST_DIM
HIDDEN_DIMS= p2r.HIDDEN_DIMS
DROPOUT    = p2r.DROPOUT

# ───────── 真アカウント RW を読み込み ─────────
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
rw_dim   = next(iter(acc_dict.values())).shape[0]

# ───────── モデル読み込み ─────────
# 1) Post2RW
post2rw = Post2RW(POST_DIM, rw_dim,
                  hidden=HIDDEN_DIMS, drop=DROPOUT).to(DEV)
post2rw.load_state_dict(torch.load(POST2RW_CK, map_location=DEV)["model"])
post2rw.eval()

# 2) RefineNet を自前で定義
class Aggregator(nn.Module):
    def __init__(self, in_dim=rw_dim, d=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d, bias=False),
            nn.LayerNorm(d)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=d*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.cls = nn.Parameter(torch.randn(1,1,d))
    def forward(self, x, mask):
        B = x.size(0)
        z = self.proj(x)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1)
        if mask is not None:
            pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([pad, mask], dim=1)
        h = self.enc(z, src_key_padding_mask=mask)
        return h[:,0]

class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg  = Aggregator()
        self.head = nn.Linear(512, rw_dim)
    def forward(self, x, mask):
        return self.head(self.agg(x, mask))

refine = RefineNet().to(DEV)
refine_state = torch.load(REFINE_CK, map_location=DEV)
refine.load_state_dict(refine_state["model"] if "model" in refine_state else refine_state)
refine.eval()

# ───────── 投稿データを UID ごとに収集 ─────────
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f)
    next(rdr)
    for uid, _, vec_s in tqdm(rdr, desc="Load posts", unit="line"):
        if uid not in acc_dict:
            continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None:
            continue
        buf = uid_posts[uid]
        buf.append(v)
        if len(buf) > args.posts_per_uid:
            buf.pop(0)

# サブサンプリングする UID を決定
if 0 < args.uid_samples < len(uid_posts):
    eval_uids = random.sample(list(uid_posts.keys()), args.uid_samples)
else:
    eval_uids = list(uid_posts.keys())

# 真 RW 行列と索引マップ
acc_mat = torch.tensor(
    np.stack([acc_dict[u] for u in uids], axis=0),
    dtype=torch.float32, device=DEV
)  # (num_uids, rw_dim)
uid2idx = {u: i for i, u in enumerate(uids)}

# ───────── 推論＆評価 ─────────
hits = {k: 0 for k in K_LIST}

with torch.no_grad():
    for uid in tqdm(eval_uids, desc="Evaluate"):
        posts = np.asarray(uid_posts[uid], np.float32)
        t_post = torch.as_tensor(posts, device=DEV)         # (S, POST_DIM)

        # 投稿ごとに一旦 RW を推定
        pred_set = post2rw(t_post).detach()                 # (S, rw_dim)

        # Attention-Pooling + MLP で集合を 1 ベクトルに
        S = pred_set.size(0)
        mask = torch.zeros((1, S), dtype=torch.bool, device=DEV)
        pred_rw = refine(pred_set.unsqueeze(0), mask)[0]    # (rw_dim,)

        # 正規化してコサイン類似度を計算
        pred_rw = pred_rw / (pred_rw.norm() + 1e-8)
        sims = torch.matmul(pred_rw, acc_mat.T).cpu().numpy()
        rank = sims.argsort()[::-1]

        true_idx = uid2idx[uid]
        for K in K_LIST:
            if true_idx in rank[:K]:
                hits[K] += 1

# ───────── 結果を表示 ─────────
N = len(eval_uids)
print(f"\nUID={N}  posts≤{args.posts_per_uid}")
for K in K_LIST:
    h = hits[K]
    print(f"Top-{K:3}: {h:4}/{N} = {h/N*100:5.2f}%")