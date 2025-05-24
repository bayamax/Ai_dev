#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_pair_classifier.py
最新の PairClassifier (BCE 0/1 学習 + ノイズ負例) で
Recall@K  ―  Top-1,5,10,20,50,100,200,500 ― を計測する。
進捗バーなし・チェックポイントを自動ロード。
"""

import os
import csv
import numpy as np
import torch

# ★ train スクリプトのファイル名に合わせてインポートを調整してください
#   例:  train_pair_classifier_stream.py なら ↓そのまま
from train_pair_classifier_stream import PairClassifier, parse_vec

# ────────── パス & 設定 ──────────
VAST_DIR      = "/workspace/edit_agent/vast"
POSTS_CSV     = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY   = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH     = os.path.join(VAST_DIR, "pair_classifier_rw_stream.ckpt")

POST_DIM      = 3072
NUM_SAMPLES   = 1000          # 評価に使う投稿数
K_LIST        = [1, 5, 10, 20, 50, 100, 200, 500]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────── モデル & アカウント行列読み込み ──────────
rw_dict = np.load(ACCOUNT_NPY, allow_pickle=True).item()
account_ids = list(rw_dict.keys())
acc2idx = {uid: i for i, uid in enumerate(account_ids)}
acc_mat = torch.tensor(
    np.stack([rw_dict[uid] for uid in account_ids], axis=0),
    dtype=torch.float32,
    device=DEVICE
)                                   # (num_accounts, D)

model = PairClassifier(post_dim=POST_DIM, rw_dim=acc_mat.size(1)).to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ────────── 投稿サンプリング ──────────
samples = []
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_str in rdr:
        if uid not in rw_dict:
            continue
        vec = parse_vec(vec_str, POST_DIM)
        if vec is None:
            continue
        samples.append((uid, vec))
        if len(samples) >= NUM_SAMPLES:
            break

# ────────── 評価 ──────────
counters = {K: 0 for K in K_LIST}
with torch.no_grad():
    for uid, vec in samples:
        post  = torch.tensor(vec, dtype=torch.float32, device=DEVICE)
        posts = post.unsqueeze(0).expand(acc_mat.size(0), -1)   # (N,D)
        logits = model(posts, acc_mat)                          # (N,)
        topk = torch.topk(logits, k=max(K_LIST)).indices.cpu().tolist()
        correct_idx = acc2idx[uid]
        for K in K_LIST:
            if correct_idx in topk[:K]:
                counters[K] += 1

# ────────── 結果表示 ──────────
N = len(samples)
print(f"Evaluated on {N} samples  (checkpoint: {os.path.basename(CKPT_PATH)})")
for K in K_LIST:
    cnt = counters[K]
    print(f"Top-{K:3d}: {cnt:4d}/{N} = {cnt/N*100:5.2f}%")