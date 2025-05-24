#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_top1.py ― PairClassifier で
投稿ベクトルから1万アカウントのうち正解アカウントを
Top-1に選べる確率(Recall@1)を計測する
"""

import os
import csv
import random
import numpy as np
import torch

from train_pair_classifier_stream import PairClassifier, parse_vec  # パスは適宜調整

# ─────────── 設定 ───────────
VAST_DIR      = "/workspace/edit_agent/vast"
POSTS_CSV     = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY   = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH     = os.path.join(VAST_DIR, "pair_classifier_rw_stream.ckpt")
POST_DIM      = 3072
NUM_SAMPLES   = 1000   # 評価に使う投稿数
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) モデル読み込み
rw_dict = np.load(ACCOUNT_NPY, allow_pickle=True).item()
account_ids = list(rw_dict.keys())
acc_mat = torch.tensor(
    np.stack([rw_dict[uid] for uid in account_ids], axis=0),
    dtype=torch.float32,
    device=DEVICE
)  # (num_accounts, D)

model = PairClassifier(
    post_dim=POST_DIM,
    rw_dim=acc_mat.size(1)
).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 2) 投稿サンプリング
samples = []
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f)
    next(rdr)
    for uid, _, vec_str in rdr:
        if uid not in rw_dict:
            continue
        vec = parse_vec(vec_str, POST_DIM)
        if vec is None:
            continue
        samples.append((uid, vec))
        if len(samples) >= NUM_SAMPLES:
            break

# 3) 評価ループ（プログレスバーなし）
correct_top1 = 0
with torch.no_grad():
    for uid, vec in samples:
        post   = torch.tensor(vec, dtype=torch.float32, device=DEVICE)
        posts  = post.unsqueeze(0).expand(acc_mat.size(0), -1)  # (num_accounts, D)
        logits = model(posts, acc_mat)                         # (num_accounts,)
        pred_idx = torch.argmax(logits).item()
        if account_ids[pred_idx] == uid:
            correct_top1 += 1

recall_at_1 = correct_top1 / len(samples)
print(f"Recall@1 (Top-1 正解率): {recall_at_1*100:.2f}% over {len(samples)} samples")
