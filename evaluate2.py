#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate2.py
──────────────────────────────────────────────────────────
PairClassifier で 1 投稿ずつ logit を合算し、
Recall@K (K = 1,5,10,20,50,100,200,500) を計測。

※ 必ず train3.py と同じディレクトリで実行してください。
"""

import os
import sys
import csv
import random
import argparse
from collections import defaultdict

import numpy as np
import torch

# ── import PairClassifier / parse_vec from train3.py ─────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from train3 import PairClassifier, parse_vec
except ImportError as e:
    raise ImportError(
        "train3.py が同じディレクトリに見つかりません。"
        "ファイル名・配置を確認してください。"
    ) from e

# ── パス & デフォルト設定 ─────────────────────────────────
DEF_VAST = "/workspace/edit_agent/vast"

ap = argparse.ArgumentParser()
ap.add_argument("--vast_dir", default=DEF_VAST, help="VAST ディレクトリ")
ap.add_argument("--posts_per_uid", type=int, default=50,
                help="UID あたり使用する投稿上限")
ap.add_argument("--uid_samples", type=int, default=1000,
                help="評価に使う UID 数 (None=全 UID)")
args = ap.parse_args()

POSTS_CSV = os.path.join(args.vast_dir, "aggregated_posting_vectors.csv")
ACCS_NPY  = os.path.join(args.vast_dir, "account_vectors.npy")
CKPT      = os.path.join(args.vast_dir, "pair_classifier_masked.ckpt")  # ← train3 の出力

K_LIST    = [1, 5, 10, 20, 50, 100, 200, 500]
POST_DIM  = 3072

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── アカウント行列 & モデル読み込み ────────────────────────
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=DEV)          # (N,D)
uid2idx  = {u: i for i, u in enumerate(uids)}

model = PairClassifier(post_dim=POST_DIM, rw_dim=acc_mat.size(1)).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── UID→最新投稿を収集 ───────────────────────────────────
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_s in rdr:
        if uid not in acc_dict:
            continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None:
            continue
        lst = uid_posts[uid]
        lst.append(v)
        if len(lst) > args.posts_per_uid:
            lst.pop(0)                      # 常に最新投稿だけ残す

eval_uids = (random.sample(list(uid_posts.keys()), args.uid_samples)
             if args.uid_samples else list(uid_posts.keys()))

# ── Recall@K 評価 ──────────────────────────────────────
hits = {k: 0 for k in K_LIST}

with torch.no_grad():
    for uid in eval_uids:
        posts = uid_posts[uid]
        logit_sum = torch.zeros(len(uids), device=DEV)

        # 投稿ごとに logit 加算
        for p in posts:
            p_t = torch.tensor(p, dtype=torch.float32, device=DEV)
            logit_sum += model(
                p_t.unsqueeze(0).expand(acc_mat.size(0), -1), acc_mat
            )

        rank = torch.argsort(logit_sum, descending=True).cpu().numpy()
        true = uid2idx[uid]

        for K in K_LIST:
            if true in rank[:K]:
                hits[K] += 1

# ── 結果表示 ───────────────────────────────────────────
N = len(eval_uids)
print(f"Samples={N}  posts≤{args.posts_per_uid}")
for K in K_LIST:
    hit = hits[K]
    print(f"Top-{K:3d}: {hit:4d}/{N} = {hit/N*100:5.2f}%")