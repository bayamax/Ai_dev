#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate2.py
────────────────────────────────────────────────────────────
PairClassifier（train3.py 内）で各投稿の logits を合算し，
Recall@K を計測する評価スクリプト。

※ train3.py の正確な場所を TRAIN3_PATH に設定してください。
"""

import os
import sys
import csv
import random
import argparse
import importlib.util
from collections import defaultdict

import numpy as np
import torch

# ───────── ここだけ環境に合わせて書き換えてください ──────────
TRAIN3_PATH = "/workspace/edit_agent/train/train3.py"   # train3.py のフルパス
# ────────────────────────────────────────────────────────────

# ---------- train3.py を動的 import ----------
spec = importlib.util.spec_from_file_location("train3", TRAIN3_PATH)
if spec is None:
    raise FileNotFoundError(f"train3.py not found at {TRAIN3_PATH}")
train3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train3)

PairClassifier = train3.PairClassifier
parse_vec      = train3.parse_vec
VAL_RATIO      = train3.VAL_RATIO           # 使う場合は取り出せる

# ---------- CLI ----------
DEF_VAST = "/workspace/edit_agent/vast"
ap = argparse.ArgumentParser()
ap.add_argument("--vast_dir", default=DEF_VAST)
ap.add_argument("--posts_per_uid", type=int, default=50)
ap.add_argument("--uid_samples",   type=int, default=1000)
args = ap.parse_args()

POSTS_CSV  = os.path.join(args.vast_dir, "aggregated_posting_vectors.csv")
ACCS_NPY   = os.path.join(args.vast_dir, "account_vectors.npy")
CKPT       = os.path.join(args.vast_dir, "pair_classifier_masked.ckpt")

POST_DIM   = 3072
K_LIST     = [1, 5, 10, 20, 50, 100, 200, 500]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- アカウント行列 & モデル ----------
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=device)        # (N,D)
uid2idx  = {u: i for i, u in enumerate(uids)}

model = PairClassifier(POST_DIM, acc_mat.size(1)).to(device)
ckpt  = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---------- UID → 最新投稿収集 ----------
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_s in rdr:
        if uid not in acc_dict:
            continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None:
            continue
        buf = uid_posts[uid]
        buf.append(v)
        if len(buf) > args.posts_per_uid:
            buf.pop(0)                       # 常に最新 posts_per_uid 本だけ残す

eval_uids = (random.sample(list(uid_posts.keys()), args.uid_samples)
             if args.uid_samples else list(uid_posts.keys()))

# ---------- 評価 ----------
hits = {k: 0 for k in K_LIST}

with torch.no_grad():
    for uid in eval_uids:
        posts = uid_posts[uid]
        logit_sum = torch.zeros(len(uids), device=device)

        for p in posts:
            p_t = torch.tensor(p, dtype=torch.float32, device=device)
            logit_sum += model(
                p_t.unsqueeze(0).expand(acc_mat.size(0), -1),
                acc_mat
            )

        rank = torch.argsort(logit_sum, descending=True).cpu().numpy()
        true_idx = uid2idx[uid]
        for K in K_LIST:
            if true_idx in rank[:K]:
                hits[K] += 1

# ---------- 出力 ----------
N = len(eval_uids)
print(f"Samples={N}  posts≤{args.posts_per_uid}")
for K in K_LIST:
    hit = hits[K]
    print(f"Top-{K:3d}: {hit:4d}/{N} = {hit/N*100:5.2f}%")