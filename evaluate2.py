#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_avg2rw.py ― Avg2RW モデルの評価スクリプト

* train_avg2rw.py で学習したモデル avg2rw.ckpt をロード
* 各 UID の最新 N 投稿を平均 → Avg2RW でアカウントベクトル予測
* 真アカウントベクトルとコサイン類似度比較 → Recall@K を算出
"""

import os
import csv
import random
import argparse
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import importlib.util

# ─────────── Paths & Defaults ───────────
BASE           = "/workspace/edit_agent"
TRAIN_DIR      = os.path.join(BASE, "train")
VAST_DIR       = os.path.join(BASE, "vast")

POSTS_CSV      = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY       = os.path.join(VAST_DIR, "account_vectors.npy")
MODEL_PY       = os.path.join(TRAIN_DIR, "train4.py")
CKPT_PATH      = os.path.join(VAST_DIR, "avg2rw.ckpt")

# ─────────── CLI Arguments ───────────
parser = argparse.ArgumentParser()
parser.add_argument("--uid_samples",   type=int, default=1000,
                    help="評価に使う UID 件数 (0=全 UID)")
parser.add_argument("--posts_per_uid", type=int, default=None,
                    help="各 UID で使う投稿最大本数 (デフォルトは学習時設定)")
parser.add_argument("--seed",          type=int, default=42,
                    help="乱数シード")
args = parser.parse_args()

# ─────────── Setup ───────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ─────────── Import training module ───────────
spec = importlib.util.spec_from_file_location("train_avg2rw", MODEL_PY)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

parse_vec      = m.parse_vec
POST_DIM       = m.POST_DIM
TRAIN_POSTS_PER_UID = getattr(m, "POSTS_PER_UID", None)
if args.posts_per_uid is None:
    posts_per_uid = TRAIN_POSTS_PER_UID
else:
    posts_per_uid = args.posts_per_uid

# ─────────── Load model ───────────
Avg2RW = m.Avg2RW
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids = list(acc_dict.keys())
rw_dim = next(iter(acc_dict.values())).shape[0]

model = Avg2RW(POST_DIM, rw_dim).to(DEV)
ckpt = torch.load(CKPT_PATH, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

# ─────────── Collect posts per UID ───────────
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f)
    next(rdr)
    for uid, _, vec_s in tqdm(rdr, desc="Loading posts", unit="line"):
        if uid not in acc_dict:
            continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None:
            continue
        buf = uid_posts[uid]
        buf.append(v)
        if posts_per_uid and len(buf) > posts_per_uid:
            buf.pop(0)

all_uids = list(uid_posts.keys())
if args.uid_samples and args.uid_samples < len(all_uids):
    eval_uids = random.sample(all_uids, args.uid_samples)
else:
    eval_uids = all_uids

# ─────────── Prepare true account matrix ───────────
acc_mat = torch.tensor(
    np.stack([acc_dict[u] for u in uids], axis=0),
    dtype=torch.float32, device=DEV
)
uid2idx = {u: i for i, u in enumerate(uids)}

# ─────────── Evaluate Recall@K ───────────
K_LIST = [1, 5, 10, 20, 50, 100, 200, 500]
hits = {k: 0 for k in K_LIST}

with torch.no_grad():
    for uid in tqdm(eval_uids, desc="Evaluating", unit="uid"):
        posts = uid_posts[uid]
        if not posts:
            continue
        arr = np.stack(posts, axis=0)                    # (S, POST_DIM)
        avg = torch.tensor(arr.mean(axis=0), device=DEV) # (POST_DIM,)
        pred_rw = model(avg.unsqueeze(0)).squeeze(0)      # (rw_dim,)
        pred_rw = pred_rw / (pred_rw.norm() + 1e-8)

        sims = torch.matmul(pred_rw, acc_mat.T).cpu().numpy()
        rank = sims.argsort()[::-1]
        true_idx = uid2idx[uid]

        for K in K_LIST:
            if true_idx in rank[:K]:
                hits[K] += 1

# ─────────── Print Results ───────────
N = len(eval_uids)
print(f"\nUID={N}  posts≤{posts_per_uid}")
for K in K_LIST:
    h = hits[K]
    print(f"Top-{K:3}: {h:4}/{N} = {h/N*100:5.2f}%")