#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_compact_attn.py  ― CompactAttnAggregator モデルの評価スクリプト

* 学習済み CompactAttnAggregator（compact_attn.ckpt）をロード
* 各 UID の最新 N 投稿からアカウントベクトルを推定
* 真のアカウントベクトルとコサイン類似度比較 → Top-K Recall を算出
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

# ─────────── 固定パス ───────────
BASE          = "/workspace/edit_agent"
TRAIN_DIR     = os.path.join(BASE, "train")
VAST_DIR      = os.path.join(BASE, "vast")

POSTS_CSV     = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY      = os.path.join(VAST_DIR, "account_vectors.npy")
MODEL_PY      = os.path.join(TRAIN_DIR, "train5.py")
CKPT_PATH     = os.path.join(VAST_DIR, "compact_attn.ckpt")

# ─────────── CLI 引数 ───────────
parser = argparse.ArgumentParser()
parser.add_argument("--uid_samples",   type=int, default=1000,
                    help="評価する UID の件数 (0=全 UID)")
parser.add_argument("--posts_per_uid", type=int, default=None,
                    help="各 UID で使う投稿最大本数 (デフォルトは学習時と同じ)")
parser.add_argument("--seed",          type=int, default=42,
                    help="乱数シード")
args = parser.parse_args()

# ─────────── 環境準備 ───────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

DEV    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_LIST = [1,5,10,20,50,100,200,500]
print("[Device]", DEV)

# ─────────── モジュール import ───────────
spec = importlib.util.spec_from_file_location("mtrain", MODEL_PY)
mtrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtrain)

parse_vec        = mtrain.parse_vec
POST_DIM         = mtrain.POST_DIM
TRAIN_POSTS_PER_UID = getattr(mtrain, "POSTS_PER_UID", None)

# 学習時と同じ投稿上限を使う
posts_per_uid = args.posts_per_uid or TRAIN_POSTS_PER_UID

# ─────────── モデル定義 & 読み込み ───────────
CompactAttnAggregator = mtrain.CompactAttnAggregator
acc_dict   = np.load(ACCS_NPY, allow_pickle=True).item()
uids       = list(acc_dict.keys())
rw_dim     = next(iter(acc_dict.values())).shape[0]

# instantiate & load
model = CompactAttnAggregator(
    d_in=POST_DIM,
    d_out=rw_dim,
    d_model=mtrain.D_MODEL,
    n_heads=mtrain.N_HEADS,
    n_layers=mtrain.N_LAYERS,
    dropout=mtrain.DROPOUT
).to(DEV)
ckpt = torch.load(CKPT_PATH, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

# ─────────── 投稿データ収集 ───────────
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
        if posts_per_uid and len(buf) > posts_per_uid:
            buf.pop(0)

# ─────────── UID サンプリング ───────────
all_uids = list(uid_posts.keys())
if args.uid_samples and args.uid_samples < len(all_uids):
    eval_uids = random.sample(all_uids, args.uid_samples)
else:
    eval_uids = all_uids

# ─────────── 真アカウント行列作成 ───────────
acc_mat = torch.tensor(
    np.stack([acc_dict[u] for u in uids], axis=0),
    dtype=torch.float32, device=DEV
)
uid2idx = {u:i for i,u in enumerate(uids)}

# ─────────── 推論＆評価 ───────────
hits = {k: 0 for k in K_LIST}

with torch.no_grad():
    for uid in tqdm(eval_uids, desc="Evaluate"):
        posts = uid_posts[uid]
        if not posts:
            continue
        arr = np.asarray(posts, np.float32)                           # (S, POST_DIM)
        t   = torch.as_tensor(arr, device=DEV).unsqueeze(0)            # (1, S, D)
        mask = torch.zeros((1, arr.shape[0]), dtype=torch.bool, device=DEV)  # no padding

        pred_rw = model(t, mask)[0]                                    # (rw_dim,)
        pred_rw = pred_rw / (pred_rw.norm() + 1e-8)

        sims = torch.matmul(pred_rw, acc_mat.T).cpu().numpy()
        rank = sims.argsort()[::-1]

        true_idx = uid2idx[uid]
        for K in K_LIST:
            if true_idx in rank[:K]:
                hits[K] += 1

# ─────────── 結果出力 ───────────
N = len(eval_uids)
print(f"\nUID={N}  posts≤{posts_per_uid}")
for K in K_LIST:
    h = hits[K]
    print(f"Top-{K:3}: {h:4}/{N} = {h/N*100:5.2f}%")