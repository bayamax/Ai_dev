#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_post2rw.py  ―  投稿集合 → (学習済み) Post→RW 回帰モデルで
推定 RW を求め，全アカウントとコサイン類似度を取りランキング。
Top-1 ～ Top-500 の Recall を出力します。

・回帰モデル／parse_vec は train4.py 内にある想定
・チェックポイント post2rw.ckpt は train4.py が保存したものを使用
"""

import os, csv, random, argparse, numpy as np, torch
from collections import defaultdict
from tqdm import tqdm
import importlib.util

# ────────────────────────────────────────────────────────
# ★★ ここを実際の学習ファイルのパスに合わせてください ★★
TRAIN_P2R = "/workspace/edit_agent/train/train4.py"
# ────────────────────────────────────────────────────────

# ---------- train4.py から必要なクラス / 関数を import ----------
spec = importlib.util.spec_from_file_location("train4", TRAIN_P2R)
if spec is None:
    raise FileNotFoundError(f"train4.py が見つかりません: {TRAIN_P2R}")
train4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(train4)

Post2RW  = train4.Post2RW
parse_vec = train4.parse_vec
POST_DIM  = train4.POST_DIM

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--vast_dir", default="/workspace/edit_agent/vast")
ap.add_argument("--uid_samples",   type=int, default=1000)
ap.add_argument("--posts_per_uid", type=int, default=30)
args = ap.parse_args()

VAST_DIR   = args.vast_dir
POSTS_CSV  = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY   = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT       = os.path.join(VAST_DIR, "post2rw.ckpt")

K_LIST = [1, 5, 10, 20, 50, 100, 200, 500]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ---------- アカウント行列 & モデル ----------
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=DEV)  # (N,D)
uid2idx  = {u: i for i, u in enumerate(uids)}

rw_dim = acc_mat.size(1)
model = Post2RW(POST_DIM, rw_dim).to(DEV)
ckpt  = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

# ---------- UID → 最新投稿平均 ----------
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_s in tqdm(rdr, desc="Scanning CSV", unit="line"):
        if uid not in acc_dict: continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None: continue
        buf = uid_posts[uid]; buf.append(v)
        if len(buf) > args.posts_per_uid: buf.pop(0)

all_uids = list(uid_posts.keys())
if args.uid_samples and args.uid_samples < len(all_uids):
    eval_uids = random.sample(all_uids, args.uid_samples)
else:
    eval_uids = all_uids

# ---------- 評価 ----------
hits = {k: 0 for k in K_LIST}

with torch.no_grad():
    for uid in eval_uids:
        posts_np = np.asarray(uid_posts[uid], np.float32)
        rep = posts_np.mean(0)                                 # (3072,)
        rep_t = torch.tensor(rep, device=DEV)
        pred = model(rep_t)                                    # (D,)
        pred = pred / (pred.norm() + 1e-8)

        sims = torch.matmul(pred, acc_mat.T).cpu().numpy()
        rank = sims.argsort()[::-1]                            # high→low

        true_idx = uid2idx[uid]
        for K in K_LIST:
            if true_idx in rank[:K]:
                hits[K] += 1

# ---------- 結果 ----------
N = len(eval_uids)
print(f"\nUID={N}  posts≤{args.posts_per_uid}  ckpt=post2rw.ckpt")
for K in K_LIST:
    h = hits[K]
    print(f"Top-{K:3}: {h:4}/{N} = {h/N*100:5.2f}%")