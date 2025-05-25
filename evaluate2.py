#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_post2rw.py
────────────────────────────────────────────────────────
投稿集合 → 回帰モデル → 推定アカウント RW でランキングし
Recall@K を出力 (Top-1 … Top-500)。

  • 学習済みモデル : /workspace/edit_agent/vast/post2rw.ckpt
  • UID ランダム抽出数・投稿本数は CLI で調整
"""

import os, csv, random, argparse, numpy as np, torch
from collections import defaultdict
from tqdm import tqdm
import importlib.util

# ---------- パス設定 ----------
VAST_DIR = "/workspace/edit_agent/vast"
POSTS_CSV = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY  = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT      = os.path.join(VAST_DIR, "post2rw.ckpt")
TRAIN_P2R = "/workspace/edit_agent/train/train_post2rw.py"  # ★実在パスに合わせて変更

# ---------- train_post2rw.py からモデル / parse_vec を import ----------
spec = importlib.util.spec_from_file_location("train_p2r", TRAIN_P2R)
train_p2r = importlib.util.module_from_spec(spec); spec.loader.exec_module(train_p2r)
Post2RW = train_p2r.Post2RW
parse_vec = train_p2r.parse_vec
POST_DIM  = train_p2r.POST_DIM

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--uid_samples",   type=int, default=1000, help="評価 UID 数 (0=全 UID)")
ap.add_argument("--posts_per_uid", type=int, default=30,   help="各 UID で使う最新投稿数")
args = ap.parse_args()

K_LIST = [1,5,10,20,50,100,200,500]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ---------- アカウント行列 & モデル ----------
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=DEV)          # (N,D)
uid2idx  = {u:i for i,u in enumerate(uids)}

rw_dim = acc_mat.size(1)
model = Post2RW(POST_DIM, rw_dim).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
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

if args.uid_samples and args.uid_samples < len(uid_posts):
    eval_uids = random.sample(list(uid_posts.keys()), args.uid_samples)
else:
    eval_uids = list(uid_posts.keys())

# ---------- 評価 ----------
hits = {k:0 for k in K_LIST}

with torch.no_grad():
    for uid in eval_uids:
        posts_np = np.asarray(uid_posts[uid], np.float32)
        rep = posts_np.mean(0)                              # (POST_DIM,)
        rep_t = torch.tensor(rep, device=DEV)
        pred_rw = model(rep_t).cpu()                        # (rw_dim,)

        pred_rw = pred_rw / (pred_rw.norm()+1e-8)
        sims = torch.matmul(pred_rw.to(DEV), acc_mat.T).cpu().numpy()
        rank = sims.argsort()[::-1]                         # high → low

        true = uid2idx[uid]
        for K in K_LIST:
            if true in rank[:K]:
                hits[K]+=1

# ---------- 結果 ----------
N = len(eval_uids)
print(f"\nUID={N}  posts≤{args.posts_per_uid}  model=post2rw.ckpt")
for K in K_LIST:
    h=hits[K]
    print(f"Top-{K:3}: {h:4}/{N} = {h/N*100:5.2f}%")