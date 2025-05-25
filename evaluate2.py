#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate2.py   ← 評価専用
投稿集合 → Post2RW → RefineNet (Attention-Pooling+MLP)
                    ↓
             推定アカウント RW
と真 RW を比べ、Recall@K (K=1,5,10,20,50,100,200,500) を出力
"""

import os, csv, random, argparse, numpy as np, torch
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
REFINE_PY  = os.path.join(TRAIN_DIR, "train5.py")   # train_refine_rw_agg.py
POST2RW_CK = os.path.join(VAST, "post2rw.ckpt")
REFINE_CK  = os.path.join(VAST, "refine_rw_agg.ckpt")

# ───────── CLI ─────────
ap = argparse.ArgumentParser()
ap.add_argument("--uid_samples",   type=int, default=1000,
                help="評価 UID 数 (0=全 UID)")
ap.add_argument("--posts_per_uid", type=int, default=30,
                help="各 UID で使う投稿上限")
args = ap.parse_args()

K_LIST = [1,5,10,20,50,100,200,500]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ───────── モジュール import ─────────
def import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

p2r  = import_from(POST2RW_PY, "p2r")
refm = import_from(REFINE_PY,  "refm")

Post2RW, parse_vec, POST_DIM = p2r.Post2RW, p2r.parse_vec, p2r.POST_DIM
RefineNet   = refm.RefineNet                      # Aggregator+MLP 一体

# ───────── モデル読み込み ─────────
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
rw_dim   = next(iter(acc_dict.values())).shape[0]

post2rw = Post2RW(POST_DIM, rw_dim,
                  hidden=p2r.HIDDEN_DIMS, drop=p2r.DROPOUT).to(DEV)
post2rw.load_state_dict(torch.load(POST2RW_CK, map_location=DEV)["model"])
post2rw.eval()

refine = RefineNet().to(DEV)
refine.load_state_dict(torch.load(REFINE_CK, map_location=DEV)["model"])
refine.eval()

# ───────── 投稿データ収集 ─────────
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_s in tqdm(rdr, desc="Load posts", unit="line"):
        if uid not in acc_dict: continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None: continue
        buf = uid_posts[uid]; buf.append(v)
        if len(buf) > args.posts_per_uid: buf.pop(0)

if args.uid_samples and args.uid_samples < len(uid_posts):
    eval_uids = random.sample(list(uid_posts.keys()), args.uid_samples)
else:
    eval_uids = list(uid_posts.keys())

# ───────── 真 RW 行列 ─────────
acc_mat = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                       dtype=torch.float32, device=DEV)
uid2idx = {u:i for i,u in enumerate(uids)}

# ───────── 推論 & 評価 ─────────
hits = {k:0 for k in K_LIST}

with torch.no_grad():
    for uid in tqdm(eval_uids, desc="Evaluate"):
        posts = np.asarray(uid_posts[uid], np.float32)
        t_post = torch.as_tensor(posts, device=DEV)
        pred_set = post2rw(t_post).detach()                # (S, rw_dim)

        # Attention Pooling
        S = pred_set.size(0)
        mask = torch.zeros(1,S, dtype=torch.bool, device=DEV)
        pred_rw = refine(pred_set.unsqueeze(0), mask)[0]   # (rw_dim,)

        pred_rw = pred_rw / (pred_rw.norm()+1e-8)
        sims = torch.matmul(pred_rw, acc_mat.T).cpu().numpy()
        rank = sims.argsort()[::-1]

        true_idx = uid2idx[uid]
        for K in K_LIST:
            if true_idx in rank[:K]:
                hits[K] += 1

# ───────── 結果 ─────────
N = len(eval_uids)
print(f"\nUID={N}  posts≤{args.posts_per_uid}")
for K in K_LIST:
    h=hits[K]
    print(f"Top-{K:3}: {h:4}/{N} = {h/N*100:5.2f}%")