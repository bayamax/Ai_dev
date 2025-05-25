#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_quick.py  ―  投稿ごと logits 合算で Recall@K を測る超軽量版
  * CSV 走査に tqdm 進捗バー
  * デフォルト 20 UID × 最新 30 投稿
  * Top-1,5,10,20,50,100,200,500 を集計
"""

import os, csv, random, argparse, numpy as np, torch
from collections import defaultdict
from tqdm import tqdm
import importlib.util

# ─────────── train3.py からモデル/関数を import ───────────
TRAIN3_PATH = "/workspace/edit_agent/train/train3.py"   # 必ず存在するパスに！
spec = importlib.util.spec_from_file_location("train3", TRAIN3_PATH)
train3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(train3)
PairClassifier, parse_vec = train3.PairClassifier, train3.parse_vec

# ─────────── CLI ───────────
ap = argparse.ArgumentParser()
ap.add_argument("--vast_dir", default="/workspace/edit_agent/vast")
ap.add_argument("--uid_samples",   type=int, default=20,
                help="評価 UID 件数 (0=全 UID)")
ap.add_argument("--posts_per_uid", type=int, default=30,
                help="各 UID で使う最新投稿本数")
args = ap.parse_args()

POSTS_CSV = os.path.join(args.vast_dir, "aggregated_posting_vectors.csv")
ACCS_NPY  = os.path.join(args.vast_dir, "account_vectors.npy")
CKPT      = os.path.join(args.vast_dir, "pair_classifier_masked.ckpt")

POST_DIM = 3072
K_LIST   = [1, 5, 10, 20, 50, 100, 200, 500]   # ← Top-500 まで拡張

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ─────────── アカウント行列 & モデル ───────────
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=DEV)
uid2idx  = {u:i for i,u in enumerate(uids)}

model = PairClassifier(POST_DIM, acc_mat.size(1)).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV)["model_state"])
model.eval()

# ─────────── UID → 最新投稿収集 ───────────
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_s in tqdm(rdr, desc="Scanning CSV", unit="line"):
        if uid not in acc_dict:
            continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None:
            continue
        buf = uid_posts[uid]; buf.append(v)
        if len(buf) > args.posts_per_uid:
            buf.pop(0)

all_uid_cnt = len(uid_posts)
if args.uid_samples and args.uid_samples < all_uid_cnt:
    eval_uids = random.sample(list(uid_posts.keys()), args.uid_samples)
else:
    eval_uids = list(uid_posts.keys())

# ─────────── 評価 ───────────
hits = {k:0 for k in K_LIST}
with torch.no_grad():
    for uid in eval_uids:
        logit_sum = torch.zeros(len(uids), device=DEV)
        for p in uid_posts[uid]:
            v = torch.tensor(p, dtype=torch.float32, device=DEV)
            logit_sum += model(v.unsqueeze(0).expand(acc_mat.size(0), -1), acc_mat)
        rank = torch.argsort(logit_sum, descending=True).cpu().numpy()
        true = uid2idx[uid]
        for K in K_LIST:
            if true in rank[:K]:
                hits[K] += 1

# ─────────── 出力 ───────────
N = len(eval_uids)
print(f"\nUID={N}  posts≤{args.posts_per_uid}")
for K in K_LIST:
    h = hits[K]
    print(f"Top-{K:3}: {h:4}/{N} = {h/N*100:5.2f}%")