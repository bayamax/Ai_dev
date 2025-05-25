#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_quick.py
────────────────────────────────────────────────────────
• たった数アカウント (default 20 UID) だけを抽出して
  投稿ごと logits を合算 → Recall@K を計算
• tqdm で CSV 読み取り進捗を表示
----------------------------------------------------------------
引数:
  --uid_samples     評価 UID 数           (default 20)
  --posts_per_uid   1 UID あたり投稿上限  (default 30)
  --vast_dir        データ置き場           (default /workspace/edit_agent/vast)
----------------------------------------------------------------
"""

import os, csv, random, argparse, numpy as np, torch
from collections import defaultdict
from tqdm import tqdm
import importlib.util, sys

# --- PairClassifier / parse_vec を train3.py から取ってくる ---
TRAIN3 = "/workspace/edit_agent/train/train3.py"   # ← 実在パスに合わせて変更
spec = importlib.util.spec_from_file_location("train3", TRAIN3)
train3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(train3)
PairClassifier, parse_vec = train3.PairClassifier, train3.parse_vec

# ------------- CLI -----------------
ap = argparse.ArgumentParser()
ap.add_argument("--vast_dir", default="/workspace/edit_agent/vast")
ap.add_argument("--uid_samples",   type=int, default=20)
ap.add_argument("--posts_per_uid", type=int, default=30)
args = ap.parse_args()

POSTS_CSV  = os.path.join(args.vast_dir, "aggregated_posting_vectors.csv")
ACCS_NPY   = os.path.join(args.vast_dir, "account_vectors.npy")
CKPT       = os.path.join(args.vast_dir, "pair_classifier_masked.ckpt")
POST_DIM   = 3072
K_LIST     = [1, 5, 10, 20, 50]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ------------- アカウント行列 & モデル -------------
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=DEV)
uid2idx  = {u:i for i,u in enumerate(uids)}

model = PairClassifier(POST_DIM, acc_mat.size(1)).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV)["model_state"])
model.eval()

# ------------- UID → 最新投稿収集 -------------
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_s in tqdm(rdr, desc="Scanning CSV", unit="line"):
        if uid not in acc_dict: continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None: continue
        buf = uid_posts[uid]; buf.append(v)
        if len(buf) > args.posts_per_uid: buf.pop(0)

if len(uid_posts) < args.uid_samples:
    print(f"⚠ UID {len(uid_posts)} 件しか見つかりませんでした")
eval_uids = random.sample(list(uid_posts.keys()), args.uid_samples)

# ------------- 推論 -------------
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
            if true in rank[:K]: hits[K]+=1

# ------------- 結果 -------------
print(f"\nUID={len(eval_uids)}  posts≤{args.posts_per_uid}")
for K in K_LIST:
    h=hits[K]; print(f"Top-{K:3}: {h:3}/{len(eval_uids)} = {h/len(eval_uids)*100:5.2f}%")