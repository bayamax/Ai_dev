#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_mc_ensemble.py
──────────────────────────────────────────────────────────
Monte-Carlo Bagging(Self-Ensemble) で PairClassifier 推論を安定化し、
Recall@K を測定する。

  • TOTAL_POSTS   : 1 UID から取り出す最新投稿上限
  • SUBSET_SIZE   : 1 ラウンドで使う投稿数
  • MC_ROUNDS     : サンプリング回数 (T)
  • K_LIST        : 集計する Top-K 値
"""

import os, csv, random, argparse, numpy as np, torch
from collections import defaultdict

# ---- train3.py を import ----
import importlib.util, sys, inspect
TRAIN3_PATH = "/workspace/edit_agent/train/train3.py"   # 必ず実在パスに！
spec = importlib.util.spec_from_file_location("train3", TRAIN3_PATH)
train3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(train3)
PairClassifier = train3.PairClassifier
parse_vec      = train3.parse_vec

# ---- CLI ----
ap = argparse.ArgumentParser()
ap.add_argument("--vast_dir", default="/workspace/edit_agent/vast")
ap.add_argument("--total_posts",  type=int, default=50,  help="UIDごと最大投稿数")
ap.add_argument("--subset_size",  type=int, default=10,  help="1ラウンド投稿数")
ap.add_argument("--mc_rounds",    type=int, default=100, help="試行回数 T")
ap.add_argument("--uid_samples",  type=int, default=1000)
ap.add_argument("--topk_list",    default="1,5,10,20,50,100,200,500")
args = ap.parse_args()

POSTS_CSV = os.path.join(args.vast_dir, "aggregated_posting_vectors.csv")
ACCS_NPY  = os.path.join(args.vast_dir, "account_vectors.npy")
CKPT      = os.path.join(args.vast_dir, "pair_classifier_masked.ckpt")

POST_DIM  = 3072
K_LIST    = [int(k) for k in args.topk_list.split(",")]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- アカウント行列 & モデル ----
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=DEV)
uid2idx  = {u:i for i,u in enumerate(uids)}

model = PairClassifier(POST_DIM, acc_mat.size(1)).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV)["model_state"])
model.eval()

# ---- UID → 投稿収集 ----
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_s in rdr:
        if uid not in acc_dict: continue
        v = parse_vec(vec_s, POST_DIM)
        if v is None: continue
        buf = uid_posts[uid]; buf.append(v)
        if len(buf) > args.total_posts: buf.pop(0)

eval_uids = random.sample(list(uid_posts.keys()), args.uid_samples)

# ---- 評価 ----
hits = {k:0 for k in K_LIST}

with torch.no_grad():
    for uid in eval_uids:
        posts_np = np.asarray(uid_posts[uid], np.float32)
        if len(posts_np) < args.subset_size:   # 足りない場合は多めに複写して補完
            reps = (args.subset_size // len(posts_np)) + 1
            posts_np = np.tile(posts_np, (reps,1))[:args.subset_size]

        logit_acc = torch.zeros(len(uids), device=DEV)

        for _ in range(args.mc_rounds):
            idx = np.random.choice(len(posts_np), args.subset_size, replace=False)
            subset = posts_np[idx]
            logits_sum = torch.zeros(len(uids), device=DEV)
            for p in subset:
                p_t = torch.tensor(p, dtype=torch.float32, device=DEV)
                logits_sum += model(p_t.unsqueeze(0).expand(acc_mat.size(0), -1),
                                    acc_mat)
            logit_acc += logits_sum

        logit_acc /= args.mc_rounds            # 平均（順位は総和でも同じ）
        rank = torch.argsort(logit_acc, descending=True).cpu().numpy()
        true = uid2idx[uid]
        for K in K_LIST:
            if true in rank[:K]:
                hits[K] += 1

# ---- 結果 ----
N = len(eval_uids)
print(f"UID={N}  TOTAL={args.total_posts}  subset={args.subset_size}  T={args.mc_rounds}")
for K in K_LIST:
    h = hits[K]
    print(f"Top-{K:>3}: {h:4}/{N} = {h/N*100:5.2f}%")