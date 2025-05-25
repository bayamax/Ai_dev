#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_mc_ensemble_gpu.py
Monte-Carlo Self-Ensemble を GPU フル活用で実装。
  • subset_size 本を 1 バッチで推論
  • acc_mat を ACC_CHUNK 件ずつに分割して VRAM を節約
  • 投稿×アカウント ペアをすべて GPU 上で計算
"""

import os, csv, random, argparse, numpy as np, torch
from collections import defaultdict
import importlib.util, sys

# ─────────── train3.py をロード ───────────
TRAIN3_PATH = "/workspace/edit_agent/train/train3.py"
spec = importlib.util.spec_from_file_location("train3", TRAIN3_PATH)
train3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(train3)
PairClassifier, parse_vec = train3.PairClassifier, train3.parse_vec

# ─────────── CLI ───────────
ap = argparse.ArgumentParser()
ap.add_argument("--vast_dir", default="/workspace/edit_agent/vast")
ap.add_argument("--total_posts", type=int, default=50)
ap.add_argument("--subset_size", type=int, default=10)
ap.add_argument("--mc_rounds",   type=int, default=100)
ap.add_argument("--uid_samples", type=int, default=1000)
ap.add_argument("--acc_chunk",   type=int, default=2000,
                help="アカウント行列を何件ずつ GPU へ流すか")
args = ap.parse_args()

POSTS_CSV = os.path.join(args.vast_dir, "aggregated_posting_vectors.csv")
ACCS_NPY  = os.path.join(args.vast_dir, "account_vectors.npy")
CKPT      = os.path.join(args.vast_dir, "pair_classifier_masked.ckpt")

POST_DIM  = 3072
K_LIST    = [1,5,10,20,50,100,200,500]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ─────────── アカウント行列 & モデル ───────────
acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat_cpu = np.stack([acc_dict[u] for u in uids]).astype(np.float32)   # (N,D)
uid2idx  = {u:i for i,u in enumerate(uids)}

model = PairClassifier(POST_DIM, acc_mat_cpu.shape[1]).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV)["model_state"])
model.eval()

# ─────────── UID -> 投稿収集 ───────────
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding='utf-8') as f:
    rdr=csv.reader(f); next(rdr)
    for uid, _, vec in rdr:
        if uid not in acc_dict: continue
        v=parse_vec(vec, POST_DIM)
        if v is None: continue
        buf=uid_posts[uid]; buf.append(v)
        if len(buf)>args.total_posts: buf.pop(0)

eval_uids = random.sample(list(uid_posts.keys()), args.uid_samples)
ACC_CHUNK = args.acc_chunk
SUBSET    = args.subset_size
T         = args.mc_rounds

# ─────────── 評価 ───────────
hits = {k:0 for k in K_LIST}
with torch.no_grad():
    for uid in eval_uids:
        posts_np = np.asarray(uid_posts[uid], np.float32)
        # 投稿不足時に複写して補完
        if len(posts_np) < SUBSET:
            reps = SUBSET // len(posts_np) + 1
            posts_np = np.tile(posts_np, (reps,1))[:SUBSET]

        logit_acc = torch.zeros(len(uids), device=DEV)

        for _ in range(T):
            idx = np.random.choice(posts_np.shape[0], SUBSET, replace=False)
            batch_posts = torch.tensor(posts_np[idx], device=DEV)        # (P,Dp)

            # アカウント行列をチャンクで GPU 転送
            offs = 0
            while offs < len(uids):
                acc_chunk = torch.tensor(
                    acc_mat_cpu[offs:offs+ACC_CHUNK], device=DEV
                )                                                       # (C,D)
                # broadcast: (P,1,Dp) → (P,C,Dp); (1,C,D) → (P,C,D)
                p_rep = batch_posts.unsqueeze(1).expand(-1, acc_chunk.size(0), -1)
                a_rep = acc_chunk.unsqueeze(0).expand(batch_posts.size(0), -1, -1)
                # reshape to 2-D batch
                p_flat = p_rep.reshape(-1, POST_DIM)
                a_flat = a_rep.reshape(-1, acc_chunk.size(1))
                logits = model(p_flat, a_flat).view(SUBSET, -1)          # (P,C)
                logit_acc[offs:offs+ACC_CHUNK] += logits.sum(0)
                offs += ACC_CHUNK

        logit_acc /= T
        rank = torch.argsort(logit_acc, descending=True).cpu().numpy()
        true = uid2idx[uid]
        for K in K_LIST:
            if true in rank[:K]:
                hits[K]+=1

# ─────────── 出力 ───────────
N=len(eval_uids)
print(f"UID={N}  total_posts={args.total_posts}  subset={SUBSET}  T={T}")
for K in K_LIST:
    h=hits[K]; print(f"Top-{K:>3}: {h:4}/{N} = {h/N*100:5.2f}%")