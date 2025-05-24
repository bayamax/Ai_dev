#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_pair_bce.py  ―  PairClassifier (BCE) 用 Recall@K 評価
"""

import os, csv, numpy as np, torch
from train_pair_classifier_stream import PairClassifier, parse_vec  # パス調整

# ----- paths & params -----
VAST_DIR    = "/workspace/edit_agent/vast"
POSTS_CSV   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH   = os.path.join(VAST_DIR, "pair_classifier_rw_stream.ckpt")
POST_DIM    = 3072
NUM_SAMPLES = 1000
K_LIST      = [1, 5, 10, 20, 50, 100, 200, 500]
DEV         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- load model & account matrix -----
acc_dict  = np.load(ACCOUNT_NPY, allow_pickle=True).item()
uids      = list(acc_dict.keys())
uid2idx   = {u:i for i,u in enumerate(uids)}
acc_mat   = torch.tensor(np.stack([acc_dict[u] for u in uids]), dtype=torch.float32, device=DEV)

model = PairClassifier(POST_DIM, acc_mat.size(1)).to(DEV)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEV)["model_state"])
model.eval()

# ----- sample posts -----
samples = []
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_str in rdr:
        if uid not in acc_dict: continue
        v = parse_vec(vec_str, POST_DIM);  # -> np.ndarray
        if v is None: continue
        samples.append((uid, v))
        if len(samples) >= NUM_SAMPLES: break

# ----- evaluate -----
hits = {K:0 for K in K_LIST}
with torch.no_grad():
    for uid, v in samples:
        post = torch.tensor(v, dtype=torch.float32, device=DEV)
        batch = post.unsqueeze(0).expand(acc_mat.size(0), -1)
        logits = model(batch, acc_mat)          # shape (N,)
        topk   = torch.topk(logits, k=max(K_LIST)).indices.cpu().tolist()
        true_i = uid2idx[uid]
        for K in K_LIST:
            if true_i in topk[:K]:
                hits[K] += 1

# ----- print -----
N = len(samples)
print(f"Checkpoint: {os.path.basename(CKPT_PATH)}   Samples: {N}")
for K in K_LIST:
    print(f"Top-{K:3d}: {hits[K]:4d}/{N} = {hits[K]/N*100:5.2f}%")