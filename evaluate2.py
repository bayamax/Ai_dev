#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_prob_weighted.py
─────────────────────────────────────────────────────────────
手順
1. PairClassifier で 全アカウント logits を取得し soft-max → 確率 p_i
2. 上位 TOP_N の RW ベクトル r_i を p_i で加重平均
       v★ = Σ p_i r_i / Σ p_i
3. v★ と 全 RW のコサイン類似度で再ソート
4. Recall@K を出力（K = 1,5,10,20,50,100,200,500）

※ PairClassifier / parse_vec の import 行は
   ご自身のファイル名に合わせて書き換えてください
"""
import os, csv, random, argparse, numpy as np, torch
from collections import defaultdict

# ── import 先を調整 ────────────────────────────────────────────
from train_pair_classifier_stream import PairClassifier, parse_vec  # ここ！

# ── デフォルト設定 ────────────────────────────────────────────
DEF_VAST      = "/workspace/edit_agent/vast"
DEF_TOP_N     = 200
DEF_POSTS     = 50
DEF_UID_SMP   = 1000
K_LIST        = [1,5,10,20,50,100,200,500]

# ── 引数 ────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--vast_dir",      default=DEF_VAST)
ap.add_argument("--posts_per_uid", type=int, default=DEF_POSTS)
ap.add_argument("--top_n",         type=int, default=DEF_TOP_N,
                help="確率加重平均に使う上位アカウント数")
ap.add_argument("--uid_samples",   type=int, default=DEF_UID_SMP,
                help="評価に使う UID 数 (None=全 UID)")
args = ap.parse_args()

POSTS_CSV = os.path.join(args.vast_dir, "aggregated_posting_vectors.csv")
ACCS_NPY  = os.path.join(args.vast_dir, "account_vectors.npy")
CKPT      = os.path.join(args.vast_dir, "pair_classifier_rw_stream.ckpt")

# ── 準備 ──────────────────────────────────────────────────────
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=DEV)        # (N,D)
uid2idx  = {u:i for i,u in enumerate(uids)}

model = PairClassifier(post_dim=3072, rw_dim=acc_mat.size(1)).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV)["model_state"])
model.eval()

# ── UID → 最新投稿収集 ────────────────────────────────────
uid_posts = defaultdict(list)
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_s in rdr:
        if uid not in acc_dict: continue
        v = parse_vec(vec_s, 3072)
        if v is None: continue
        lst = uid_posts[uid]
        lst.append(v)
        if len(lst) > args.posts_per_uid:
            lst.pop(0)                             # 最新 posts_per_uid 本のみ

eval_uids = (random.sample(list(uid_posts.keys()), args.uid_samples)
             if args.uid_samples else list(uid_posts.keys()))

# ── 評価ループ ──────────────────────────────────────────────
hits = {k:0 for k in K_LIST}

with torch.no_grad():
    for uid in eval_uids:
        posts = np.asarray(uid_posts[uid], np.float32)   # (m,D)
        logits = torch.zeros(len(uids), device=DEV)

        # 全アカウント logits 合算
        for p in posts:
            p_t = torch.tensor(p, dtype=torch.float32, device=DEV)
            logits += model(p_t.unsqueeze(0).expand(acc_mat.size(0), -1), acc_mat)

        prob = torch.softmax(logits, dim=0)              # (N,)
        top_idx = torch.topk(prob, k=args.top_n).indices  # (top_n,)

        w = prob[top_idx]                                # 重み
        rw = acc_mat[top_idx]                            # RW
        v_star = (w.unsqueeze(1) * rw).sum(0) / w.sum()
        v_star = v_star / (torch.norm(v_star)+1e-8)      # L2 正規化

        sims = torch.matmul(v_star, acc_mat.T)           # (N,)
        rank = torch.argsort(sims, descending=True).cpu().numpy()

        true = uid2idx[uid]
        for K in K_LIST:
            if true in rank[:K]:
                hits[K] += 1

# ── 出力 ────────────────────────────────────────────────────
N = len(eval_uids)
print(f"UID={N}  posts≤{args.posts_per_uid}  TOP_N={args.top_n}")
for K in K_LIST:
    print(f"Top-{K:3d}: {hits[K]:4d}/{N} = {hits[K]/N*100:5.2f}%")