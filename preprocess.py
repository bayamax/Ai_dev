#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py  ―  フォロー予測用データセット作成
    • 正例 = 1
    • Hard / Random 負例 = 0
    • それ以外の列 = -1   （→ 学習時に損失を計算しない）
    → 全データ（CSV/NPY/PT）は /workspace/edit_agent/vast 下にまとめる
"""

import os
import csv
import re
import random
import argparse
import sys
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ───────── パス既定 ─────────
VAST_DIR    = "/workspace/edit_agent/vast"
DEF_POSTS   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
DEF_ACCNPY  = os.path.join(VAST_DIR, "account_vectors.npy")
DEF_EDGES   = os.path.join(VAST_DIR, "edges.csv")
DEF_OUTPT   = os.path.join(VAST_DIR, "follow_dataset.pt")

# ───────── 定数 ─────────
POST_DIM    = 3072
MAX_POST    = 50
NEG_RATIO   = 5   # ランダム負例 : 正例
HARD_NEG    = 2   # Hard-Neg   : 正例
MIN_POSTS   = 2   # 最低投稿数

# ───────── utilities ─────────
def parse_vec(s, dim):
    s = s.strip()
    if s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    v = np.fromstring(re.sub(r'[\s,]+', ' ', s), dtype=np.float32, sep=' ')
    return v if v.size == dim else None

def load_accounts(path):
    data = np.load(path, allow_pickle=True).item()
    acc_list = sorted(data.keys())
    acc2idx  = {a:i for i,a in enumerate(acc_list)}
    return acc_list, acc2idx

# ───────── main preprocess ─────────
def preprocess(posts_csv, acc_npy, edges_csv, out_pt,
               max_posts, neg_ratio, hard_neg, min_posts):
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)

    # アカウント一覧
    acc_list, acc2idx = load_accounts(acc_npy)
    n_acc = len(acc_list)

    # (1) 投稿ベクトル読み込み
    posts_map = defaultdict(list)
    with open(posts_csv, encoding='utf-8') as f:
        rdr = csv.reader(f); next(rdr, None)
        for row in tqdm(rdr, desc="Loading posts"):
            if len(row) < 3:
                continue
            uid = row[0]
            if uid not in acc2idx:
                continue
            vec = parse_vec(row[2], POST_DIM)
            if vec is not None:
                posts_map[uid].append(vec)
    # トランケート & テンソル化 & 投稿数フィルタ
    processed_posts = {
        uid: torch.tensor(vecs[-max_posts:], dtype=torch.float32)
        for uid, vecs in posts_map.items()
        if len(vecs) >= min_posts
    }
    print(f"Users with ≥{min_posts} posts: {len(processed_posts)}")
    if not processed_posts:
        sys.exit("No users remain after post filtering.")

    # (2) フォローグラフ読み込み
    follows   = defaultdict(set)
    pop_count = Counter()
    df_edges  = pd.read_csv(edges_csv)
    for _, row in tqdm(df_edges.iterrows(), total=len(df_edges), desc="Loading edges"):
        src = str(row['source']); dst = str(row['target'])
        if src in acc2idx and dst in acc2idx:
            follows[src].add(dst)
            pop_count[dst] += 1
    print(f"Users with follow info: {len(follows)}")

    # (3) Hard-neg pool (人気上位1000)
    popular = [a for a,_ in pop_count.most_common(1000)]

    # (4) データセット構築
    dataset = []
    for uid, post_tensor in tqdm(processed_posts.items(), desc="Building dataset"):
        targ = torch.full((n_acc,), -1.0, dtype=torch.float32)
        pos_set = follows.get(uid, set())
        # 正例
        for p in pos_set:
            targ[acc2idx[p]] = 1.0
        # Hard negative
        hard_cand = [h for h in popular if h not in pos_set and h != uid]
        hard_sample = hard_cand[: len(pos_set) * hard_neg]
        # Random negative
        rand_cand = [a for a in acc_list if a not in pos_set and a not in hard_sample and a != uid]
        rand_sample = random.sample(rand_cand, min(len(rand_cand), len(pos_set) * neg_ratio))
        for neg in hard_sample + rand_sample:
            targ[acc2idx[neg]] = 0.0
        dataset.append((post_tensor, targ, uid))
    print(f"Total samples: {len(dataset)}")

    # (5) 保存
    torch.save({
        'dataset': dataset,
        'all_account_list': acc_list,
        'account_to_idx': acc2idx
    }, out_pt)
    print("Saved follow_dataset.pt →", out_pt)

# ───────── CLI ─────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--posts_csv",        default=DEF_POSTS)
    p.add_argument("--accounts_npy",     default=DEF_ACCNPY)
    p.add_argument("--edges_csv",        default=DEF_EDGES)
    p.add_argument("--dataset_output_path", default=DEF_OUTPT)
    p.add_argument("--max_posts",        type=int, default=MAX_POST)
    p.add_argument("--negative_ratio",   type=int, default=NEG_RATIO)
    p.add_argument("--hard_neg",         type=int, default=HARD_NEG)
    p.add_argument("--min_posts",        type=int, default=MIN_POSTS)
    return p.parse_args()

if __name__ == "__main__":
    args = cli()
    preprocess(
        args.posts_csv,
        args.accounts_npy,
        args.edges_csv,
        args.dataset_output_path,
        args.max_posts,
        args.negative_ratio,
        args.hard_neg,
        args.min_posts
    )