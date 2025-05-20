#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py  ―  フォロー予測用データセット作成
    • 正例 = 1
    • Hard / Random 負例 = 0
    • それ以外の列 = -1   （→ 学習時に損失を計算しない）
"""

import os, csv, re, random, argparse, sys
from collections import defaultdict, Counter

import numpy as np, pandas as pd, torch
from tqdm import tqdm

# ───────── パス既定 ─────────
ROOT = "/Users/oobayashikoushin/Enishi_system"
DEF_POSTS   = f"{ROOT}/embedding_openAI/aggregated_posting_vectors.csv"
DEF_ACCNPY  = f"{ROOT}/account_vectors.npy"
DEF_EDGES   = f"{ROOT}/edges.csv"
DEF_OUTDIR  = f"{ROOT}/data_processed"
DEF_OUTPT   = f"{DEF_OUTDIR}/follow_dataset.pt"

# ───────── 定数 ─────────
POST_DIM = 3072
MAX_POST = 50
NEG_RATIO = 5          # ランダム負例 : 正例
HARD_NEG  = 2          # Hard-Neg   : 正例
MIN_POSTS = 2

# ───────── utils ─────────
def parse_vec(s, dim):
    s=s.strip()
    if s in ("[]","\"[]\""): return None
    if s.startswith('"[') and s.endswith(']"'): s=s[1:-1]
    l,r=s.find('['),s.rfind(']')
    if 0<=l<r: s=s[l+1:r]
    v=np.fromstring(re.sub(r'[\s,]+',' ',s),dtype=np.float32,sep=' ')
    return v if v.size==dim else None

def load_accounts(path):
    data=np.load(path,allow_pickle=True).item()
    acc=sorted(data.keys())
    return acc,{a:i for i,a in enumerate(acc)}

# ───────── main preprocess ─────────
def preprocess(posts_csv, acc_npy, edges_csv, out_pt,
               max_posts, neg_ratio, hard_neg, min_posts):
    os.makedirs(os.path.dirname(out_pt),exist_ok=True)

    acc_list, acc2idx = load_accounts(acc_npy)
    n_acc = len(acc_list)

    # 1. 投稿読み込み
    posts=defaultdict(list)
    with open(posts_csv,encoding='utf-8') as f:
        rdr=csv.reader(f); next(rdr,None)
        for row in tqdm(rdr,desc="posts"):
            if len(row)<3 or row[0] not in acc2idx: continue
            v=parse_vec(row[2],POST_DIM)
            if v is not None: posts[row[0]].append(v)
    # truncate & min posts
    posts={u:torch.tensor(v[-max_posts:],dtype=torch.float32)
           for u,v in posts.items() if len(v)>=min_posts}
    print("users with posts:", len(posts))

    if not posts: sys.exit("no users after post filter")

    # 2. フォローグラフ
    follows=defaultdict(set)
    pop_cnt=Counter()     # ← popularity for Hard-Neg
    df=pd.read_csv(edges_csv)
    for s,t in tqdm(df.itertuples(index=False), total=len(df), desc="edges"):
        s=str(s); t=str(t)
        if s in acc2idx and t in acc2idx:
            follows[s].add(t); pop_cnt[t]+=1
    print("users with follow info:", len(follows))

    # 3. Hard-Neg プール（人気上位 1000）
    popular=[a for a,_ in pop_cnt.most_common(1000)]

    # 4. データセット作成
    ds=[]
    for u,post_tensor in tqdm(posts.items(),desc="make ds"):
        targ = torch.full((n_acc,), -1.0)          # ★ -1 = ignore
        pos = follows.get(u,set())

        # 正例
        for p in pos:
            targ[acc2idx[p]]=1.0

        # Hard-Neg：人気だがフォローしていない
        hard = [h for h in popular if h not in pos and h!=u][:len(pos)*hard_neg]

        # ランダム負例
        cand=[a for a in acc_list if a not in pos and a not in hard and a!=u]
        rand=random.sample(cand, min(len(cand), len(pos)*neg_ratio))

        for v in hard+rand:
            targ[acc2idx[v]]=0.0

        ds.append((post_tensor, targ, u))
    print("dataset size:", len(ds))

    torch.save({"dataset":ds,
                "all_account_list":acc_list,
                "account_to_idx":acc2idx},
               out_pt)
    print("saved to", out_pt)

# ───────── CLI ─────────
def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--posts_csv",  default=DEF_POSTS)
    p.add_argument("--accounts_npy", default=DEF_ACCNPY)
    p.add_argument("--edges_csv",  default=DEF_EDGES)
    p.add_argument("--dataset_output_path", default=DEF_OUTPT)
    p.add_argument("--max_posts", type=int, default=MAX_POST)
    p.add_argument("--negative_ratio", type=int, default=NEG_RATIO)
    p.add_argument("--hard_neg", type=int, default=HARD_NEG)
    p.add_argument("--min_posts", type=int, default=MIN_POSTS)
    return p.parse_args()

if __name__=="__main__":
    a=cli()
    preprocess(a.posts_csv,a.accounts_npy,a.edges_csv,
               a.dataset_output_path,a.max_posts,
               a.negative_ratio,a.hard_neg,a.min_posts)