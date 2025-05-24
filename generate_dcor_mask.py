#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dcor_mask_stream.py
UID を UID_BLOCK 件ずつ処理し、
  投稿集合(≤30本) × アカウント集合(UID_BLOCK本) の距離相関 dCor を計算。
投稿ごとの寄与 Δ を行内積近似で求め、上位 TOP_P% を 1 とした
ビットマスク <MASK_DIR>/<uid>.npy を保存。
CSV はブロック毎に 1 パスだけ読み、常時メモリ≲40 MB。
"""

import os, csv, math, argparse, gc
import numpy as np
from collections import deque, defaultdict
from scipy.spatial.distance import pdist, squareform
# ---------- paths ----------
VAST   = "/workspace/edit_agent/vast"
POSTS  = os.path.join(VAST, "aggregated_posting_vectors.csv")
ACCS   = os.path.join(VAST, "account_vectors.npy")
MASK_DIR = os.path.join(VAST, "dcor_masks")
os.makedirs(MASK_DIR, exist_ok=True)
# ---------- hyper ----------
POST_DIM       = 3072
POSTS_PER_UID  = 30
UID_BLOCK      = 100   # ←増やして限界テスト可
TOP_P          = 0.05  # 上位5%採用
# ---------- utils ----------
def parse_vec(s, d):
    s=s.strip();    null = ("[]",'\"[]\"')
    if not s or s in null: return None
    if s.startswith('"[') and s.endswith(']"'): s=s[1:-1]
    l,r=s.find('['),s.rfind(']')
    if 0<=l<r: s=s[l+1:r]
    v=np.fromstring(s.replace(',',' '),sep=' ',dtype=np.float32)
    return v if v.size==d else None
def center(M):
    row=M.mean(1,keepdims=True)
    col=M.mean(0,keepdims=True)
    g = M.mean()
    return M-row-col+g
# ---------- load account vectors ----------
acc_dict = np.load(ACCS, allow_pickle=True).item()
uids_all = list(acc_dict.keys())
print(f"UID total {len(uids_all)}  → masks dir {MASK_DIR}")
# ---------- main loop ----------
for blk_idx, beg in enumerate(range(0, len(uids_all), UID_BLOCK)):
    uid_batch = uids_all[beg:beg+UID_BLOCK]
    uid_set   = set(uid_batch)
    # 1) collect latest posts streaming
    posts_buf   = {u: deque(maxlen=POSTS_PER_UID) for u in uid_batch}
    with open(POSTS, encoding='utf-8') as f:
        rdr=csv.reader(f); next(rdr)
        for uid, _, vec_str in rdr:
            if uid not in uid_set: continue
            v=parse_vec(vec_str, POST_DIM)
            if v is not None:
                posts_buf[uid].append(v)         # deque keeps latest
    # remove UID w/o posts
    uid_batch=[u for u in uid_batch if posts_buf[u]]
    if not uid_batch: continue
    # 2) build matrices
    P, uid_of_row = [], []
    for u in uid_batch:
        P.extend(posts_buf[u])
        uid_of_row.extend([u]*len(posts_buf[u]))
    P=np.asarray(P,np.float32)                  # (R,Dp)
    A=np.stack([acc_dict[u] for u in uid_batch]).astype(np.float32)  # (B,Drw)
    R=len(P); B=len(A)
    print(f"[Block {blk_idx}] UID={B} Posts={R}")
    # 3) distance matrices & centering
    dP = squareform(pdist(P, 'euclidean'))
    dA = squareform(pdist(A, 'euclidean'))
    Pc = center(dP).astype(np.float32)
    Ac = center(dA).astype(np.float32)
    Ac_big = np.repeat(np.repeat(Ac, POSTS_PER_UID,0), POSTS_PER_UID,1)[:R,:R]
    Z = math.sqrt((Pc*Pc).mean() * (Ac_big*Ac_big).mean()) + 1e-8
    delta = (Pc*Ac_big).mean(1) / Z            # 投稿寄与の近似値
    # 4) keep top p%
    thr  = np.percentile(delta, 100*(1-TOP_P))
    keep = delta >= thr                        # bool (R,)
    # 5) save mask per UID
    ptr=0
    for u in uid_batch:
        n=len(posts_buf[u])
        np.save(os.path.join(MASK_DIR,f"{u}.npy"),
                keep[ptr:ptr+n].astype(np.uint8))
        ptr+=n
    # clean
    del posts_buf, P, A, dP, dA, Pc, Ac, Ac_big, delta, keep
    gc.collect()
print("✓ dCor masks generated (stream-safe).")