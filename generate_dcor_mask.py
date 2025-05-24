#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dcor_mask.py
UID を 100 件ずつバッチ処理し、投稿ベクトル X_i と
本人アカウントベクトル Y_i (= 同 UID RW ベクトル) の distance correlation を計算。
各投稿の寄与値 Δ_i を近似し、上位 TOP_P% を採用 (=1) として
<mask_dir>/<uid>.npy に uint8 ビットマスクを保存。
"""

import os, csv, math, argparse
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform

# ─────────── パス & パラメータ ───────────
VAST_DIR          = "/workspace/edit_agent/vast"
POSTS_CSV         = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY       = os.path.join(VAST_DIR, "account_vectors.npy")
MASK_DIR          = os.path.join(VAST_DIR, "dcor_masks")

POST_DIM          = 3072        # 次元数
POSTS_PER_UID     = 30          # 各 UID 何投稿扱うか
UID_BLOCK         = 100         # 一度に処理する UID 件数
TOP_P             = 0.05        # 上位 5% を採用

os.makedirs(MASK_DIR, exist_ok=True)

# ─────────── ユーティリティ ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r: s = s[l+1:r]
    v = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return v if v.size == dim else None

def center_dist(M):
    """距離行列を二重中心化 (O(n²) mem)"""
    row = M.mean(1, keepdims=True)
    col = M.mean(0, keepdims=True)
    g   = M.mean()
    return M - row - col + g

# ─────────── 1) 各 UID → 最新投稿ベクトル 30 本収集 ───────────
uid_posts = defaultdict(list)       # uid → [vec … 最新が最後]
acc_dict  = np.load(ACCOUNT_NPY, allow_pickle=True).item()

with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_str in rdr:
        if uid not in acc_dict:               # skip UID w/o RW
            continue
        vec = parse_vec(vec_str, POST_DIM)
        if vec is None:                       # parse failure
            continue
        lst = uid_posts[uid]
        lst.append(vec)
        if len(lst) > POSTS_PER_UID:          # keep only latest
            lst.pop(0)

uids_all = [u for u in uid_posts if len(uid_posts[u])]
print(f"UID loaded: {len(uids_all)} ; masks will be saved to {MASK_DIR}")

# ─────────── 2) ブロックごとに dCor & Δ_i 近似計算 ───────────
for blk_start in range(0, len(uids_all), UID_BLOCK):
    batch_uids = uids_all[blk_start:blk_start+UID_BLOCK]
    # --- 行列作成 ---
    posts_mat, uid_of_row = [], []
    for uid in batch_uids:
        vecs = uid_posts[uid]
        posts_mat.extend(vecs)
        uid_of_row.extend([uid]*len(vecs))
    posts_mat = np.asarray(posts_mat, dtype=np.float32)          # (R, Dp)
    acc_mat   = np.stack([acc_dict[uid] for uid in batch_uids]).astype(np.float32)  # (B,Drw)

    R = posts_mat.shape[0]
    print(f"[Block {blk_start//UID_BLOCK}] UID {len(batch_uids)}  Posts {R}")

    # --- 距離行列 ---
    dX = squareform(pdist(posts_mat, 'euclidean'))               # (R,R)  ≲ 36 MB
    dY = squareform(pdist(acc_mat,  'euclidean'))                # (B,B)
    # acc 行列を投稿数に合わせてタイル
    dY_big = np.repeat(np.repeat(dY, POSTS_PER_UID, 0), POSTS_PER_UID, 1)[:R,:R]

    # --- 中心化 ---
    A = center_dist(dX).astype(np.float32)
    B = center_dist(dY_big).astype(np.float32)

    # --- 規格化定数 ---
    Z = math.sqrt( (A*A).mean() * (B*B).mean() ) + 1e-8

    # --- 行ベクトル内積で Δ_i 近似 ---
    delta = (A*B).mean(1) / Z          # shape (R,)

    # --- 投稿ごとの keep 判定 ---
    thr = np.percentile(delta, 100*(1-TOP_P))
    keep_flag = delta >= thr           # bool (R,)

    # --- UID 別マスク保存 ---
    ptr = 0
    for uid in batch_uids:
        n = len(uid_posts[uid])
        mask = keep_flag[ptr:ptr+n].astype(np.uint8)
        np.save(os.path.join(MASK_DIR, f"{uid}.npy"), mask)
        ptr += n

print("✓ distance-correlation masks generated.")
