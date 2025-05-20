#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_follow_probabilities.py
────────────────────────────────────────────────────────
1. account_freq.csv がなければ follow_dataset.pt から自動生成
2. temp_i = baseT * (freq_i / mean)^alpha を計算
3. score_i = sigmoid(logit_i) / temp_i で人気アカウントを減点
────────────────────────────────────────────────────────
"""

import os
import csv
import re
import argparse
import collections

import numpy as np
import torch
from tqdm import tqdm

from model import SetToVectorPredictor  # 学習時と同じ model.py を使う

# ───────── 既定パス ─────────────────────────────────
ROOT       = "/Users/oobayashikoushin/Enishi_system"
DEF_MODEL  = f"{ROOT}/saved_models/set_transformer_follow_predictor.pt"
DEF_DATA   = f"{ROOT}/data_processed/follow_dataset.pt"
DEF_POSTS  = f"{ROOT}/embedding_openAI/aggregated_posting_vectors.csv"
DEF_NPY    = f"{ROOT}/account_vectors.npy"
DEF_FREQ   = f"{ROOT}/account_freq.csv"

# ───────── モデル構造（学習設定に合わせて調整） ─────────
POST_DIM  = 3072
ENC_DIM   = 384    # 学習コードで 512→384 に縮小
N_HEADS   = 4
N_LAYERS  = 2
DROPOUT   = 0.3    # 学習コードで 0.1→0.3 に増加

# ───────── 推論パラメータ ─────────────────────────
MAX_POST     = 50
TOP_K        = 10
BASE_T       = 1.0
ALPHA        = 1.3   # 学習コードで設定した α
MAX_TEMP     = 50.0  # 学習コードで設定した上限温度

# ───────── ユーティリティ関数 ─────────────────────
def parse_vector(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l + 1:r]
    s = re.sub(r'[\s,]+', ' ', s).strip()
    v = np.fromstring(s, dtype=np.float32, sep=' ')
    return v if v.size == dim else None

def make_freq_csv(freq_csv, dataset_pt, account_list):
    if os.path.isfile(freq_csv):
        return
    if not os.path.isfile(dataset_pt):
        print("[Warn] no freq CSV & no dataset.pt → debias disabled")
        return
    print(f"[Info] generating {freq_csv} from {dataset_pt}...")
    data = torch.load(dataset_pt)
    counter = collections.Counter()
    for _, tgt, _ in tqdm(data["dataset"], desc="counting follows"):
        idxs = (tgt == 1).nonzero(as_tuple=False).flatten().tolist()
        counter.update(idxs)
    with open(freq_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for idx, cnt in counter.items():
            w.writerow([account_list[idx], int(cnt)])
    print("[Info] freq CSV written.")

def load_account_list(npy_path):
    return sorted(np.load(npy_path, allow_pickle=True).item().keys())

def load_temp_array(freq_csv, acc_list, baseT, alpha, max_t):
    if not os.path.isfile(freq_csv):
        print("[Warn] freq CSV absent → no debias")
        return np.full(len(acc_list), baseT, dtype=np.float32)
    freq = {r[0]: max(1.0, float(r[1])) for r in csv.reader(open(freq_csv))}
    arr  = np.array([freq.get(a,1.0) for a in acc_list], dtype=np.float32)
    mean = arr.mean() or 1.0
    temp = baseT * np.power(arr / mean, alpha)
    return np.clip(temp, 0.1, max_t)

def load_posts(csv_path, uid, max_n, dim):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for r in rdr:
            if len(r) >= 3 and r[0] == uid:
                v = parse_vector(r[2], dim)
                if v is not None:
                    rows.append(v)
    if not rows:
        return None
    if 0 < max_n < len(rows):
        rows = rows[-max_n:]
    return torch.tensor(np.array(rows), dtype=torch.float32)

# ───────── 予測ロジック ─────────────────────────
def predict(args):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print("[Device]", device)

    # アカウント一覧と freq CSV の準備
    acc_list = load_account_list(args.accounts_npy)
    make_freq_csv(args.freq_csv, args.dataset_pt, acc_list)
    temp_np  = load_temp_array(
        args.freq_csv, acc_list,
        args.base_temp, args.alpha, args.max_temp
    )
    temp_t   = torch.tensor(temp_np, dtype=torch.float32, device=device)

    # モデル読み込み
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model = SetToVectorPredictor(
        post_embedding_dim=args.post_dim,
        encoder_output_dim=args.enc_dim,
        num_all_accounts=len(acc_list),
        num_attention_heads=args.n_heads,
        num_encoder_layers=args.n_layers,
        dropout_rate=args.dropout
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # 投稿ベクトル読み込み
    posts = load_posts(
        args.posts_csv, args.account_id,
        args.max_posts, args.post_dim
    )
    if posts is None:
        print(f"No posts found for account '{args.account_id}'")
        return
    mask = torch.ones(1, posts.size(0), dtype=torch.bool, device=device)

    # 推論 & debias
    with torch.no_grad():
        logits, _ = model(posts.unsqueeze(0).to(device), mask)
        probs     = torch.sigmoid(logits.squeeze(0)).cpu().numpy()
        scores    = probs / temp_np  # 人気アカウントを温度で割って減点

    # 上位 K 抽出
    idxs = np.argsort(scores)[::-1][: args.top_k]
    print(f"\n--- Top {args.top_k} for {args.account_id} (α={args.alpha}) ---")
    for rank, i in enumerate(idxs, 1):
        print(f"{rank:2d}. {acc_list[i]:<20}  score={scores[i]:.4f}  temp={temp_np[i]:.2f}")

# ───────── CLI ─────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--account_id", required=True)
    p.add_argument("--model_path",  default=DEF_MODEL)
    p.add_argument("--dataset_pt",  default=DEF_DATA)
    p.add_argument("--posts_csv",   default=DEF_POSTS)
    p.add_argument("--accounts_npy", default=DEF_NPY)
    p.add_argument("--freq_csv",    default=DEF_FREQ)
    p.add_argument("--top_k",       type=int,   default=TOP_K)
    p.add_argument("--max_posts",   type=int,   default=MAX_POST)
    p.add_argument("--base_temp",   type=float, default=BASE_T)
    p.add_argument("--alpha",       type=float, default=ALPHA)
    p.add_argument("--max_temp",    type=float, default=MAX_TEMP)
    p.add_argument("--post_dim",    type=int,   default=POST_DIM)
    p.add_argument("--enc_dim",     type=int,   default=ENC_DIM)
    p.add_argument("--n_heads",     type=int,   default=N_HEADS)
    p.add_argument("--n_layers",    type=int,   default=N_LAYERS)
    p.add_argument("--dropout",     type=float, default=DROPOUT)
    return p.parse_args()

if __name__ == "__main__":
    predict(get_args())