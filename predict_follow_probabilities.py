#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py ― フォロー予測モデルの推論スクリプト

  1. account_freq.csv がなければ follow_dataset.pt から自動生成
  2. temp_i = base_t * (freq_i / mean_freq)^alpha を計算
  3. score_i = sigmoid(logit_i) / temp_i で人気アカウントを減点
  4. top_k 件を表示
"""

import os
import csv
import re
import argparse
import collections

import numpy as np
import torch
from tqdm import tqdm

from model import SetToVectorPredictor  # train 時と同じ model.py を同ディレクトリに置く

# ─────────── 定数 ―――――――――――――――――――――――
VAST_DIR        = "/workspace/edit_agent/vast"
MODEL_PATH      = os.path.join(VAST_DIR, "set_transformer_follow_predictor.pt")
DATASET_PT_PATH = os.path.join(VAST_DIR, "follow_dataset.pt")
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNTS_NPY    = os.path.join(VAST_DIR, "account_vectors.npy")
FREQ_CSV        = os.path.join(VAST_DIR, "account_freq.csv")

# モデル構造（学習時と同じものに）
POST_DIM   = 3072
ENC_DIM    = 512
N_HEADS    = 4
N_LAYERS   = 2
DROPOUT    = 0.1

# 推論パラメータのデフォルト
MAX_POSTS  = 50
TOP_K      = 10
BASE_T     = 1.0
ALPHA      = 1.0
MAX_TEMP   = 30.0

# ―― 文字列→ベクトル変換 ――
def parse_vector(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    s = re.sub(r'[\s,]+', ' ', s).strip()
    v = np.fromstring(s, dtype=np.float32, sep=' ')
    return v if v.size == dim else None

# ―― アカウント一覧ロード ――
def load_account_list(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    return sorted(data.keys())

# ―― freq CSV 自動生成 ――
def make_freq_csv(freq_csv, dataset_pt, account_list):
    if os.path.isfile(freq_csv):
        return
    if not os.path.isfile(dataset_pt):
        print("[Warn] freq CSV も dataset.pt も存在しない → デバイアス無効化")
        return
    print(f"[Info] {freq_csv} を {dataset_pt} から生成中…")
    data = torch.load(dataset_pt)
    counter = collections.Counter()
    for _, tgt, _ in tqdm(data["dataset"], desc="Counting positives"):
        idxs = (tgt == 1.0).nonzero(as_tuple=False).flatten().tolist()
        counter.update(idxs)
    with open(freq_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx, cnt in counter.items():
            writer.writerow([account_list[idx], cnt])
    print("[Info] freq CSV を書き出しました。")

# ―― 温度配列ロード ――
def load_temp_array(freq_csv, account_list, base_t, alpha, max_temp):
    if not os.path.isfile(freq_csv):
        print("[Warn] freq CSV がないためデバイアス無効化")
        return np.full(len(account_list), base_t, dtype=np.float32)
    freq = {row[0]: max(1.0, float(row[1])) for row in csv.reader(open(freq_csv))}
    arr  = np.array([freq.get(a, 1.0) for a in account_list], np.float32)
    mean = arr.mean() or 1.0
    temp = base_t * np.power(arr / mean, alpha)
    return np.clip(temp, 0.1, max_temp)

# ―― 投稿ベクトルロード ――
def load_posts(csv_path, account_id, max_posts, dim):
    vectors = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3 or row[0] != account_id:
                continue
            v = parse_vector(row[2], dim)
            if v is not None:
                vectors.append(v)
    if not vectors:
        return None
    if 0 < max_posts < len(vectors):
        vectors = vectors[-max_posts:]
    return torch.tensor(np.stack(vectors, axis=0), dtype=torch.float32)

# ―― 推論本体 ――
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) アカウント一覧と freq CSV
    account_list = load_account_list(args.accounts_npy)
    make_freq_csv(args.freq_csv, args.dataset_pt, account_list)

    # 2) 温度テンソル
    temp_np = load_temp_array(args.freq_csv, account_list,
                              args.base_temp, args.alpha, args.max_temp)
    temp_t  = torch.tensor(temp_np, device=device)

    # 3) モデルロード
    state = torch.load(args.model_path, map_location=device)
    model = SetToVectorPredictor(
        post_embedding_dim=args.post_dim,
        encoder_output_dim=args.enc_dim,
        num_all_accounts=len(account_list),
        num_attention_heads=args.n_heads,
        num_encoder_layers=args.n_layers,
        dropout_rate=args.dropout
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # 4) 投稿ベクトル取得
    posts = load_posts(args.posts_csv, args.account_id,
                       args.max_posts, args.post_dim)
    if posts is None:
        print(f"[Error] アカウント {args.account_id} の投稿が見つかりません。")
        return
    mask = torch.zeros(1, posts.size(0), dtype=torch.bool, device=device)

    # 5) 推論＋デバイアス
    with torch.no_grad():
        logits, _ = model(posts.unsqueeze(0).to(device), mask)
        probs = torch.sigmoid(logits).squeeze(0).cpu()  # (N,)
        scores = probs / temp_t.cpu()

    # 6) 上位 k 件表示
    idxs = scores.argsort(descending=True)[: args.top_k]
    print(f"\n--- Top {args.top_k} for {args.account_id} (α={args.alpha}, max_temp={args.max_temp}) ---")
    for rank, i in enumerate(idxs.tolist(), start=1):
        print(f"{rank:2d}. {account_list[i]:<20}  score={scores[i]:.4f}  temp={temp_np[i]:.2f}")

# ―― CLI ――
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--account_id",    required=True)
    p.add_argument("--model_path",    default=MODEL_PATH)
    p.add_argument("--dataset_pt",    default=DATASET_PT_PATH)
    p.add_argument("--posts_csv",     default=POSTS_CSV)
    p.add_argument("--accounts_npy",  default=ACCOUNTS_NPY)
    p.add_argument("--freq_csv",      default=FREQ_CSV)
    p.add_argument("--top_k",    type=int,   default=TOP_K)
    p.add_argument("--max_posts", type=int,   default=MAX_POSTS)
    p.add_argument("--post_dim",  type=int,   default=POST_DIM)
    p.add_argument("--enc_dim",   type=int,   default=ENC_DIM)
    p.add_argument("--n_heads",   type=int,   default=N_HEADS)
    p.add_argument("--n_layers",  type=int,   default=N_LAYERS)
    p.add_argument("--dropout",   type=float, default=DROPOUT)
    p.add_argument("--base_temp", type=float, default=BASE_T)
    p.add_argument("--alpha",     type=float, default=ALPHA)
    p.add_argument("--max_temp",  type=float, default=MAX_TEMP)
    args = p.parse_args()
    predict(args)

if __name__ == "__main__":
    main()