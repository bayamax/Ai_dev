#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_rw.py ― ランダムウォーク埋め込み予測のランキング精度評価（ファイルパス固定版）

ハードコードされたパスのモデルとデータを使い、
ランダムにサンプリングした投稿から予測し、
Top-1,5,10,20,50,100,200,500 精度を出力します。
"""

import os
import sys
import csv
import random

import numpy as np
import torch
import torch.nn as nn

# ──────────── ハードコード済みパス ────────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CHECKPOINT_PATH = os.path.join(VAST_DIR, "set_transformer_rw.ckpt")

# ──────────── ハイパラ ────────────
POST_DIM     = 3072
CHUNK_SIZE   = 128
D_MODEL      = 256
ENC_LAYERS   = 16
N_HEADS      = 4
DROPOUT      = 0.3
NUM_SAMPLES  = 1000
TOP_K_LIST   = [1, 5, 10, 20, 50, 100, 200, 500]

# ──────────────────── モデル定義 ────────────────────
class ChunkedTransformerRW(nn.Module):
    def __init__(self, post_dim, chunk_size, d_model,
                 n_heads, enc_layers, dropout, rw_dim):
        super().__init__()
        assert post_dim % chunk_size == 0, "post_dim must be divisible by chunk_size"
        self.chunk_size = chunk_size
        self.num_chunks = post_dim // chunk_size
        self.patch_proj = nn.Linear(chunk_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.decoder     = nn.Linear(d_model, rw_dim)

    def forward(self, posts, pad_mask):
        B, S, D = posts.size()
        # 1) チャンク化 → (B, S*num_chunks, chunk_size)
        x = posts.view(B, S, self.num_chunks, self.chunk_size)
        x = x.flatten(1, 2)
        # 2) 射影 → (B, T, d_model)
        x = self.patch_proj(x)
        # 3) マスク → (B, T)
        pm = pad_mask.unsqueeze(-1).expand(B, S, self.num_chunks)
        token_mask = pm.flatten(1, 2)
        # 4) Transformer → (T, B, d_model) → (B, T, d_model)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, src_key_padding_mask=token_mask)
        x = x.permute(1, 0, 2)
        # 5) マスク平均プーリング
        valid = (~token_mask).unsqueeze(-1).float()
        sum_vec = (x * valid).sum(dim=1)
        lengths = valid.sum(dim=1).clamp(min=1.0)
        pooled  = sum_vec / lengths
        # 6) デコーダー
        return self.decoder(pooled)

# ──────────── ベクトル文字列パース ────────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

# ──────────── ランダムサンプリング ────────────
def sample_posts(k):
    reservoir = []
    with open(POSTS_CSV, encoding='utf-8') as f:
        rdr = csv.reader(f)
        next(rdr)
        for i, row in enumerate(rdr):
            uid, vec_str = row[0], row[2]
            vec = parse_vec(vec_str, POST_DIM)
            if vec is None:
                continue
            if len(reservoir) < k:
                reservoir.append((uid, vec))
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = (uid, vec)
    return reservoir

# ──────────── メイン評価 ────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}", file=sys.stderr)

    # 1) アカウントベクトル読み込み
    rw_dict = np.load(ACCOUNT_NPY, allow_pickle=True).item()
    account_list = sorted(rw_dict.keys())
    acc2idx = {a:i for i,a in enumerate(account_list)}
    mat = np.stack([rw_dict[a] for a in account_list], axis=0)  # (N, rw_dim)

    # 2) モデルロード
    rw_dim = mat.shape[1]
    model = ChunkedTransformerRW(
        post_dim=POST_DIM,
        chunk_size=CHUNK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        enc_layers=ENC_LAYERS,
        dropout=DROPOUT,
        rw_dim=rw_dim
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    # 3) 投稿サンプリング
    samples = sample_posts(NUM_SAMPLES)

    # 4) 推論＆ランキング
    counters = {K: 0 for K in TOP_K_LIST}
    with torch.no_grad():
        for uid, vec in samples:
            if uid not in acc2idx:
                continue
            post = torch.tensor(vec, dtype=torch.float32, device=device)
            post = post.unsqueeze(0).unsqueeze(0)  # (1,1,D)
            mask = torch.zeros((1,1), dtype=torch.bool, device=device)
            pred = model(post, mask).cpu().numpy()[0]
            pred /= np.linalg.norm(pred)
            sims = mat.dot(pred)
            rank = int((-sims).argsort().tolist().index(acc2idx[uid]) + 1)
            for K in TOP_K_LIST:
                if rank <= K:
                    counters[K] += 1

    # 5) 結果出力
    N = len(samples)
    print(f"Samples: {N}")
    for K in TOP_K_LIST:
        cnt = counters[K]
        print(f"Top-{K:3d}: {cnt:4d}/{N} = {cnt/N*100:5.2f}%")

if __name__ == "__main__":
    main()