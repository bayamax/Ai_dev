#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate2.py  ―  PairClassifier (BCE + ノイズ負例) を単体で評価
学習コードに依存せず、このファイルだけで動くよう
PairClassifier と parse_vec を再掲しています。

出力: Recall@K  (K = 1,5,10,20,50,100,200,500)
"""

import os, csv, numpy as np, torch, argparse

# ───────────────────  再掲：投稿ベクトル文字列を numpy 配列へ  ───────────────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l + 1 : r]
    v = np.fromstring(s.replace(",", " "), dtype=np.float32, sep=" ")
    return v if v.size == dim else None

# ───────────────────  再掲：PairClassifier 本体  ───────────────────
import torch.nn as nn

class PairClassifier(nn.Module):
    """
    投稿ベクトル (D_post) とアカウントベクトル (D_rw) を結合し、
    2 層 MLP で「同一UIDか？」(logit) を出力
    """
    def __init__(self, post_dim: int, rw_dim: int,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim + rw_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, post_vec: torch.Tensor, acc_vec: torch.Tensor):
        x = torch.cat([post_vec, acc_vec], dim=1)
        return self.net(x).squeeze(1)          # shape (B,)

# ───────────────────  パス & 評価パラメータ  ───────────────────
VAST_DIR    = "/workspace/edit_agent/vast"
POSTS_CSV   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_PATH   = os.path.join(VAST_DIR, "pair_classifier_rw_stream.ckpt")

POST_DIM    = 3072
NUM_SAMPLES = 1000
K_LIST      = [1, 5, 10, 20, 50, 100, 200, 500]

# ───────────────────  デバイス  ───────────────────
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEV)

# ───────────────────  アカウント行列 & モデル読み込み  ───────────────────
acc_dict = np.load(ACCOUNT_NPY, allow_pickle=True).item()
uids     = list(acc_dict.keys())
uid2idx  = {u: i for i, u in enumerate(uids)}
acc_mat  = torch.tensor(np.stack([acc_dict[u] for u in uids]),
                        dtype=torch.float32, device=DEV)          # (N,D_rw)

model = PairClassifier(post_dim=POST_DIM, rw_dim=acc_mat.size(1)).to(DEV)
ckpt  = torch.load(CKPT_PATH, map_location=DEV)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint: {os.path.basename(CKPT_PATH)}  (epoch {ckpt['epoch']})")

# ───────────────────  投稿サンプリング  ───────────────────
samples = []
with open(POSTS_CSV, encoding="utf-8") as f:
    rdr = csv.reader(f); next(rdr)
    for uid, _, vec_str in rdr:
        if uid not in acc_dict: continue
        vec = parse_vec(vec_str, POST_DIM)
        if vec is None: continue
        samples.append((uid, vec))
        if len(samples) >= NUM_SAMPLES: break
print(f"Sampled {len(samples)} posts for evaluation")

# ───────────────────  Recall@K 評価  ───────────────────
hits = {K: 0 for K in K_LIST}
with torch.no_grad():
    for uid, vec in samples:
        post  = torch.tensor(vec, dtype=torch.float32, device=DEV)
        batch = post.unsqueeze(0).expand(acc_mat.size(0), -1)     # (N,D_post)
        logits = model(batch, acc_mat)                            # (N,)
        topk   = torch.topk(logits, k=max(K_LIST)).indices.cpu().tolist()
        true_i = uid2idx[uid]
        for K in K_LIST:
            if true_i in topk[:K]:
                hits[K] += 1

# ───────────────────  結果表示  ───────────────────
N = len(samples)
print("\nRecall@K")
for K in K_LIST:
    print(f"Top-{K:3d}: {hits[K]:4d}/{N} = {hits[K]/N*100:5.2f}%")