#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pair_classifier_stream.py  ―  BCE(0/1) 版 + ノイズ負例
投稿ベクトル ⇔ アカウントベクトルのペアを
正例(同一UID)=1／負例(異UID or ノイズ)=0 で学習するストリーミング版。

主な追加点
・正例ごとに “ノイズを加えたアカウントベクトル” を 1 本負例として生成
・エポック開始ごとに乱数シードを切るので毎回ノイズが変わる
・resume/early-stopping/進捗バー無し は従来どおり
"""

import os, csv, random, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# ─────────── ファイルパス ───────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CHECKPOINT_PATH = os.path.join(VAST_DIR, "pair_classifier_rw_stream.ckpt")

# ─────────── ハイパーパラメータ ───────────
POST_DIM      = 3072
BATCH_SIZE    = 128
EPOCHS        = 500
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
NEG_RATIO     = 5      # ランダム負例本数
VAL_RATIO     = 0.1
DROPOUT_RATE  = 0.1
NOISE_STD     = 0.2    # ノイズ負例 σ
PATIENCE      = 15
MIN_DELTA     = 1e-4

# ─────────── ヘルパ関数 ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r: s = s[l+1:r]
    vec = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return vec if vec.size == dim else None

def uid_to_val(uid: str, ratio: float) -> bool:
    h = int(hashlib.md5(uid.encode()).hexdigest(), 16)
    return (h % 10000) / 10000 < ratio

def l2_normalize(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n else v

# ─────────── データセット ───────────
class PairStreamDataset(IterableDataset):
    def __init__(self, posts_csv, account_npy, split="train"):
        assert split in ("train", "val")
        self.posts_csv = posts_csv
        self.acc_dict  = np.load(account_npy, allow_pickle=True).item()
        self.uids      = list(self.acc_dict.keys())
        self.split     = split
        self.rw_dim    = next(iter(self.acc_dict.values())).shape[0]

    def __iter__(self):
        np.random.seed()  # エポック毎に乱数リセット
        with open(self.posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_str in rdr:
                if uid not in self.acc_dict:
                    continue
                if uid_to_val(uid, VAL_RATIO) ^ (self.split == "val"):
                    continue

                post_np = parse_vec(vec_str, POST_DIM)
                if post_np is None:
                    continue
                post_t = torch.from_numpy(post_np)

                # ---------- 正例 ----------
                acc_pos = torch.from_numpy(self.acc_dict[uid].astype(np.float32))
                yield post_t, acc_pos, torch.tensor(1.0)

                # ---------- ノイズ負例 ----------
                noise = np.random.normal(0, NOISE_STD, size=acc_pos.shape).astype(np.float32)
                acc_noise = torch.from_numpy(l2_normalize(self.acc_dict[uid] + noise))
                yield post_t, acc_noise, torch.tensor(0.0)

                # ---------- ランダム負例 ----------
                for _ in range(NEG_RATIO):
                    neg_uid = uid
                    while neg_uid == uid:
                        neg_uid = random.choice(self.uids)
                    acc_neg = torch.from_numpy(self.acc_dict[neg_uid].astype(np.float32))
                    yield post_t, acc_neg, torch.tensor(0.0)

# ─────────── モデル ───────────
class PairClassifier(nn.Module):
    def __init__(self, post_dim, rw_dim, hidden_dim=512, dropout=DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim + rw_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, post_vec, acc_vec):
        x = torch.cat([post_vec, acc_vec], dim=1)
        return self.net(x).squeeze(1)

# ─────────── 学習ループ ───────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    train_ds = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, "train")
    val_ds   = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, "val")
    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    va_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = PairClassifier(POST_DIM, train_ds.rw_dim).to(device)
    optim_ = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit   = nn.BCEWithLogitsLoss()

    start, best, wait = 1, float("inf"), 0
    if args.resume and os.path.isfile(CHECKPOINT_PATH):
        ck = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ck["model_state"])
        optim_.load_state_dict(ck["optim_state"])
        start, best, wait = ck["epoch"] + 1, ck["best_val"], ck["patience"]
        print(f"[Resume] epoch={ck['epoch']} best_val={best:.4f}")

    for ep in range(start, EPOCHS + 1):
        # ----- train -----
        model.train(); tr_loss = n_tr = 0
        for post, acc, lbl in tr_loader:
            post, acc, lbl = post.to(device), acc.to(device), lbl.to(device)
            loss = crit(model(post, acc), lbl)
            optim_.zero_grad(); loss.backward(); optim_.step()
            bs = post.size(0); tr_loss += loss.item()*bs; n_tr += bs
        tr_loss /= n_tr

        # ----- val -----
        model.eval(); va_loss = n_va = 0
        with torch.no_grad():
            for post, acc, lbl in va_loader:
                post, acc, lbl = post.to(device), acc.to(device), lbl.to(device)
                loss = crit(model(post, acc), lbl)
                va_loss += loss.item()*post.size(0); n_va += post.size(0)
        va_loss /= n_va
        print(f"Ep{ep:03d} train={tr_loss:.4f}  val={va_loss:.4f}")

        # ----- ckpt / early-stop -----
        if va_loss < best - MIN_DELTA:
            best, wait = va_loss, 0
            torch.save({"epoch":ep,"model_state":model.state_dict(),
                        "optim_state":optim_.state_dict(),
                        "best_val":best,"patience":wait}, CHECKPOINT_PATH)
            print("  ✔ checkpoint saved")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping."); break
    print(f"[Done] best_val={best:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    train(ap.parse_args())