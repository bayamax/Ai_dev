#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train3.py  ―  dCor マスク採用版 PairClassifier 学習
"""

import os
import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from collections import defaultdict

# ─────────── paths ───────────
VAST_DIR = "/workspace/edit_agent/vast"
POSTS_CSV = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY = os.path.join(VAST_DIR, "account_vectors.npy")
MASK_DIR = os.path.join(VAST_DIR, "dcor_masks")
CKPT_PATH = os.path.join(VAST_DIR, "pair_classifier_masked.ckpt")

# ─────────── hyperparams ───────────
POST_DIM = 3072
BATCH_SIZE = 128
EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-5
NEG_RATIO = 5
VAL_RATIO = 0.1
DROPOUT_RATE = 0.1
NOISE_STD = 0.2
PATIENCE = 15
MIN_DELTA = 1e-4

# ─────────── utils ───────────
def parse_vec(s: str, dim: int):
    """"[0.1, 0.2, ...]" 形式の文字列 -> np.ndarray"""
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l + 1 : r]
    v = np.fromstring(s.replace(',', ' '), sep=' ', dtype=np.float32)
    return v if v.size == dim else None


def uid_to_val(uid: str, ratio: float = VAL_RATIO) -> bool:
    """UID を md5 ハッシュして train / val split"""
    import hashlib

    h = int(hashlib.md5(uid.encode()).hexdigest(), 16)
    return (h % 10000) / 10000 < ratio


def l2_norm(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n else v


# ─────────── dataset ───────────
class MaskedStreamDataset(IterableDataset):
    def __init__(self, split: str = "train"):
        assert split in ("train", "val")
        self.split = split
        self.acc = np.load(ACCS_NPY, allow_pickle=True).item()
        self.uids = list(self.acc.keys())
        self.rw_dim = next(iter(self.acc.values())).shape[0]

        self.mask = {}
        loaded = 0
        for u in self.uids:
            p = os.path.join(MASK_DIR, f"{u}.npy")
            if os.path.exists(p):
                self.mask[u] = np.load(p)
                loaded += 1
            else:
                self.mask[u] = None
        print(f"[Dataset] loaded masks for {loaded} UID")

    def __iter__(self):
        idx_counter = defaultdict(int)
        with open(POSTS_CSV, encoding="utf-8") as f:
            rdr = csv.reader(f)
            next(rdr)
            for uid, _, vec_s in rdr:
                if uid not in self.acc:
                    continue
                if uid_to_val(uid) ^ (self.split == "val"):
                    continue

                m = self.mask[uid]
                j = idx_counter[uid]
                idx_counter[uid] += 1
                if m is None or j >= len(m) or m[j] == 0:
                    continue

                vec = parse_vec(vec_s, POST_DIM)
                if vec is None:
                    continue
                post_t = torch.from_numpy(vec)
                acc_np = self.acc[uid].astype(np.float32)
                acc_t = torch.from_numpy(acc_np)

                # 正例
                yield post_t, acc_t, torch.tensor(1.0)

                # ノイズ負例
                noise = np.random.normal(0, NOISE_STD, acc_np.shape).astype(
                    np.float32
                )
                acc_noise = torch.from_numpy(l2_norm(acc_np + noise))
                yield post_t, acc_noise, torch.tensor(0.0)

                # ランダム負例
                for _ in range(NEG_RATIO):
                    neg_uid = random.choice(self.uids)
                    while neg_uid == uid:
                        neg_uid = random.choice(self.uids)
                    acc_neg = torch.from_numpy(
                        self.acc[neg_uid].astype(np.float32)
                    )
                    yield post_t, acc_neg, torch.tensor(0.0)


# ─────────── model ───────────
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
        return self.net(torch.cat([post_vec, acc_vec], 1)).squeeze(1)


# ─────────── train loop ───────────
def train(resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    tr_ds = MaskedStreamDataset("train")
    va_ds = MaskedStreamDataset("val")
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)

    model = PairClassifier(POST_DIM, tr_ds.rw_dim).to(device)
    optim_ = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    crit = nn.BCEWithLogitsLoss()

    start_ep, best_val, patience = 1, float("inf"), 0
    if resume and os.path.isfile(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ck["model_state"])
        optim_.load_state_dict(ck["optim_state"])
        start_ep = ck["epoch"] + 1
        best_val = ck["best_val"]
        patience = ck["patience"]
        print(f"[Resume] epoch={ck['epoch']} best_val={best_val:.4f}")

    for ep in range(start_ep, EPOCHS + 1):
        # ---- training ----
        model.train()
        tr_loss = n_tr = 0
        for p, a, lbl in tr_loader:
            p, a, lbl = p.to(device), a.to(device), lbl.to(device)
            loss = crit(model(p, a), lbl)
            optim_.zero_grad()
            loss.backward()
            optim_.step()
            tr_loss += loss.item() * p.size(0)
            n_tr += p.size(0)
        tr_loss /= n_tr

        # ---- validation ----
        model.eval()
        va_loss = n_va = 0
        with torch.no_grad():
            for p, a, lbl in va_loader:
                p, a, lbl = p.to(device), a.to(device), lbl.to(device)
                va_loss += crit(model(p, a), lbl).item() * p.size(0)
                n_va += p.size(0)
        va_loss /= n_va

        print(f"Ep{ep:03d} train={tr_loss:.4f} val={va_loss:.4f}")

        # ---- early stopping / ckpt ----
        if va_loss < best_val - MIN_DELTA:
            best_val, patience = va_loss, 0
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optim_state": optim_.state_dict(),
                    "best_val": best_val,
                    "patience": patience,
                },
                CKPT_PATH,
            )
            print("  ✔ checkpoint saved")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break
    print(f"[Done] best_val={best_val:.4f}")


# ─────────── entry ───────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.resume)