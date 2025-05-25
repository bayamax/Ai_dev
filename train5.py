#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_refine_rw_agg.py
────────────────────────────────────────────────────────
Step-A : Post→RW 回帰モデル (train4.py + post2rw.ckpt) で
         各投稿を推定 RW に変換し、UID ごとに “推定 RW の集合” を作成
Step-B : その集合を Attention Aggregator (CLS Pooling) で 512-d に圧縮し
         小 MLP で “真” のアカウント RW に回帰
         ─ 損失 = 0.5 * MSE + 0.5 * CosineEmbeddingLoss
----------------------------------------------------------------
生成物 : refine_rw_agg.ckpt
キャッシュ : pred_rw_seq.pkl  ← 推定 RW 集合を再利用して高速化
"""

import os, csv, argparse, random, pickle, importlib.util
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

# ───────── パス設定 ─────────
VAST = "/workspace/edit_agent/vast"
POSTS = os.path.join(VAST, "aggregated_posting_vectors.csv")
ACCS  = os.path.join(VAST, "account_vectors.npy")

# ★ Post→RW モデルを定義している学習ファイル (train4.py) へパスを合わせる
POST2RW_PY   = "/workspace/edit_agent/tra/train4.py"
POST2RW_CKPT = os.path.join(VAST, "post2rw.ckpt")

CACHE_PKL = os.path.join(VAST, "pred_rw_seq.pkl")
CKPT_OUT  = os.path.join(VAST, "refine_rw_agg.ckpt")

# ───────── ハイパラ ─────────
BATCH_POST = 1024      # 投稿→RW 推論バッチ
D_MODEL    = 512
N_LAYER    = 4
N_HEAD     = 8
BATCH_UID  = 64        # UID バッチ
EPOCHS     = 100
LR         = 3e-4
WEIGHT_DECAY = 1e-5
DROP       = 0.1
LAMBDA     = 0.5       # MSE / Cosine の重み
PATIENCE   = 10
MIN_DELTA  = 1e-4

# ───────── Post→RW モデル読み込み ─────────
spec = importlib.util.spec_from_file_location("p2r", POST2RW_PY)
p2r = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2r)
Post2RW   = p2r.Post2RW
parse_vec = p2r.parse_vec
POST_DIM  = p2r.POST_DIM

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", dev)

post2rw = Post2RW(
    POST_DIM,
    p2r.HIDDEN_DIMS[-1] if isinstance(p2r.HIDDEN_DIMS, list) else p2r.HIDDEN_DIMS,
    p2r.HIDDEN_DIMS,
    p2r.DROPOUT,
).to(dev)
post2rw.load_state_dict(torch.load(POST2RW_CKPT, map_location=dev)["model"])
post2rw.eval()

# ───────── ① 投稿 → 推定 RW 集合作成 (キャッシュ) ─────────
if os.path.exists(CACHE_PKL):
    uid_seqs = pickle.load(open(CACHE_PKL, "rb"))
    print("loaded cache:", len(uid_seqs), "UID")
else:
    uid_seqs = {}
    with open(POSTS, encoding="utf-8") as f:
        rdr = csv.reader(f); next(rdr)
        buf_x, buf_u = [], []
        for uid, _, vec_s in tqdm(rdr, desc="Post→RW infer"):
            v = parse_vec(vec_s, POST_DIM)
            if v is None: continue
            buf_x.append(v); buf_u.append(uid)
            if len(buf_x) == BATCH_POST:
                t = torch.tensor(buf_x, device=dev)
                pred = post2rw(t).cpu().numpy()
                for u, r in zip(buf_u, pred):
                    uid_seqs.setdefault(u, []).append(r)
                buf_x, buf_u = [], []
        # 残り
        if buf_x:
            t = torch.tensor(buf_x, device=dev)
            pred = post2rw(t).cpu().numpy()
            for u, r in zip(buf_u, pred):
                uid_seqs.setdefault(u, []).append(r)
    pickle.dump(uid_seqs, open(CACHE_PKL, "wb"))
    print("saved cache:", CACHE_PKL)

# ───────── 真 RW 読み込み ─────────
acc_dict = np.load(ACCS, allow_pickle=True).item()
common_uids = [u for u in uid_seqs if u in acc_dict]
print("common UID:", len(common_uids))
rw_dim = acc_dict[common_uids[0]].shape[0]

# ───────── Dataset generator ─────────
def uid_batch_iter(uids, bs):
    for i in range(0, len(uids), bs):
        yield uids[i:i+bs]

# ───────── モデル定義 ─────────
class Aggregator(nn.Module):
    def __init__(self, in_dim=rw_dim, d=D_MODEL, nlayer=N_LAYER, nhead=N_HEAD, drop=DROP):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d, bias=False), nn.LayerNorm(d))
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d*4,
            dropout=drop, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=nlayer)
        self.cls = nn.Parameter(torch.randn(1, 1, d))
    def forward(self, x, mask):
        B = x.size(0)
        z = self.proj(x)
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, z], 1)
        if mask is not None:
            mask = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=z.device), mask], 1)
        h = self.enc(z, src_key_padding_mask=mask)
        return h[:,0]            # CLS

class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg  = Aggregator()
        self.head = nn.Linear(D_MODEL, rw_dim)
    def forward(self, x, mask):
        return self.head(self.agg(x, mask))

model = RefineNet().to(dev)
opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
mse   = nn.MSELoss()
cos   = nn.CosineEmbeddingLoss()

best, wait = 1e9, 0
for ep in range(1, EPOCHS+1):
    # ---- train ----
    model.train(); tl=n=0
    random.shuffle(common_uids)
    for uid_batch in uid_batch_iter(common_uids, BATCH_UID):
        seqs = [uid_seqs[u] for u in uid_batch]
        max_len = max(len(s) for s in seqs)
        pad = np.zeros((len(uid_batch), max_len, rw_dim), np.float32)
        pad_mask = np.ones((len(uid_batch), max_len), bool)
        for i, s in enumerate(seqs):
            pad[i,:len(s)] = s
            pad_mask[i,:len(s)] = False
        x = torch.tensor(pad, device=dev)
        mask = torch.tensor(pad_mask, device=dev)
        y = torch.tensor([acc_dict[u] for u in uid_batch], device=dev)
        out = model(x, mask)
        loss = (1-LAMBDA)*mse(out,y) + LAMBDA*cos(out,y,torch.ones(len(uid_batch),device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item()*len(uid_batch); n += len(uid_batch)
    tl /= n

    # ---- validation (フル UID) ----
    model.eval(); vl=n=0
    with torch.no_grad():
        for uid_batch in uid_batch_iter(common_uids, BATCH_UID):
            seqs = [uid_seqs[u] for u in uid_batch]
            max_len = max(len(s) for s in seqs)
            pad = np.zeros((len(uid_batch), max_len, rw_dim), np.float32)
            pad_mask = np.ones((len(uid_batch), max_len), bool)
            for i, s in enumerate(seqs):
                pad[i,:len(s)] = s
                pad_mask[i,:len(s)] = False
            x = torch.tensor(pad, device=dev)
            mask = torch.tensor(pad_mask, device=dev)
            y = torch.tensor([acc_dict[u] for u in uid_batch], device=dev)
            out = model(x, mask)
            loss = (1-LAMBDA)*mse(out,y) + LAMBDA*cos(out,y,torch.ones(len(uid_batch),device=dev))
            vl += loss.item()*len(uid_batch); n += len(uid_batch)
    vl /= n
    print(f"Ep{ep:03d} train={tl:.4f} val={vl:.4f}")

    # ---- Early-Stopping ----
    if vl < best - MIN_DELTA:
        best, wait = vl, 0
        torch.save({"epoch": ep, "model": model.state_dict(), "best": best}, CKPT_OUT)
        print("  ✔ saved", CKPT_OUT)
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping."); break

print("best_val =", best)