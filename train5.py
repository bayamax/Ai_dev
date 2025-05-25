#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_refine_rw_agg.py   (fixed output-dim mismatch)
"""

import os, csv, random, pickle, importlib.util, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

# ───────── ディレクトリ固定 ─────────
BASE       = "/workspace/edit_agent"
TRAIN_DIR  = os.path.join(BASE, "train")
VAST       = os.path.join(BASE, "vast")

POSTS      = os.path.join(VAST, "aggregated_posting_vectors.csv")
ACCS       = os.path.join(VAST, "account_vectors.npy")

POST2RW_PY   = os.path.join(TRAIN_DIR, "train4.py")
POST2RW_CKPT = os.path.join(VAST, "post2rw.ckpt")

CACHE_PKL = os.path.join(VAST, "pred_rw_seq.pkl")
CKPT_OUT  = os.path.join(VAST, "refine_rw_agg.ckpt")

# ───────── ハイパラ ─────────
BATCH_POST = 1024
D_MODEL, N_LAYER, N_HEAD = 512, 4, 8
BATCH_UID  = 64
EPOCHS     = 100
LR, WD     = 3e-4, 1e-5
DROP       = 0.1
LAMBDA     = 0.5
PATIENCE   = 10
MIN_DELTA  = 1e-4

# ───────── Post→RW クラス / 関数 import ─────────
spec = importlib.util.spec_from_file_location("p2r", POST2RW_PY)
p2r = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2r)
Post2RW, parse_vec, POST_DIM = p2r.Post2RW, p2r.parse_vec, p2r.POST_DIM

# ───────── 真アカウント RW 読み込み (rw_dim を確定) ─────────
acc_dict = np.load(ACCS, allow_pickle=True).item()
rw_dim   = next(iter(acc_dict.values())).shape[0]

# ───────── Post→RW モデル構築 & 重みロード ─────────
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", dev)

post2rw = Post2RW(
    POST_DIM,
    rw_dim,                       # ← ★ 出力次元は 128 (= rw_dim)
    hidden=p2r.HIDDEN_DIMS,
    drop=p2r.DROPOUT,
).to(dev)
post2rw.load_state_dict(torch.load(POST2RW_CKPT, map_location=dev)["model"])
post2rw.eval()
print("[Import] Post2RW from", POST2RW_PY)

# ───────── ① 推定 RW 集合キャッシュ ─────────
if os.path.exists(CACHE_PKL):
    uid_seqs = pickle.load(open(CACHE_PKL, "rb"))
    print("loaded cache:", len(uid_seqs), "UID")
else:
    uid_seqs = {}
    with open(POSTS, encoding="utf-8") as f:
        rdr = csv.reader(f); next(rdr)
        buf_x, buf_u = [], []
        for uid, _, vec in tqdm(rdr, desc="Post→RW infer"):
            v = parse_vec(vec, POST_DIM);  # train4.py 同梱の parse_vec
            if v is None: continue
            buf_x.append(v); buf_u.append(uid)
            if len(buf_x) == BATCH_POST:
                t = torch.tensor(buf_x, device=dev)
                pred = post2rw(t).cpu().numpy()
                for u, r in zip(buf_u, pred):
                    uid_seqs.setdefault(u, []).append(r)
                buf_x, buf_u = [], []
        if buf_x:
            t = torch.tensor(buf_x, device=dev)
            pred = post2rw(t).cpu().numpy()
            for u, r in zip(buf_u, pred):
                uid_seqs.setdefault(u, []).append(r)
    pickle.dump(uid_seqs, open(CACHE_PKL, "wb"))
    print("saved cache:", CACHE_PKL)

# ───────── 共通 UID 抽出 ─────────
common_uids = [u for u in uid_seqs if u in acc_dict]
print("common UID:", len(common_uids))

# ───────── Dataset iterator ─────────
def batch_iter(uids, bs):
    for i in range(0, len(uids), bs):
        yield uids[i:i+bs]

# ───────── Attention Aggregator + MLP ─────────
class Aggregator(nn.Module):
    def __init__(self, in_dim=rw_dim, d=D_MODEL):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d, bias=False), nn.LayerNorm(d))
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=N_HEAD, dim_feedforward=d*4,
            dropout=DROP, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=N_LAYER)
        self.cls = nn.Parameter(torch.randn(1,1,d))
    def forward(self,x,mask):
        B=x.size(0)
        z=self.proj(x)
        cls=self.cls.expand(B,-1,-1)
        z=torch.cat([cls,z],1)
        if mask is not None:
            mask=torch.cat([torch.zeros(B,1,dtype=torch.bool,device=z.device),mask],1)
        h=self.enc(z,src_key_padding_mask=mask)
        return h[:,0]

class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg  = Aggregator()
        self.head = nn.Linear(D_MODEL, rw_dim)
    def forward(self,x,mask):
        return self.head(self.agg(x,mask))

model = RefineNet().to(dev)
opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
mse   = nn.MSELoss(); cos = nn.CosineEmbeddingLoss()

# ───────── ② 学習ループ ─────────
best, wait = 1e9, 0
for ep in range(1, EPOCHS+1):
    # ---- train ----
    model.train(); tl=n=0
    random.shuffle(common_uids)
    for batch in batch_iter(common_uids, BATCH_UID):
        seqs=[uid_seqs[u] for u in batch]
        L=max(len(s) for s in seqs)
        pad=np.zeros((len(batch),L,rw_dim),np.float32)
        msk=np.ones((len(batch),L),bool)
        for i,s in enumerate(seqs):
            pad[i,:len(s)]=s; msk[i,:len(s)]=False
        x=torch.tensor(pad,device=dev)
        mask=torch.tensor(msk,device=dev)
        y=torch.tensor([acc_dict[u] for u in batch],device=dev)
        out=model(x,mask)
        loss=(1-LAMBDA)*mse(out,y)+LAMBDA*cos(out,y,torch.ones(len(batch),device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
        tl+=loss.item()*len(batch); n+=len(batch)
    tl/=n

    # ---- val ----
    model.eval(); vl=n=0
    with torch.no_grad():
        for batch in batch_iter(common_uids, BATCH_UID):
            seqs=[uid_seqs[u] for u in batch]
            L=max(len(s) for s in seqs)
            pad=np.zeros((len(batch),L,rw_dim),np.float32)
            msk=np.ones((len(batch),L),bool)
            for i,s in enumerate(seqs):
                pad[i,:len(s)]=s; msk[i,:len(s)]=False
            x=torch.tensor(pad,device=dev)
            mask=torch.tensor(msk,device=dev)
            y=torch.tensor([acc_dict[u] for u in batch],device=dev)
            out=model(x,mask)
            loss=(1-LAMBDA)*mse(out,y)+LAMBDA*cos(out,y,torch.ones(len(batch),device=dev))
            vl+=loss.item()*len(batch); n+=len(batch)
    vl/=n
    print(f"Ep{ep:03d}  train={tl:.4f}  val={vl:.4f}")

    if vl < best - MIN_DELTA:
        best, wait = vl, 0
        torch.save({"epoch":ep,"model":model.state_dict(),"best":best}, CKPT_OUT)
        print("  ✔ saved", CKPT_OUT)
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping."); break

print("best_val =", best)