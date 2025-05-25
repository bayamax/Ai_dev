#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_refine_rw_agg.py   (train5.py)
投稿 → Post2RW → 推定 RW 集合
→ Attention CLS-Pooling → MLP → 真 RW を回帰

* `train()` 関数化 + __main__ ガード で
  import 時に学習が走らないように修正
"""

import os, csv, random, pickle, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import importlib.util, argparse

# ────────── パス固定 ──────────
BASE       = "/workspace/edit_agent"
TRN        = os.path.join(BASE, "train")
VAST       = os.path.join(BASE, "vast")

POSTS_CSV  = os.path.join(VAST, "aggregated_posting_vectors.csv")
ACCS_NPY   = os.path.join(VAST, "account_vectors.npy")

POST2RW_PY = os.path.join(TRN,  "train4.py")
POST2RW_CK = os.path.join(VAST, "post2rw.ckpt")

CACHE_PKL  = os.path.join(VAST, "pred_rw_seq.pkl")
CKPT_OUT   = os.path.join(VAST, "refine_rw_agg.ckpt")

# ────────── ハイパラ ──────────
BATCH_POST   = 1024
D_MODEL      = 512
N_LAYER      = 4
N_HEAD       = 8
BATCH_UID    = 64
EPOCHS       = 100
LR           = 3e-4
WD           = 1e-5
DROP         = 0.1
LAMBDA       = 0.5        # MSE vs Cos
PATIENCE     = 10
MIN_DELTA    = 1e-4

# ────────── util ──────────
def import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

# ─────────────────────────────────────────────────────────────
def train(resume=False):
    # ---------- import train4.py ----------
    p2r = import_from(POST2RW_PY, "p2r")
    Post2RW, parse_vec, POST_DIM = p2r.Post2RW, p2r.parse_vec, p2r.POST_DIM

    # ---------- device ----------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", dev)

    # ---------- load true RW ----------
    acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
    rw_dim   = next(iter(acc_dict.values())).shape[0]

    # ---------- Post→RW model ----------
    post2rw = Post2RW(POST_DIM, rw_dim,
                      hidden=p2r.HIDDEN_DIMS, drop=p2r.DROPOUT).to(dev)
    post2rw.load_state_dict(torch.load(POST2RW_CK, map_location=dev)["model"])
    post2rw.eval()

    # ---------- 推定 RW キャッシュ ----------
    if os.path.exists(CACHE_PKL):
        uid_seqs = pickle.load(open(CACHE_PKL, "rb"))
        print("loaded cache:", len(uid_seqs), "UID")
    else:
        uid_seqs = {}
        with open(POSTS_CSV, encoding="utf-8") as f:
            rdr = csv.reader(f); next(rdr)
            buf_x, buf_u = [], []
            for uid, _, vec_s in tqdm(rdr, desc="Post→RW infer"):
                v = parse_vec(vec_s, POST_DIM)
                if v is None: continue
                buf_x.append(v); buf_u.append(uid)
                if len(buf_x) == BATCH_POST:
                    with torch.no_grad():
                        t = torch.as_tensor(np.asarray(buf_x, np.float32), device=dev)
                        pred = post2rw(t).detach().cpu().numpy()
                    for u, r in zip(buf_u, pred):
                        uid_seqs.setdefault(u, []).append(r)
                    buf_x, buf_u = [], []
            if buf_x:
                with torch.no_grad():
                    t = torch.as_tensor(np.asarray(buf_x, np.float32), device=dev)
                    pred = post2rw(t).detach().cpu().numpy()
                for u, r in zip(buf_u, pred):
                    uid_seqs.setdefault(u, []).append(r)
        pickle.dump(uid_seqs, open(CACHE_PKL, "wb"))
        print("saved cache:", CACHE_PKL)

    common = [u for u in uid_seqs if u in acc_dict]
    print("UID:", len(common))

    # ---------- mini-batch iterator ----------
    def batch_iter(seq, bs):
        for i in range(0, len(seq), bs):
            yield seq[i:i+bs]

    # ---------- Aggregator + MLP ----------
    class Aggregator(nn.Module):
        def __init__(self, d=D_MODEL):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(rw_dim, d, bias=False),
                                      nn.LayerNorm(d))
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=N_HEAD, dim_feedforward=d*4,
                dropout=DROP, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=N_LAYER)
            self.cls = nn.Parameter(torch.randn(1,1,d))
        def forward(self,x,mask):
            B = x.size(0)
            z = self.proj(x)
            z = torch.cat([self.cls.expand(B,-1,-1), z], 1)
            if mask is not None:
                mask = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=z.device),
                                  mask], 1)
            return self.enc(z, src_key_padding_mask=mask)[:,0]

    class RefineNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.agg  = Aggregator()
            self.head = nn.Linear(D_MODEL, rw_dim)
        def forward(self,x,mask):
            return self.head(self.agg(x,mask))

    model = RefineNet().to(dev)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    mse,cos = nn.MSELoss(), nn.CosineEmbeddingLoss()

    # ---------- resume ----------
    start_ep, best, wait = 1, 1e9, 0
    if resume and os.path.exists(CKPT_OUT):
        ck = torch.load(CKPT_OUT, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"] + 1
        best     = ck["best"]
        wait     = ck["wait"]
        print(f"[Resume] epoch {start_ep-1}  best {best:.4f}")

    # ---------- training loop ----------
    for ep in range(start_ep, EPOCHS+1):
        # --- train ---
        model.train(); tl=n=0
        random.shuffle(common)
        for batch in batch_iter(common, BATCH_UID):
            seqs=[uid_seqs[u] for u in batch]
            L=max(len(s) for s in seqs)
            pad=np.zeros((len(batch),L,rw_dim),np.float32)
            msk=np.ones((len(batch),L),bool)
            for i,s in enumerate(seqs):
                pad[i,:len(s)]=s; msk[i,:len(s)]=False
            x=torch.as_tensor(pad,device=dev)
            mask=torch.as_tensor(msk,device=dev)
            y=torch.as_tensor([acc_dict[u] for u in batch],device=dev)
            out=model(x,mask)
            loss=(1-LAMBDA)*mse(out,y)+LAMBDA*cos(out,y,torch.ones(len(batch),device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()*len(batch); n+=len(batch)
        tl/=n

        # --- val ---
        model.eval(); vl=n=0
        with torch.no_grad():
            for batch in batch_iter(common, BATCH_UID):
                seqs=[uid_seqs[u] for u in batch]
                L=max(len(s) for s in seqs)
                pad=np.zeros((len(batch),L,rw_dim),np.float32)
                msk=np.ones((len(batch),L),bool)
                for i,s in enumerate(seqs):
                    pad[i,:len(s)]=s; msk[i,:len(s)]=False
                x=torch.as_tensor(pad,device=dev)
                mask=torch.as_tensor(msk,device=dev)
                y=torch.as_tensor([acc_dict[u] for u in batch],device=dev)
                out=model(x,mask)
                loss=(1-LAMBDA)*mse(out,y)+LAMBDA*cos(out,y,torch.ones(len(batch),device=dev))
                vl+=loss.item()*len(batch); n+=len(batch)
        vl/=n
        print(f"Ep{ep:03d} train={tl:.4f} val={vl:.4f}")

        # --- ckpt / early-stop ---
        if vl < best - MIN_DELTA:
            best, wait = vl, 0
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "opt":   opt.state_dict(),
                "best":  best,
                "wait":  wait,
            }, CKPT_OUT)
            print("  ✔ saved", CKPT_OUT)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping."); break

    print("best_val =", best)

# ─────────── main guard ───────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="resume from refine_rw_agg.ckpt")
    train(parser.parse_args().resume)