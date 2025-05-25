#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_refine_rw_agg.py  (resume 対応版)
"""

import os, csv, random, pickle, argparse, importlib.util, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

# ---------- 固定パス ----------
BASE  = "/workspace/edit_agent"
TRN   = os.path.join(BASE, "train")
VAST  = os.path.join(BASE, "vast")
POSTS = os.path.join(VAST, "aggregated_posting_vectors.csv")
ACCS  = os.path.join(VAST, "account_vectors.npy")

POST2RW_PY = os.path.join(TRN, "train4.py")
POST2RW_CK = os.path.join(VAST,"post2rw.ckpt")

CACHE_PKL  = os.path.join(VAST,"pred_rw_seq.pkl")
CKPT_OUT   = os.path.join(VAST,"refine_rw_agg.ckpt")

# ---------- hyper ----------
BATCH_POST=1024;  D_MODEL,N_LAYER,N_HEAD=512,4,8
BATCH_UID =64;    EPOCHS=100
LR,WD=3e-4,1e-5;  DROP=0.1
LAMBDA=0.5;       PATIENCE=10; MIN_DELTA=1e-4

# ---------- argparse ----------
ap=argparse.ArgumentParser()
ap.add_argument("--resume",action="store_true",help="checkpoint から再開")
args=ap.parse_args()

# ---------- import train4.py ----------
spec=importlib.util.spec_from_file_location("p2r",POST2RW_PY)
p2r=importlib.util.module_from_spec(spec); spec.loader.exec_module(p2r)
Post2RW,parse_vec,POST_DIM=p2r.Post2RW,p2r.parse_vec,p2r.POST_DIM

# ---------- load true RW ----------
acc_dict=np.load(ACCS,allow_pickle=True).item()
rw_dim=next(iter(acc_dict.values())).shape[0]

# ---------- Post→RW model ----------
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
post2rw=Post2RW(POST_DIM,rw_dim,hidden=p2r.HIDDEN_DIMS,drop=p2r.DROPOUT).to(dev)
post2rw.load_state_dict(torch.load(POST2RW_CK,map_location=dev)["model"])
post2rw.eval()

# ---------- 推定 RW cache ----------
if os.path.exists(CACHE_PKL):
    uid_seqs=pickle.load(open(CACHE_PKL,"rb"))
else:
    uid_seqs={}
    with open(POSTS,encoding="utf-8") as f:
        rdr=csv.reader(f); next(rdr)
        buf_x,buf_u=[],[]
        for uid,_,vec in tqdm(rdr,desc="Post→RW infer"):
            v=parse_vec(vec,POST_DIM);  # skip bad row
            if v is None: continue
            buf_x.append(v); buf_u.append(uid)
            if len(buf_x)==BATCH_POST:
                with torch.no_grad():
                    t=torch.as_tensor(np.asarray(buf_x,np.float32),device=dev)
                    pr=post2rw(t).detach().cpu().numpy()
                for u,r in zip(buf_u,pr):
                    uid_seqs.setdefault(u,[]).append(r)
                buf_x,buf_u=[],[]
        if buf_x:
            with torch.no_grad():
                t=torch.as_tensor(np.asarray(buf_x,np.float32),device=dev)
                pr=post2rw(t).detach().cpu().numpy()
            for u,r in zip(buf_u,pr):
                uid_seqs.setdefault(u,[]).append(r)
    pickle.dump(uid_seqs,open(CACHE_PKL,"wb"))

common=list(uid_seqs.keys()&acc_dict.keys())
print("UID:",len(common))

# ---------- mini dataset iterator ----------
def batch_iter(seq,bs):
    for i in range(0,len(seq),bs): yield seq[i:i+bs]

# ---------- Aggregator+MLP ----------
class Aggregator(nn.Module):
    def __init__(self,d=D_MODEL):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(rw_dim,d,bias=False),nn.LayerNorm(d))
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=N_HEAD,
                                       dim_feedforward=d*4,dropout=DROP,
                                       batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(enc,N_LAYER)
        self.cls=nn.Parameter(torch.randn(1,1,d))
    def forward(self,x,mask):
        B=x.size(0)
        z=self.proj(x)
        z=torch.cat([self.cls.expand(B,-1,-1),z],1)
        if mask is not None:
            mask=torch.cat([torch.zeros(B,1,dtype=torch.bool,device=z.device),mask],1)
        return self.enc(z,src_key_padding_mask=mask)[:,0]

class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg=Aggregator()
        self.head=nn.Linear(D_MODEL,rw_dim)
    def forward(self,x,mask): return self.head(self.agg(x,mask))

model=RefineNet().to(dev)
opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
mse,cos=nn.MSELoss(),nn.CosineEmbeddingLoss()

# ---------- resume ----------
start_ep,best,wait=1,1e9,0
if args.resume and os.path.exists(CKPT_OUT):
    ck=torch.load(CKPT_OUT,map_location=dev)
    model.load_state_dict(ck["model"])
    if "opt" in ck: opt.load_state_dict(ck["opt"])
    start_ep   = ck.get("epoch",0)+1
    best       = ck.get("best",best)
    wait       = ck.get("wait",0)
    print(f"[Resume] epoch {start_ep-1} best {best:.4f}")

# ---------- training loop ----------
for ep in range(start_ep,EPOCHS+1):
    # train
    model.train(); tl=n=0
    random.shuffle(common)
    for batch in batch_iter(common,BATCH_UID):
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

    # val
    model.eval(); vl=n=0
    with torch.no_grad():
        for batch in batch_iter(common,BATCH_UID):
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

    # save / early‐stop
    if vl<best-MIN_DELTA:
        best,wait=vl,0
        torch.save({"epoch":ep,"model":model.state_dict(),
                    "opt":opt.state_dict(),"best":best,"wait":wait},CKPT_OUT)
        print("  ✔ saved",CKPT_OUT)
    else:
        wait+=1
        if wait>=PATIENCE: print("Early stopping"); break

print("best_val",best)