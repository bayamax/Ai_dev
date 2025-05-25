#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_refine_rw_agg.py
────────────────────────────────────────────────────────
投稿 → Post2RW → {推定RW集合} → AttentionAggregator → RefineMLP → 真RW
  • 投稿平均は一切せず、集合を Attention でプーリング
  • 26 万投稿規模でも RAM ≈ 30 MB に収まる
"""

import os, csv, argparse, numpy as np, torch, torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import importlib.util, pickle, math

# ───────── パス設定 ─────────
VAST = "/workspace/edit_agent/vast"
POSTS = os.path.join(VAST, "aggregated_posting_vectors.csv")
ACCS  = os.path.join(VAST, "account_vectors.npy")

POST2RW_PY  = "/workspace/edit_agent/train/train_post2rw.py"
POST2RW_CKPT= os.path.join(VAST, "post2rw.ckpt")

CACHE_PKL = os.path.join(VAST, "pred_rw_seq.pkl")      # UID→list[np]
CKPT_OUT  = os.path.join(VAST, "refine_rw_agg.ckpt")

# ───────── ハイパラ ─────────
BATCH_POST = 1024          # 投稿→RW 推論バッチ
D_MODEL  = 512
N_LAYER  = 4
N_HEAD   = 8
EPOCHS   = 100
BATCH_UID= 64              # UID バッチ
LR       = 3e-4
WD       = 1e-5
DROP     = 0.1
LAMBDA   = 0.5
PATIENCE = 10
MIN_DELTA= 1e-4

# ───────── Post→RW モデル読み込み ─────────
spec = importlib.util.spec_from_file_location("p2r", POST2RW_PY)
p2r = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2r)
Post2RW = p2r.Post2RW; parse_vec = p2r.parse_vec; POST_DIM = p2r.POST_DIM

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", dev)

p2r_model = Post2RW(POST_DIM, p2r.HIDDEN_DIMS[-1] if isinstance(p2r.HIDDEN_DIMS,list)
                    else p2r.HIDDEN_DIMS, p2r.HIDDEN_DIMS, p2r.DROPOUT).to(dev)
p2r_model.load_state_dict(torch.load(POST2RW_CKPT, map_location=dev)["model"])
p2r_model.eval()

# ───────── ① 推定 RW 集合を作成 (or 読み込み) ─────────
if not os.path.exists(CACHE_PKL):
    uid_seqs={}
    with open(POSTS,encoding='utf-8') as f:
        rdr=csv.reader(f); next(rdr)
        buf_x, buf_u = [], []
        for uid,_,vec_s in tqdm(rdr,desc="Post→RW infer"):
            v=parse_vec(vec_s, POST_DIM)
            if v is None: continue
            buf_x.append(v); buf_u.append(uid)
            if len(buf_x)==BATCH_POST:
                t=torch.tensor(buf_x,device=dev)
                pred=p2r_model(t).cpu().numpy()
                for u,r in zip(buf_u,pred):
                    uid_seqs.setdefault(u,[]).append(r)
                buf_x,buf_u=[],[]
        if buf_x:
            t=torch.tensor(buf_x,device=dev)
            pred=p2r_model(t).cpu().numpy()
            for u,r in zip(buf_u,pred):
                uid_seqs.setdefault(u,[]).append(r)
    with open(CACHE_PKL,"wb") as fw: pickle.dump(uid_seqs,fw)
    print("saved cache:",CACHE_PKL)
else:
    uid_seqs = pickle.load(open(CACHE_PKL,"rb"))
    print("loaded cache", len(uid_seqs),"UID")

# ───────── 真RW 読み込み & 共通 UID 抽出 ─────────
acc_dict = np.load(ACCS,allow_pickle=True).item()
common = [u for u in uid_seqs if u in acc_dict]
print("common UID:", len(common))
rw_dim = acc_dict[common[0]].shape[0]

# ───────── dataset generator ─────────
def batch_iter(uids, batch):
    for i in range(0,len(uids),batch):
        yield uids[i:i+batch]

# ───────── Model 定義 ─────────
class Aggregator(nn.Module):
    def __init__(self, dim_in=rw_dim, d_model=D_MODEL,
                 n_layer=N_LAYER, n_head=N_HEAD, drop=DROP):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim_in,d_model,bias=False),
            nn.LayerNorm(d_model))
        enc=nn.TransformerEncoderLayer(d_model,n_head,d_model*4,
                                       dropout=drop,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(enc,n_layer)
        self.cls=nn.Parameter(torch.randn(1,1,d_model))
    def forward(self,x,mask):
        """
        x: (B,S,dim_in)  mask: (B,S)=True for pad
        """
        B=x.size(0)
        z=self.proj(x)
        cls=self.cls.expand(B,-1,-1)
        z=torch.cat([cls,z],1)
        if mask is not None:
            mask=torch.cat([torch.zeros(B,1,dtype=torch.bool,device=z.device),mask],1)
        h=self.enc(z,src_key_padding_mask=mask)
        return h[:,0]             # CLS

class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg=Aggregator()
        self.head=nn.Linear(D_MODEL,rw_dim)
    def forward(self,x,mask):     # x set → rw_pred
        return self.head(self.agg(x,mask))

model=RefineNet().to(dev)
opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
mse=nn.MSELoss(); cos=nn.CosineEmbeddingLoss()

best,pat=1e9,0
for ep in range(1,EPOCHS+1):
    # ---- train ----
    model.train(); tl=n=0
    random_idxs = np.random.permutation(len(common))
    for batch_uid in batch_iter(list(np.array(common)[random_idxs]), BATCH_UID):
        seqs=[uid_seqs[u] for u in batch_uid]
        max_len=max(len(s) for s in seqs)
        pad=np.zeros((len(batch_uid),max_len,rw_dim),np.float32)
        pad_mask=np.ones((len(batch_uid),max_len),bool)
        for i,s in enumerate(seqs):
            pad[i,:len(s)]=s; pad_mask[i,:len(s)]=False
        x=torch.tensor(pad,device=dev)
        mask=torch.tensor(pad_mask,device=dev)
        y=torch.tensor([acc_dict[u] for u in batch_uid],device=dev)
        out=model(x,mask)
        loss=(1-LAMBDA)*mse(out,y)+LAMBDA*cos(out,y,torch.ones(len(batch_uid),device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
        tl+=loss.item()*len(batch_uid); n+=len(batch_uid)
    tl/=n
    # ---- val (全 UID) ----
    model.eval(); vl=n=0
    with torch.no_grad():
        for batch_uid in batch_iter(common, BATCH_UID):
            seqs=[uid_seqs[u] for u in batch_uid]
            max_len=max(len(s) for s in seqs)
            pad=np.zeros((len(batch_uid),max_len,rw_dim),np.float32)
            pad_mask=np.ones((len(batch_uid),max_len),bool)
            for i,s in enumerate(seqs):
                pad[i,:len(s)]=s; pad_mask[i,:len(s)]=False
            x=torch.tensor(pad,device=dev)
            mask=torch.tensor(pad_mask,device=dev)
            y=torch.tensor([acc_dict[u] for u in batch_uid],device=dev)
            out=model(x,mask)
            loss=(1-LAMBDA)*mse(out,y)+LAMBDA*cos(out,y,torch.ones(len(batch_uid),device=dev))
            vl+=loss.item()*len(batch_uid); n+=len(batch_uid)
    vl/=n
    print(f"Ep{ep:03d} train={tl:.4f} val={vl:.4f}")
    if vl<best-MIN_DELTA:
        best,pat=vl,0
        torch.save({"epoch":ep,"model":model.state_dict(),"best":best},CKPT_OUT)
        print("  ✔ saved",CKPT_OUT)
    else:
        pat+=1
        if pat>=PATIENCE:
            print("Early stop"); break

print("best_val",best)
