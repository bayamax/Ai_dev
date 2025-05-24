#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pair_classifier_masked.py
距離相関マスク(1/0) を参照し、mask==1 の投稿だけで 0/1 BCE 学習。
"""

import os, csv, random, argparse, collections, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# ----- paths -----
VAST = "/workspace/edit_agent/vast"
POSTS_CSV = os.path.join(VAST, "aggregated_posting_vectors.csv")
ACCS_NPY = os.path.join(VAST, "account_vectors.npy")
MASK_DIR = os.path.join(VAST, "dcor_masks")
CKPT     = os.path.join(VAST, "pair_classifier_masked.ckpt")
# ----- hyper -----
POST_DIM, BATCH, LR, NEG_RATIO, NOISE_STD = 3072, 128, 1e-4, 5, 0.2
EPOCHS, PATIENCE, DROPOUT = 500, 15, 0.1
VAL_RATIO = 0.1
# ----- util -----
def parse_vec(s,d):
    s=s.strip(); null=("[]",'\"[]\"')
    if not s or s in null: return None
    if s.startswith('"[') and s.endswith(']"'): s=s[1:-1]
    l,r=s.find('['),s.rfind(']');  s=s[l+1:r] if 0<=l<r else s
    v=np.fromstring(s.replace(',',' '),sep=' ',dtype=np.float32)
    return v if v.size==d else None
def l2(v): n=np.linalg.norm(v); return v/n if n else v
def uid_to_val(uid): import hashlib,math
    return (int(hashlib.md5(uid.encode()).hexdigest(),16)%10000)/10000 < VAL_RATIO
# ----- dataset -----
class MaskedStream(IterableDataset):
    def __init__(self, split="train"):
        self.acc=np.load(ACCS_NPY,allow_pickle=True).item()
        self.uids=list(self.acc.keys()); self.rw_dim=len(next(iter(self.acc.values())))
        self.split=split
        self.mask={u:(np.load(os.path.join(MASK_DIR,f"{u}.npy")) if os.path.exists(os.path.join(MASK_DIR,f"{u}.npy")) else None)
                   for u in self.uids}
    def __iter__(self):
        idx=collections.defaultdict(int)
        with open(POSTS_CSV,encoding='utf-8') as f:
            rdr=csv.reader(f); next(rdr)
            for uid,_,vec in rdr:
                if uid not in self.acc: continue
                if uid_to_val(uid) ^ (self.split=="val"): continue
                m=self.mask[uid]; j=idx[uid]; idx[uid]+=1
                if m is None or j>=len(m) or m[j]==0: continue
                x=parse_vec(vec,POST_DIM);  y=self.acc[uid].astype(np.float32)
                if x is None: continue
                post=torch.from_numpy(x)
                acc=torch.from_numpy(y)
                yield post,acc,torch.tensor(1.)
                # ノイズ負例
                noise=l2(y+np.random.normal(0,NOISE_STD,size=y.shape).astype(np.float32))
                yield post,torch.from_numpy(noise),torch.tensor(0.)
                # ランダム負例
                for _ in range(NEG_RATIO):
                    neg_uid=random.choice(self.uids)
                    while neg_uid==uid: neg_uid=random.choice(self.uids)
                    yield post,torch.from_numpy(self.acc[neg_uid].astype(np.float32)),torch.tensor(0.)
# ----- model -----
class Pair(nn.Module):
    def __init__(self,pd,rd,hid=512,drop=DROPOUT):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(pd+rd,hid),nn.ReLU(True),nn.Dropout(drop),nn.Linear(hid,1))
    def forward(self,p,a): return self.f(torch.cat([p,a],1)).squeeze(1)
# ----- train -----
def main(resume=False):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr,va=MaskedStream("train"),MaskedStream("val")
    dl_tr,dl_va=DataLoader(tr,batch_size=BATCH),DataLoader(va,batch_size=BATCH)
    model=Pair(POST_DIM,tr.rw_dim).to(dev)
    opt=optim.AdamW(model.parameters(),lr=LR)
    crit=nn.BCEWithLogitsLoss()
    best=float("inf"); wait=0; start=1
    if resume and os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=dev)
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optim_state"])
        best,wait,start=ck["best_val"],ck["patience"],ck["epoch"]+1
    for ep in range(start,EPOCHS+1):
        # train
        model.train(); tl=n=0
        for p,a,l in dl_tr:
            p,a,l=p.to(dev),a.to(dev),l.to(dev)
            loss=crit(model(p,a),l); opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()*p.size(0); n+=p.size(0)
        tl/=n
        # val
        model.eval(); vl=n=0
        with torch.no_grad():
            for p,a,l in dl_va:
                p,a,l=p.to(dev),a.to(dev),l.to(dev)
                vl+=crit(model(p,a),l).item()*p.size(0); n+=p.size(0)
        vl/=n
        print(f"Ep{ep:03d} train={tl:.4f} val={vl:.4f}")
        if vl<best-1e-4:
            best,wait=vl,0
            torch.save({"epoch":ep,"model_state":model.state_dict(),
                        "optim_state":opt.state_dict(),
                        "best_val":best,"patience":wait},CKPT)
            print("  ✔ saved")
        else:
            wait+=1
            if wait>=PATIENCE: print("Early stop"); break
    print(f"best_val={best:.4f}")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--resume",action="store_true")
    main(ap.parse_args().resume)