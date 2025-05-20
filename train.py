#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
set_transformer_train_debias.py
────────────────────────────────────────────────────────────
  • 列マスク (-1) で不要列を除外
  • BCE + label-smoothing (ε=0.1)
  • pos_weight を temp で割って人気列の勾配を抑制
  • CosineAnnealingLR で収束を滑らかに
────────────────────────────────────────────────────────────
"""

import os, csv, argparse
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from model import SetToVectorPredictor

# ─── パス既定 ─────────────────────────────────
ROOT = "/Users/oobayashikoushin/Enishi_system"
DEF_DATA  = f"{ROOT}/data_processed/follow_dataset.pt"
DEF_MODEL = f"{ROOT}/saved_models/set_transformer_follow_predictor.pt"
DEF_FREQ  = f"{ROOT}/account_freq.csv"

# ─── ハイパラ ─────────────────────────────────
POST_DIM=3072; ENC_DIM=512; N_HEADS=4; N_LAYERS=2; DROPOUT=0.1
BATCH=64; EPOCHS=500
LR=1e-5; WD=1e-5
VAL_SPLIT=.1; PATIENCE=20; MIN_DELTA=1e-4
BASE_T=1.; ALPHA=1.3; MAX_TEMP=50.
SMOOTH_EPS=0.1; POSW_MAX=10.

# ─── Dataset ─────────────────────────────────
class FollowDS(Dataset):
    def __init__(self, pt):
        obj = torch.load(pt)
        self.data = obj["dataset"]; self.acc = obj["all_account_list"]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x,t,_ = self.data[i]; return x,t

def collate(b):
    posts=[p[0] for p in b]; targ=torch.stack([p[1] for p in b])
    lens=torch.tensor([p.size(0) for p in posts]); M=lens.max().item()
    pad_posts=torch.nn.utils.rnn.pad_sequence(posts,batch_first=True)
    pad_mask =(torch.arange(M).unsqueeze(0)>=lens.unsqueeze(1))
    return pad_posts,pad_mask,targ

# ─── 温度テンソル ─────────────────────────────
def make_temp(csv_path, acc_list, alpha, max_t, device):
    freq={r[0]:max(1.,float(r[1])) for r in csv.reader(open(csv_path))}
    arr=np.array([freq.get(a,1.) for a in acc_list], np.float32)
    t=BASE_T*np.power(arr/arr.mean(), alpha); t=np.clip(t,.1,max_t)
    return torch.tensor(t, dtype=torch.float32, device=device)

# ─── soft-label helper ─────────────────────────
def smooth_target(y, eps=SMOOTH_EPS):
    y = y.float()
    pos = (y == 1)
    neg = (y == 0)
    n_neg = neg.sum(1, keepdim=True).clamp(min=1)
    prob_neg = eps / n_neg
    y_soft = torch.zeros_like(y)
    y_soft[pos] = 1.0 - eps
    y_soft[neg] = prob_neg.expand_as(y)[neg]
    # 無視列 (-1) は 0 のまま
    return y_soft

# ─── train ────────────────────────────────────
def train(cfg):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print("[Device]", device)

    ds = FollowDS(cfg.data)
    n_val = int(len(ds)*VAL_SPLIT); n_tr = len(ds)-n_val
    tr_set, va_set = random_split(ds, [n_tr, n_val])
    trL = DataLoader(tr_set, batch_size=BATCH, shuffle=True,  collate_fn=collate)
    vaL = DataLoader(va_set, batch_size=BATCH, shuffle=False, collate_fn=collate)

    temp = make_temp(cfg.freq, ds.acc, ALPHA, MAX_TEMP, device)

    # 列ごとの pos_weight
    full_t = torch.cat([t for _,t,_ in torch.load(cfg.data)["dataset"]])
    pos = (full_t==1).sum(0).float(); neg=(full_t==0).sum(0).float()
    pos_w = (neg / (pos+1e-6)).clamp(max=POSW_MAX).to(device)
    eff_pw = (pos_w / temp).clamp(max=POSW_MAX)          # 温度で割る
    eff_pw_row = eff_pw.unsqueeze(0)                     # (1,N) broadcast
    print(f"[eff_pos_weight] min={eff_pw.min():.2f} mean={eff_pw.mean():.2f} max={eff_pw.max():.2f}")

    model = SetToVectorPredictor(POST_DIM,ENC_DIM,len(ds.acc),
                                 N_HEADS,N_LAYERS,DROPOUT).to(device)
    if cfg.resume and os.path.isfile(cfg.model):
        model.load_state_dict(torch.load(cfg.model,map_location=device,weights_only=True))
        print("[Resume] loaded", cfg.model)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best, patience = 1e9, 0
    for ep in range(EPOCHS):
        # ---------- train ----------
        model.train(); num=den=0.
        for x,m,y in tqdm(trL,desc=f"Ep{ep+1}[train]",leave=False):
            x,m,y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()

            soft = smooth_target(y)                       # (B,N)
            logit,_ = model(x,m)
            bce = nn.functional.binary_cross_entropy_with_logits(
                      logit, soft, weight=eff_pw_row.expand_as(y),
                      reduction='none')

            mask = (y != -1).float()
            numer = (bce * mask).sum()
            denom = mask.sum()
            (numer/denom).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); num += numer.item(); den += denom.item()
        sch.step(); train_loss = num/den

        # ---------- val ----------
        model.eval(); num=den=0.
        with torch.no_grad():
            for x,m,y in vaL:
                x,m,y = x.to(device), m.to(device), y.to(device)
                soft = smooth_target(y)
                logit,_ = model(x,m)
                bce = nn.functional.binary_cross_entropy_with_logits(
                          logit, soft,
                          weight=eff_pw_row.expand_as(y),
                          reduction='none')
                mask = (y!=-1).float()
                num += (bce*mask).sum().item()
                den += mask.sum().item()
        val_loss = num/den
        print(f"Ep {ep+1}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best - MIN_DELTA:
            best, patience = val_loss, 0
            os.makedirs(os.path.dirname(cfg.model), exist_ok=True)
            torch.save(model.state_dict(), cfg.model)
            print("  ✔ saved best →", cfg.model)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break
    print(f"[Done] best val = {best:.4f}")

# ─── CLI ─────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default=DEF_DATA)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--freq",  default=DEF_FREQ)
    ap.add_argument("--resume", action="store_true")
    cfg = ap.parse_args()
    train(cfg)