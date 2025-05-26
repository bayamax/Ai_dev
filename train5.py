#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_multitask_attn.py  ★Disk-safe 圧縮キャッシュ版★
──────────────────────────────────────────────────────────
投稿ベクトル集合 (≤30 × 3072d) → アカウント RW ベクトルを高精度推定
【Cross-Entropy (10k分類) ＋ InfoNCE】マルチタスク学習。

● 投稿キャッシュは gzip 圧縮 pickle (≈4× 圧縮)＋float16 で最大 600 MB
● --no-cache でディスクを一切使わず CSV を毎回ストリーム学習
● --resume でチェックポイント継続学習
"""

import os, csv, argparse, random, pickle, gzip
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#──────────────────────── Paths & Hyperparams ────────────────────────
BASE_DIR  = "/workspace/edit_agent"
VAST_DIR  = os.path.join(BASE_DIR, "vast")
POSTS_CSV = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCS_NPY  = os.path.join(VAST_DIR, "account_vectors.npy")

CKPT_PATH = os.path.join(VAST_DIR, "multitask_attn.ckpt")
CACHE_GZ  = os.path.join(VAST_DIR, "posts_cache.pkl.gz")

POST_DIM      = 3072
POSTS_PER_UID = 30
D_MODEL       = 256
NUM_SEEDS     = 4
N_HEADS       = 8
N_LAYERS      = 2
DROPOUT       = 0.1

BATCH_SIZE    = 128
EPOCHS        = 100
LR            = 2e-4
WEIGHT_DECAY  = 1e-5
VAL_RATIO     = 0.1
PATIENCE      = 10
MIN_DELTA     = 1e-4
TEMP          = 0.1          # InfoNCE 温度
ALPHA         = 0.6          # CE / NCE 重み

#──────────────────────── Utils ────────────────────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l,r = s.find('['), s.rfind(']')
    if 0<=l<r: s = s[l+1:r]
    arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

#──────────────────────── Dataset ────────────────────────
class PostSetDataset(Dataset):
    def __init__(self, uids, uid2posts):
        self.samples = [(uid2posts[u], u) for u in uids]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        posts, uid = self.samples[idx]
        return posts, uid

def collate(batch):
    posts_list, uid_list = zip(*batch)
    B = len(batch)
    Ls = [p.shape[0] for p in posts_list]
    S  = max(Ls)
    x  = torch.zeros(B, S, POST_DIM, dtype=torch.float32)
    mask = torch.ones(B, S, dtype=torch.bool)
    for i, p in enumerate(posts_list):
        L = p.shape[0]
        x[i,:L] = torch.from_numpy(p.astype(np.float32))
        mask[i,:L] = False
    return x, mask, list(uid_list)

#──────────────────────── Model ────────────────────────
class MultiSeedPool(nn.Module):
    def __init__(self, d_model, num_seeds, n_heads):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, d_model))
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=DROPOUT,
                                           batch_first=True)
    def forward(self, x, mask):
        B = x.size(0)
        q = self.seeds.unsqueeze(0).expand(B, -1, -1)
        out,_ = self.attn(q, x, x, key_padding_mask=mask)
        return out.mean(dim=1)

class AttnAgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(POST_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, D_MODEL)
        )
        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, D_MODEL*4,
            dropout=DROPOUT, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, N_LAYERS)
        self.pool = MultiSeedPool(D_MODEL, NUM_SEEDS, N_HEADS)

    def forward(self, x, mask):
        h = self.in_proj(x)
        h = self.encoder(h, src_key_padding_mask=mask)
        return self.pool(h, mask)      # (B,D_MODEL)

class MultiTask(nn.Module):
    def __init__(self, num_accounts, rw_dim):
        super().__init__()
        self.agg = AttnAgg()
        self.cls = nn.Linear(D_MODEL, num_accounts)
        self.map = nn.Linear(D_MODEL, rw_dim)

    def forward(self, x, mask):
        z = self.agg(x, mask)
        logits = self.cls(z)
        rw = F.normalize(self.map(z), dim=1)
        return logits, rw

#──────────────────────── Main ────────────────────────
def load_or_build_posts(no_cache: bool, force_recache: bool):
    if (not no_cache) and (not force_recache) and os.path.exists(CACHE_GZ):
        with gzip.open(CACHE_GZ,'rb') as f:
            data = pickle.load(f)
        print(f"[Cache] loaded {len(data)} UID posts from compressed cache")
        return data

    uid_posts = defaultdict(list)
    total = sum(1 for _ in open(POSTS_CSV,encoding='utf-8')) - 1
    with open(POSTS_CSV,encoding='utf-8') as f:
        rdr=csv.reader(f); next(rdr)
        for uid,_,vec in tqdm(rdr,total=total,desc='CSV'):
            v=parse_vec(vec,POST_DIM)
            if v is None: continue
            uid_posts[uid].append(v.astype(np.float16))
            if len(uid_posts[uid])>POSTS_PER_UID:
                uid_posts[uid].pop(0)
    print(f"[Cache] built posts for {len(uid_posts)} UIDs")

    if not no_cache:
        with gzip.open(CACHE_GZ,'wb',compresslevel=5) as f:
            pickle.dump(uid_posts,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Cache] compressed cache saved to {CACHE_GZ}")

    return uid_posts

def train(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[Device]', dev)

    # account vectors
    acc = np.load(ACCS_NPY,allow_pickle=True).item()
    uids_all = list(acc.keys()); uid2idx={u:i for i,u in enumerate(uids_all)}
    rw_dim = next(iter(acc.values())).shape[0]

    # posts
    uid_posts = load_or_build_posts(args.no_cache, args.force_recache)

    # split
    all_uids = list(uid_posts.keys()); random.shuffle(all_uids)
    n_val=int(len(all_uids)*VAL_RATIO)
    val_uids, tr_uids = all_uids[:n_val], all_uids[n_val:]
    print(f'[Split] train={len(tr_uids)} val={len(val_uids)}')

    tr_ld = DataLoader(PostSetDataset(tr_uids,uid_posts),
                       batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)
    va_ld = DataLoader(PostSetDataset(val_uids,uid_posts),
                       batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)

    model = MultiTask(len(uids_all), rw_dim).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, patience, start_ep = float('inf'), 0, 1
    if args.resume and os.path.exists(CKPT_PATH):
        ck=torch.load(CKPT_PATH,map_location=dev)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        best_val,patience,start_ep=ck['best_val'],ck['patience'],ck['epoch']+1
        print(f'[Resume] epoch={start_ep-1} best={best_val:.4f}')

    for ep in range(start_ep,EPOCHS+1):
        print(f'\n=== Epoch {ep}/{EPOCHS} ===')
        # Train
        model.train(); s=n=0
        for x,mask,uids in tqdm(tr_ld,desc='Train'):
            x,mask = x.to(dev),mask.to(dev)
            logits,rw_hat = model(x,mask)
            tgt_idx = torch.tensor([uid2idx[u] for u in uids], device=dev)
            loss_ce = F.cross_entropy(logits,tgt_idx)
            rw_true = F.normalize(torch.tensor(
                     np.stack([acc[u] for u in uids]),
                     dtype=torch.float32,device=dev),dim=1)
            sim = rw_hat@rw_true.T / TEMP
            labels=torch.arange(sim.size(0),device=dev)
            loss_nce = F.cross_entropy(sim,labels)
            loss = ALPHA*loss_ce + (1-ALPHA)*loss_nce
            opt.zero_grad(); loss.backward(); opt.step()
            s+=loss.item()*len(uids); n+=len(uids)
        tr_loss=s/n

        # Val
        model.eval(); s=n=0
        with torch.no_grad():
            for x,mask,uids in tqdm(va_ld,desc='Val'):
                x,mask=x.to(dev),mask.to(dev)
                logits,rw_hat=model(x,mask)
                tgt_idx=torch.tensor([uid2idx[u] for u in uids],device=dev)
                loss_ce=F.cross_entropy(logits,tgt_idx)
                rw_true=F.normalize(torch.tensor(
                         np.stack([acc[u] for u in uids]),
                         dtype=torch.float32,device=dev),dim=1)
                sim=rw_hat@rw_true.T / TEMP
                labels=torch.arange(sim.size(0),device=dev)
                loss_nce=F.cross_entropy(sim,labels)
                loss=ALPHA*loss_ce+(1-ALPHA)*loss_nce
                s+=loss.item()*len(uids); n+=len(uids)
        va_loss=s/n
        print(f'Ep{ep:03d} train={tr_loss:.4f} val={va_loss:.4f}')

        if va_loss<best_val-MIN_DELTA:
            best_val,patience=va_loss,0
            torch.save({'epoch':ep,'model':model.state_dict(),
                        'opt':opt.state_dict(),'best_val':best_val,
                        'patience':patience},CKPT_PATH)
            print(f'  ✔ saved {CKPT_PATH}')
        else:
            patience+=1
            if patience>=PATIENCE: print('Early stopping'); break
    print(f'[Done] best_val={best_val:.4f}')

#──────────────────────── CLI ────────────────────────
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--resume',action='store_true',help='resume training')
    ap.add_argument('--no-cache',action='store_true',
                    help='do not save/read gzip cache')
    ap.add_argument('--force-recache',action='store_true',
                    help='rebuild cache even if file exists')
    args=ap.parse_args()
    train(args)