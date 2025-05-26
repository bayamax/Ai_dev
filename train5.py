#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_multitask_attn.py
──────────────────────────────────────────────────────────
投稿ベクトル集合 (≤30×3072d) からアカウント RW ベクトルを推定し、
【分類＋コントラスト学習】のマルチタスクでベースラインを凌駕する高精度モデルを学習する。

  • CompactAttnAggregator: 3072d → d_model=256 に射影し 2-layer Transformer
  • 4-seed Pooling で多視点アテンション集約
  • Loss = α·CrossEntropy + (1-α)·InfoNCE（in-batch hard negatives）
  • 10,000クラス規模の softmax だが GPU 1枚で収まるパラメータ量
  • CSV ストリーミング → posts_cache.pkl に保存（RAM を一定に保つ）
  • --resume でチェックポイント継続学習
──────────────────────────────────────────────────────────
"""

import os, csv, argparse, random, pickle, math
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
CACHE_PKL = os.path.join(VAST_DIR, "posts_cache.pkl")

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
ALPHA         = 0.6          # CE と InfoNCE の重み

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
        x[i,:L] = torch.from_numpy(p)
        mask[i,:L] = False
    return x, mask, list(uid_list)

#──────────────────────── Model ────────────────────────
class MultiSeedPool(nn.Module):
    """Pooling by Multi-Seed Attention (Set Transformer の PMA)"""
    def __init__(self, d_model, num_seeds, n_heads):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, d_model))
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=DROPOUT,
                                           batch_first=True)
    def forward(self, x, pad_mask):
        B = x.size(0)
        q = self.seeds.unsqueeze(0).repeat(B,1,1)     # (B, num_seeds, d_model)
        out, _ = self.attn(q, x, x, key_padding_mask=pad_mask)
        return out.mean(dim=1)                         # (B, d_model)

class AttnAggregator(nn.Module):
    def __init__(self, d_in=POST_DIM, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS,
                 num_seeds=NUM_SEEDS, dropout=DROPOUT):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.pool = MultiSeedPool(d_model, num_seeds, n_heads)

    def forward(self, x, pad_mask):
        h = self.in_proj(x)            # (B,S,d_model)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        pooled = self.pool(h, pad_mask)
        return pooled                  # (B,d_model)

class MultiTaskModel(nn.Module):
    def __init__(self, num_accounts, rw_dim):
        super().__init__()
        self.agg  = AttnAggregator()
        self.cls_head = nn.Linear(D_MODEL, num_accounts)
        self.map_head = nn.Linear(D_MODEL, rw_dim)

    def forward(self, x, mask):
        z = self.agg(x, mask)                # (B,D_MODEL)
        logits = self.cls_head(z)            # (B, num_accounts)
        rw_pred = F.normalize(self.map_head(z), dim=1)  # (B,rw_dim), ℓ2 正規化
        return logits, rw_pred

#──────────────────────── Main ────────────────────────
def train(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[Device]', dev)

    # 1) アカウントベクトル
    acc_dict = np.load(ACCS_NPY, allow_pickle=True).item()
    uids_all = list(acc_dict.keys())
    rw_dim   = acc_dict[uids_all[0]].shape[0]
    uid2idx  = {u:i for i,u in enumerate(uids_all)}

    # 2) 投稿キャッシュ構築 / 読込
    if os.path.exists(CACHE_PKL) and not args.force_recache:
        uid_posts = pickle.load(open(CACHE_PKL,'rb'))
        print(f'[Cache] loaded {len(uid_posts)} UID posts')
    else:
        uid_posts = defaultdict(list)
        total_lines = sum(1 for _ in open(POSTS_CSV,encoding='utf-8'))-1
        with open(POSTS_CSV,encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid,_,vec_s in tqdm(rdr,total=total_lines,desc='CSV'):
                if uid not in acc_dict: continue
                v = parse_vec(vec_s, POST_DIM)
                if v is None: continue
                lst = uid_posts[uid]; lst.append(v)
                if len(lst)>POSTS_PER_UID: lst.pop(0)
        pickle.dump(uid_posts,open(CACHE_PKL,'wb'))
        print(f'[Cache] saved {len(uid_posts)} UID posts')

    # 3) Split
    all_uids = list(uid_posts.keys())
    random.shuffle(all_uids)
    n_val = int(len(all_uids)*VAL_RATIO)
    val_uids = all_uids[:n_val]; tr_uids = all_uids[n_val:]
    print(f'[Split] train={len(tr_uids)}  val={len(val_uids)}')

    tr_ds = PostSetDataset(tr_uids, uid_posts)
    va_ds = PostSetDataset(val_uids, uid_posts)
    tr_ld = DataLoader(tr_ds,batch_size=BATCH_SIZE,shuffle=True,
                       collate_fn=collate)
    va_ld = DataLoader(va_ds,batch_size=BATCH_SIZE,shuffle=False,
                       collate_fn=collate)

    # 4) Model / Optimizer
    model = MultiTaskModel(num_accounts=len(uids_all), rw_dim=rw_dim).to(dev)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, patience, start_ep = float('inf'), 0, 1
    if args.resume and os.path.exists(CKPT_PATH):
        ck = torch.load(CKPT_PATH,map_location=dev)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        best_val, patience, start_ep = ck['best_val'], ck['patience'], ck['epoch']+1
        print(f'[Resume] epoch={start_ep-1} best_val={best_val:.4f}')

    # 5) Training
    for ep in range(start_ep, EPOCHS+1):
        print(f'\n=== Epoch {ep}/{EPOCHS} ===')
        # ---- Train ----
        model.train(); sum_tr=n_tr=0
        for x,mask,uids in tqdm(tr_ld,desc='Train'):
            x,mask=x.to(dev),mask.to(dev)
            logits, rw_hat = model(x,mask)
            # Cross-entropy targets
            tgt_idx = torch.tensor([uid2idx[u] for u in uids],
                                   dtype=torch.long, device=dev)
            loss_ce = F.cross_entropy(logits, tgt_idx)

            # InfoNCE (in-batch)
            rw_true = F.normalize(torch.tensor(
                       np.stack([acc_dict[u] for u in uids]),
                       dtype=torch.float32,device=dev), dim=1)
            sim = rw_hat @ rw_true.T / TEMP                # (B,B)
            labels = torch.arange(sim.size(0),device=dev)
            loss_nce = F.cross_entropy(sim, labels)

            loss = ALPHA*loss_ce + (1-ALPHA)*loss_nce
            opt.zero_grad(); loss.backward(); opt.step()
            sum_tr += loss.item()*len(uids); n_tr+=len(uids)
        tr_loss = sum_tr/n_tr

        # ---- Val ----
        model.eval(); sum_va=n_va=0
        with torch.no_grad():
            for x,mask,uids in tqdm(va_ld,desc='Val'):
                x,mask=x.to(dev),mask.to(dev)
                logits,rw_hat = model(x,mask)
                tgt_idx = torch.tensor([uid2idx[u] for u in uids],
                                       dtype=torch.long,device=dev)
                loss_ce  = F.cross_entropy(logits,tgt_idx)
                rw_true  = F.normalize(torch.tensor(
                            np.stack([acc_dict[u] for u in uids]),
                            dtype=torch.float32,device=dev),dim=1)
                sim = rw_hat@rw_true.T / TEMP
                labels = torch.arange(sim.size(0),device=dev)
                loss_nce = F.cross_entropy(sim,labels)
                loss = ALPHA*loss_ce + (1-ALPHA)*loss_nce
                sum_va += loss.item()*len(uids); n_va+=len(uids)
        va_loss=sum_va/n_va
        print(f'Ep{ep:03d}  train={tr_loss:.4f}  val={va_loss:.4f}')

        # ---- Checkpoint ----
        if va_loss < best_val - MIN_DELTA:
            best_val, patience = va_loss, 0
            torch.save({
                'epoch':ep,'model':model.state_dict(),
                'opt':opt.state_dict(),
                'best_val':best_val,'patience':patience},
                CKPT_PATH)
            print(f'  ✔ saved {CKPT_PATH}')
        else:
            patience += 1
            if patience >= PATIENCE:
                print('Early stopping'); break

    print(f'[Done] best_val={best_val:.4f}')

#──────────────────────── CLI ────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume',action='store_true',help='resume from ckpt')
    ap.add_argument('--force-recache',action='store_true',
                    help='rebuild posts cache')
    args = ap.parse_args()
    train(args)