#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― SetTransformer フォロー予測 + HardNeg ＋ RankingLoss（BPR）
  • BCEWithLogitsLoss + λ * BPR Loss
  • padding-mask による ignore 列
  • 既存モデルの resume ／ early stopping
  • temperature weighting はなし（必要なら復帰してください）
"""

import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from model import SetToVectorPredictor  # 学習時と同じ model.py を使う

# ──────────────────────────────────────────────────────────────
# パス定数
# ──────────────────────────────────────────────────────────────
VAST_DIR                    = "/workspace/edit_agent/vast"
DEFAULT_DATA_PATH           = os.path.join(VAST_DIR, "follow_dataset.pt")
DEFAULT_MODEL_SAVE_PATH     = os.path.join(VAST_DIR, "set_transformer_follow_predictor.pt")

# ──────────────────────────────────────────────────────────────
# モデル／学習パラメータ（CLI で上書き可）
# ──────────────────────────────────────────────────────────────
DEFAULT_POST_EMBEDDING_DIM      = 3072
DEFAULT_ENCODER_OUTPUT_DIM      = 512
DEFAULT_NUM_ATTENTION_HEADS     = 4
DEFAULT_NUM_ENCODER_LAYERS      = 2
DEFAULT_DROPOUT_RATE            = 0.1

DEFAULT_LEARNING_RATE           = 1e-5
DEFAULT_BATCH_SIZE              = 64
DEFAULT_NUM_EPOCHS              = 500
DEFAULT_WEIGHT_DECAY            = 1e-5
DEFAULT_VALIDATION_SPLIT        = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_EARLY_STOPPING_MIN_DELTA= 1e-4

# ─── Ranking Loss ハイパラ ─────────────────
DEFAULT_LAMBDA_RANK = 1.0   # BCELoss に対する RankingLoss の重み

# ──────────────────────────────────────────────────────────────
# データセットクラス
# ──────────────────────────────────────────────────────────────
class FollowPredictionDataset(Dataset):
    def __init__(self, processed_data_path: str):
        data = torch.load(processed_data_path)
        self.dataset       = data["dataset"]          # List of (posts_tensor, target_vector, uid)
        self.all_accounts  = data["all_account_list"]
        self.account_to_idx= data["account_to_idx"]
        self.num_all_accounts = len(self.all_accounts)
        print(f"[Dataset] {len(self.dataset)} samples, {self.num_all_accounts} accounts")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        posts, target, uid = self.dataset[idx]
        # posts: (S, D), target: (N,), uid: str
        return posts, target

# ──────────────────────────────────────────────────────────────
# collate_fn：可変長の投稿リストをパディング
# ──────────────────────────────────────────────────────────────
def collate_set_transformer(batch):
    posts_list, targets = zip(*batch)
    # パディング
    lengths = torch.tensor([p.size(0) for p in posts_list])
    max_len = lengths.max().item()
    padded_posts = torch.nn.utils.rnn.pad_sequence(
        posts_list, batch_first=True, padding_value=0.0
    )  # (B, S, D)
    # MHA 用マスク：True が「パディング」
    padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets)  # (B, num_accounts)
    return padded_posts, padding_mask, targets

# ──────────────────────────────────────────────────────────────
# BPR Ranking Loss
# ──────────────────────────────────────────────────────────────
def bpr_loss(logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, N)
    targets: {0,1,-1}, shape (B, N)
    padding_mask: (B, S) but unused here
    For each sample in batch, randomly pick one positive and one negative,
    then compute -log sigmoid( score_pos - score_neg ), average over batch.
    """
    B, N = logits.shape
    losses = []
    for i in range(B):
        pos_idx = (targets[i] == 1).nonzero(as_tuple=False).view(-1)
        neg_idx = (targets[i] == 0).nonzero(as_tuple=False).view(-1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue  # このサンプルは RankingLoss 対象外
        # ランダムに 1 つずつサンプリング
        p = pos_idx[torch.randint(len(pos_idx), (1,))].item()
        n = neg_idx[torch.randint(len(neg_idx), (1,))].item()
        score_p = logits[i, p]
        score_n = logits[i, n]
        losses.append(-torch.log(torch.sigmoid(score_p - score_n) + 1e-8))
    if len(losses) == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()

# ──────────────────────────────────────────────────────────────
# 学習ルーチン
# ──────────────────────────────────────────────────────────────
def train_model(args) -> bool:
    # デバイス選択
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("[Device]", device)

    # データロード
    ds = FollowPredictionDataset(args.data_path)
    if len(ds) == 0:
        print("Dataset is empty."); return False

    # train/val split
    n_total = len(ds)
    n_val   = int(args.validation_split * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    print(f"[Split] train={n_train}, val={n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_set_transformer,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_set_transformer,
        num_workers=0
    )

    # モデル／オプティマイザ
    model = SetToVectorPredictor(
        post_embedding_dim=args.post_embedding_dim,
        encoder_output_dim=args.encoder_output_dim,
        num_all_accounts=ds.num_all_accounts,
        num_attention_heads=args.num_attention_heads,
        num_encoder_layers=args.num_encoder_layers,
        dropout_rate=args.dropout_rate,
    ).to(device)

    if args.resume and os.path.isfile(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
        print(f"[Resume] loaded {args.model_save_path}")

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    patience = 0

    λ = args.lambda_rank

    for epoch in range(1, args.epochs + 1):
        # — train —
        model.train()
        sum_loss = 0.0
        sum_samples = 0
        for posts, mask, targets in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            posts, mask, targets = (
                posts.to(device),
                mask.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()

            logits, _ = model(posts, mask)  # (B, N)
            # BCE 部分
            bce_mat = bce_criterion(logits, torch.clamp(targets, 0, 1))  # (B, N)
            # ignore マスク (-1)
            valid_cols = (targets != -1).float()
            bce_loss = (bce_mat * valid_cols).sum() / valid_cols.sum().clamp(min=1.0)

            # Ranking Loss 部分
            rank_loss = bpr_loss(logits, targets, mask)

            loss = bce_loss + λ * rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sum_loss += loss.item() * posts.size(0)
            sum_samples += posts.size(0)

        train_loss = sum_loss / sum_samples

        # — validation —
        model.eval()
        sum_val = 0.0
        sum_val_samples = 0
        with torch.no_grad():
            for posts, mask, targets in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                posts, mask, targets = (
                    posts.to(device),
                    mask.to(device),
                    targets.to(device),
                )
                logits, _ = model(posts, mask)

                bce_mat = bce_criterion(logits, torch.clamp(targets, 0, 1))
                valid_cols = (targets != -1).float()
                bce_loss = (bce_mat * valid_cols).sum() / valid_cols.sum().clamp(min=1.0)
                rank_loss = bpr_loss(logits, targets, mask)
                loss = bce_loss + λ * rank_loss

                sum_val += loss.item() * posts.size(0)
                sum_val_samples += posts.size(0)

        val_loss = sum_val / sum_val_samples

        print(f"Epoch {epoch}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        # early stopping & save
        if val_loss < best_val - args.early_stopping_min_delta:
            best_val = val_loss
            patience = 0
            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), args.model_save_path)
            print("  ✔ saved best →", args.model_save_path)
        else:
            patience += 1
            print(f"  (no improvement {patience}/{args.early_stopping_patience})")
            if patience >= args.early_stopping_patience:
                print("Early stopping."); break

    print(f"[Done] best_val_loss = {best_val:.4f}")
    return True

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train SetTransformer follow model with BPR ranking loss")
    p.add_argument("--data_path",             default=DEFAULT_DATA_PATH)
    p.add_argument("--model_save_path",       default=DEFAULT_MODEL_SAVE_PATH)
    p.add_argument("--resume",                action="store_true")
    p.add_argument("--epochs",      type=int, default=DEFAULT_NUM_EPOCHS)
    p.add_argument("--batch_size",  type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--weight_decay",type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--validation_split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    p.add_argument("--early_stopping_patience", type=int,   default=DEFAULT_EARLY_STOPPING_PATIENCE)
    p.add_argument("--early_stopping_min_delta", type=float, default=DEFAULT_EARLY_STOPPING_MIN_DELTA)

    p.add_argument("--post_embedding_dim", type=int,   default=DEFAULT_POST_EMBEDDING_DIM)
    p.add_argument("--encoder_output_dim", type=int,   default=DEFAULT_ENCODER_OUTPUT_DIM)
    p.add_argument("--num_attention_heads", type=int,  default=DEFAULT_NUM_ATTENTION_HEADS)
    p.add_argument("--num_encoder_layers",  type=int,  default=DEFAULT_NUM_ENCODER_LAYERS)
    p.add_argument("--dropout_rate",        type=float,default=DEFAULT_DROPOUT_RATE)

    p.add_argument("--lambda_rank", type=float, default=DEFAULT_LAMBDA_RANK,
                   help="weight for BPR ranking loss")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not train_model(args):
        sys.exit(1)