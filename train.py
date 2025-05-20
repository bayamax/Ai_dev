#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― SetTransformer フォロー予測モデル学習
  ・前処理済みデータ: /workspace/edit_agent/vast/follow_dataset.pt
  ・モデル保存先    : /workspace/edit_agent/vast/set_transformer_follow_predictor.pt
  ・–1 を ignore して、0/1 のみ BCE 計算するよう修正
"""

import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from model import SetToVectorPredictor  # 先ほどお渡しした model.py を同ディレクトリに置いてください

# ──────────────────────────────────────────────────────────────
# パス定数
# ──────────────────────────────────────────────────────────────
VAST_DIR                    = "/workspace/edit_agent/vast"
DEFAULT_PROCESSED_DATA_PATH = os.path.join(VAST_DIR, "follow_dataset.pt")
DEFAULT_MODEL_SAVE_PATH     = os.path.join(VAST_DIR, "set_transformer_follow_predictor.pt")

# ──────────────────────────────────────────────────────────────
# モデル／学習パラメータ（CLI で上書き可）
# ──────────────────────────────────────────────────────────────
DEFAULT_POST_EMBEDDING_DIM     = 3072
DEFAULT_ENCODER_OUTPUT_DIM     = 512
DEFAULT_NUM_ATTENTION_HEADS    = 4
DEFAULT_NUM_ENCODER_LAYERS     = 2
DEFAULT_DROPOUT_RATE           = 0.1

DEFAULT_LEARNING_RATE          = 1e-5
DEFAULT_BATCH_SIZE             = 64
DEFAULT_NUM_EPOCHS             = 50
DEFAULT_WEIGHT_DECAY           = 1e-5
DEFAULT_VALIDATION_SPLIT       = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4

# ──────────────────────────────────────────────────────────────
# データセットクラス
# ──────────────────────────────────────────────────────────────
class FollowPredictionDataset(Dataset):
    def __init__(self, processed_data_path: str):
        data = torch.load(processed_data_path)
        self.dataset = data["dataset"]              # List of (posts_tensor, target_vector, uid)
        self.all_accounts = data["all_account_list"]
        self.account_to_idx = data["account_to_idx"]
        self.num_all_accounts = len(self.all_accounts)
        print(f"[Dataset] {len(self.dataset)} samples, {self.num_all_accounts} accounts")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        posts, target, uid = self.dataset[idx]
        return posts, target

# ──────────────────────────────────────────────────────────────
# collate_fn：可変長の投稿リストをパディング
# ──────────────────────────────────────────────────────────────
def collate_set_transformer(batch):
    posts_list, targets = zip(*batch)
    # 各シーケンス長を取得
    lengths = torch.tensor([p.size(0) for p in posts_list])
    max_len = lengths.max().item()
    # padding
    padded_posts = torch.nn.utils.rnn.pad_sequence(
        posts_list, batch_first=True, padding_value=0.0
    )  # (B, S, D)
    # MHA 用マスク：True が「パディング」
    padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets)  # (B, num_accounts)
    return padded_posts, padding_mask, targets

# ──────────────────────────────────────────────────────────────
# 学習ルーチン
# ──────────────────────────────────────────────────────────────
def train_model(args) -> bool:
    # デバイス選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_set_transformer,
    )

    # モデル／オプティマイザ／損失関数(reduction='none')
    model = SetToVectorPredictor(
        post_embedding_dim=args.post_embedding_dim,
        encoder_output_dim=args.encoder_output_dim,
        num_all_accounts=ds.num_all_accounts,
        num_attention_heads=args.num_attention_heads,
        num_encoder_layers=args.num_encoder_layers,
        dropout_rate=args.dropout_rate,
    ).to(device)

    # BCEWithLogitsLoss を要素ごとに計算してからマスク平均を取る
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        # — train —
        model.train()
        train_num = 0.0
        train_den = 0.0
        for posts, mask, targets in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            posts, mask, targets = (
                posts.to(device),
                mask.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            logits, _ = model(posts, mask)

            # targets の –1 は ignore、0/1 のみを clamp
            y_clamped = torch.clamp(targets, 0.0, 1.0)
            loss_mat  = criterion(logits, y_clamped)           # (B, N)
            valid_mask= (targets != -1.0).float()              # (B, N)

            # 合計／有効要素数 で平均
            numer = (loss_mat * valid_mask).sum()
            denom = valid_mask.sum().clamp(min=1.0)
            loss  = numer / denom

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_num += numer.item()
            train_den += denom.item()

        train_loss = train_num / train_den

        # — validation —
        model.eval()
        val_num = 0.0
        val_den = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                posts, mask, targets = (
                    posts.to(device),
                    mask.to(device),
                    targets.to(device),
                )
                logits, _ = model(posts, mask)

                y_clamped = torch.clamp(targets, 0.0, 1.0)
                loss_mat  = criterion(logits, y_clamped)
                valid_mask= (targets != -1.0).float()

                val_num += (loss_mat * valid_mask).sum().item()
                val_den += valid_mask.sum().item()

        val_loss = val_num / val_den

        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}"
        )

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
    p = argparse.ArgumentParser(description="Train SetTransformer follow model")
    p.add_argument("--data_path",               default=DEFAULT_PROCESSED_DATA_PATH)
    p.add_argument("--model_save_path",         default=DEFAULT_MODEL_SAVE_PATH)
    p.add_argument("--epochs",       type=int,   default=DEFAULT_NUM_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--validation_split",         type=float, default=DEFAULT_VALIDATION_SPLIT)
    p.add_argument("--early_stopping_patience",  type=int,   default=DEFAULT_EARLY_STOPPING_PATIENCE)
    p.add_argument("--early_stopping_min_delta", type=float, default=DEFAULT_EARLY_STOPPING_MIN_DELTA)

    p.add_argument("--post_embedding_dim", type=int,   default=DEFAULT_POST_EMBEDDING_DIM)
    p.add_argument("--encoder_output_dim", type=int,   default=DEFAULT_ENCODER_OUTPUT_DIM)
    p.add_argument("--num_attention_heads",type=int,   default=DEFAULT_NUM_ATTENTION_HEADS)
    p.add_argument("--num_encoder_layers", type=int,   default=DEFAULT_NUM_ENCODER_LAYERS)
    p.add_argument("--dropout_rate",        type=float, default=DEFAULT_DROPOUT_RATE)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not train_model(args):
        sys.exit(1)