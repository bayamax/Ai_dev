#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_masked_follow_predictor.py ― 投稿セットからマスクしたフォローラベルを予測する学習スクリプト
  • Multi‐Label Transformer を用いて、一部ラベルをマスクして復元
  • mask_prob でマスクするラベル割合を指定 (デフォルト 0.15)
  • -1 のラベルは常に ignore（損失計算に含めない）
  • BCEWithLogitsLoss(reduction="none") を使い、マスクした位置のみ平均
  • 早期停止・モデル保存機能つき
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
# デフォルト設定
# ──────────────────────────────────────────────────────────────
VAST_DIR                     = "/workspace/edit_agent/vast"
DEFAULT_DATA_PATH            = os.path.join(VAST_DIR, "follow_dataset.pt")
DEFAULT_MODEL_SAVE_PATH      = os.path.join(VAST_DIR, "set_transformer_follow_predictor.pt")

DEFAULT_POST_EMBEDDING_DIM   = 3072
DEFAULT_ENCODER_OUTPUT_DIM   = 512
DEFAULT_NUM_ATTENTION_HEADS  = 4
DEFAULT_NUM_ENCODER_LAYERS   = 2
DEFAULT_DROPOUT_RATE         = 0.1

DEFAULT_LEARNING_RATE        = 1e-5
DEFAULT_BATCH_SIZE           = 64
DEFAULT_NUM_EPOCHS           = 50
DEFAULT_WEIGHT_DECAY         = 1e-5
DEFAULT_VALIDATION_SPLIT     = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE   = 5
DEFAULT_EARLY_STOPPING_MIN_DELTA   = 1e-4

DEFAULT_MASK_PROB           = 0.15  # ラベルをマスクする割合

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
    posts_list, targets_list = zip(*batch)
    # 長さ情報
    lengths = torch.tensor([p.size(0) for p in posts_list])
    max_len = int(lengths.max().item())
    # パディング
    padded_posts = torch.nn.utils.rnn.pad_sequence(
        posts_list, batch_first=True, padding_value=0.0
    )  # (B, S, D)
    # Transformer の key_padding_mask: True が「無視」
    padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets_list)  # (B, N_accounts)
    return padded_posts, padding_mask, targets

# ──────────────────────────────────────────────────────────────
# 学習ルーチン
# ──────────────────────────────────────────────────────────────
def train_model(args) -> bool:
    # デバイス
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

    # モデル初期化
    model = SetToVectorPredictor(
        post_embedding_dim=args.post_embedding_dim,
        encoder_output_dim=args.encoder_output_dim,
        num_all_accounts=ds.num_all_accounts,
        num_attention_heads=args.num_attention_heads,
        num_encoder_layers=args.num_encoder_layers,
        dropout_rate=args.dropout_rate,
    ).to(device)

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
        num_loss = 0.0
        num_count = 0.0
        for posts, mask, targets in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)

            optimizer.zero_grad()

            # マスクの作成：-valid (targets != -1) かつ rand < mask_prob
            valid = (targets != -1)
            rand_mask = torch.rand_like(targets, dtype=torch.float32) < args.mask_prob
            mask_positions = valid & rand_mask

            # マスクしない位置は ignore (-1)
            raw_targets = targets.clone()
            targets_masked = targets.clone()
            targets_masked[~mask_positions] = -1

            # シグマイド用に 0/1 にクランプ
            raw_for_loss = raw_targets.clamp(min=0.0, max=1.0)

            # 順伝搬
            logits, _ = model(posts, mask)

            # 損失マトリクス計算
            loss_mat = criterion(logits, raw_for_loss)  # (B, N)
            # マスク位置のみ足し合わせ
            loss = (loss_mat * mask_positions.float()).sum() / mask_positions.sum().clamp(min=1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            num_loss += loss.item() * mask_positions.sum().item()
            num_count += mask_positions.sum().item()

        train_loss = num_loss / num_count

        # — val —
        model.eval()
        num_loss = 0.0
        num_count = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)

                valid = (targets != -1)
                rand_mask = torch.rand_like(targets, dtype=torch.float32) < args.mask_prob
                mask_positions = valid & rand_mask

                raw_targets = targets.clone()
                raw_for_loss = raw_targets.clamp(min=0.0, max=1.0)

                logits, _ = model(posts, mask)
                loss_mat = criterion(logits, raw_for_loss)

                num_loss += (loss_mat * mask_positions.float()).sum().item()
                num_count += mask_positions.sum().item()

        val_loss = num_loss / num_count

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
                print("Early stopping.")
                break

    print(f"[Done] best_val_loss = {best_val:.4f}")
    return True

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train masked follow predictor")
    p.add_argument("--data_path",               default=DEFAULT_DATA_PATH)
    p.add_argument("--model_save_path",         default=DEFAULT_MODEL_SAVE_PATH)
    p.add_argument("--epochs",      type=int,   default=DEFAULT_NUM_EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--weight_decay",type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--validation_split",       type=float, default=DEFAULT_VALIDATION_SPLIT)
    p.add_argument("--early_stopping_patience",type=int,   default=DEFAULT_EARLY_STOPPING_PATIENCE)
    p.add_argument("--early_stopping_min_delta",type=float,default=DEFAULT_EARLY_STOPPING_MIN_DELTA)
    p.add_argument("--post_embedding_dim",     type=int,   default=DEFAULT_POST_EMBEDDING_DIM)
    p.add_argument("--encoder_output_dim",     type=int,   default=DEFAULT_ENCODER_OUTPUT_DIM)
    p.add_argument("--num_attention_heads",    type=int,   default=DEFAULT_NUM_ATTENTION_HEADS)
    p.add_argument("--num_encoder_layers",     type=int,   default=DEFAULT_NUM_ENCODER_LAYERS)
    p.add_argument("--dropout_rate",           type=float, default=DEFAULT_DROPOUT_RATE)
    p.add_argument("--mask_prob",              type=float, default=DEFAULT_MASK_PROB,
                   help="マスクするラベル割合 (0.0–1.0)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not train_model(args):
        sys.exit(1)