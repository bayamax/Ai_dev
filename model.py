# model.py
import torch
import torch.nn as nn

class SetToVectorPredictor(nn.Module):
    """
    入力: バッチ x シーケンス長 x post_embedding_dim の投稿埋め込みセット
    マスク: バッチ x シーケンス長 の padding_mask (True=パディング位置)
    出力: 
      - logits: バッチ x num_all_accounts のフォロー確率スコア
      - pooled: バッチ x encoder_output_dim の集合全体表現
    """
    def __init__(
        self,
        post_embedding_dim: int,
        encoder_output_dim: int,
        num_all_accounts: int,
        num_attention_heads: int,
        num_encoder_layers: int,
        dropout_rate: float
    ):
        super().__init__()

        # 1) 投稿埋め込みを encoder_output_dim に射影
        self.initial_projection = nn.Linear(post_embedding_dim, encoder_output_dim)

        # 2) TransformerEncoderLayer をスタック
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_output_dim,
            nhead=num_attention_heads,
            dim_feedforward=encoder_output_dim * 4,
            dropout=dropout_rate,
            batch_first=False,  # 以下で permute するので False にしておく
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 3) プーリング後の分類器
        hidden_dim = encoder_output_dim // 2
        self.decoder = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_all_accounts)
        )

    def forward(self, posts: torch.Tensor, padding_mask: torch.Tensor):
        """
        posts: Tensor of shape (B, S, D_in)
        padding_mask: BoolTensor of shape (B, S), True = padding
        """
        # (B, S, D_out)
        x = self.initial_projection(posts)

        # Transformer に入れるには (S, B, D_out)
        x = x.permute(1, 0, 2)

        # key_padding_mask: (B, S), True で ignore
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )  # -> (S, B, D_out)

        # 戻して (B, S, D_out)
        x = x.permute(1, 0, 2)

        # マスク付き平均プーリング
        # padding_mask: True=pad, False=実データ
        valid = (~padding_mask).unsqueeze(2).float()  # (B, S, 1)
        sum_vec = (x * valid).sum(dim=1)              # (B, D_out)
        lengths = valid.sum(dim=1).clamp(min=1.0)     # (B, 1)
        pooled = sum_vec / lengths                   # (B, D_out)

        # decoder で各アカウントへのスコアを出力
        logits = self.decoder(pooled)                # (B, num_all_accounts)
        return logits, pooled
