# model.py
import torch
import torch.nn as nn

class SetToVectorPredictor(nn.Module):
    """
    入力: 
      - posts: (B, S, D_in) の投稿埋め込み集合
      - padding_mask: (B, S) BoolTensor, True=padding
    出力:
      - logits: (B, num_all_accounts) フォロー確率スコア
      - pooled: (B, encoder_output_dim) プーリング後表現
    """
    def __init__(
        self,
        post_embedding_dim: int,
        encoder_output_dim: int,
        num_all_accounts: int,
        num_attention_heads: int,
        num_encoder_layers: int,
        dropout_rate: float,
        num_pma_seeds: int = 1,         # PMA のシード数
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
            batch_first=False,  # permute で (S,B,D) にするので False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 3) PMA (Pooling by Multihead Attention)
        #    key/value にエンコーダ出力、query に learnable シードを使う
        self.num_pma_seeds = num_pma_seeds
        self.pma_seed = nn.Parameter(torch.empty(num_pma_seeds, encoder_output_dim))
        nn.init.xavier_uniform_(self.pma_seed)
        self.pma = nn.MultiheadAttention(
            embed_dim=encoder_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=False
        )

        # 4) プーリング後の分類器
        hidden_dim = encoder_output_dim // 2
        self.decoder = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_all_accounts)
        )

    def forward(self, posts: torch.Tensor, padding_mask: torch.Tensor):
        """
        posts:         (B, S, D_in)
        padding_mask:  (B, S), True=padding
        """
        B, S, _ = posts.size()

        # (B, S, D)
        x = self.initial_projection(posts)

        # Transformer に入れる形式 (S, B, D)
        x = x.permute(1, 0, 2)

        # エンコード (S, B, D)
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # PMA プーリング
        # query: (num_seeds, B, D)
        seeds = self.pma_seed.unsqueeze(1).expand(-1, B, -1)
        # key/value: x (S, B, D)
        pooled, _ = self.pma(
            query=seeds,
            key=x,
            value=x,
            key_padding_mask=padding_mask
        )
        # pooled: (num_seeds, B, D)
        if self.num_pma_seeds > 1:
            # 複数シードは平均する
            pooled = pooled.mean(dim=0)  # → (B, D)
        else:
            pooled = pooled.squeeze(0)  # → (B, D)

        # decoder へ (B, D)
        logits = self.decoder(pooled)  # (B, num_all_accounts)

        return logits, pooled