import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class _TransformerBlock(nn.Module):

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class _TransformerEncoder(nn.Module):
    """Stacked Transformer blocks"""

    def __init__(self, depth: int, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [_TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FusionEmbeddingMAE(nn.Module):


    def __init__(
        self,
        embedding_dim: int,
        patch_size: int = 64,
        encoder_dim: int = 256,
        encoder_depth: int = 4,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        heads: int = 8,
        mlp_dim: int = 512,
        mask_ratio: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.pad = (patch_size - (embedding_dim % patch_size)) % patch_size
        self.num_patches = (embedding_dim + self.pad) // patch_size
        self.mask_ratio = min(max(mask_ratio, 0.05), 0.9)

        self.patch_embed = nn.Linear(patch_size, encoder_dim)
        self.encoder_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim))
        self.encoder = _TransformerEncoder(
            depth=encoder_depth,
            dim=encoder_dim,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))
        self.decoder_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, decoder_dim))
        self.decoder = _TransformerEncoder(
            depth=decoder_depth,
            dim=decoder_dim,
            heads=heads,
            mlp_dim=mlp_dim * 2,
            dropout=dropout,
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.output_proj = nn.Linear(decoder_dim, patch_size)

    def forward(self, embeddings: torch.Tensor, mask_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        patches = self._patchify(embeddings)
        batch, num_patches, _ = patches.shape

        mask, visible_idx, masked_idx = self._build_mask(batch, num_patches, embeddings.device, mask_indices)

        token_embeddings = self.patch_embed(patches) + self.encoder_pos_emb[:, :num_patches]
        visible_tokens = token_embeddings.gather(
            1, visible_idx.unsqueeze(-1).expand(-1, -1, token_embeddings.size(-1))
        )
        encoded = self.encoder_norm(self.encoder(visible_tokens))
        decoded_visible = self.encoder_to_decoder(encoded)

        dtype = decoded_visible.dtype
        decoder_tokens = torch.zeros(batch, num_patches, decoded_visible.size(-1), device=embeddings.device, dtype=dtype)
        decoder_tokens.scatter_(
            1,
            visible_idx.unsqueeze(-1).expand_as(decoded_visible),
            decoded_visible,
        )
        mask_tokens = self.mask_token.expand(batch, masked_idx.size(1), -1).to(dtype=dtype)
        decoder_tokens.scatter_(
            1, masked_idx.unsqueeze(-1).expand(-1, -1, mask_tokens.size(-1)), mask_tokens
        )

        decoder_tokens = decoder_tokens + self.decoder_pos_emb[:, :num_patches]
        decoded = self.decoder_norm(self.decoder(decoder_tokens))
        pred_patches = self.output_proj(decoded)

        loss = F.mse_loss(pred_patches[mask], patches[mask])
        recon = self._unpatchify(pred_patches)
        return loss, recon, mask

    def reconstruct(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Utility for inference-only reconstruction."""
        _, recon, _ = self.forward(embeddings)
        return recon

    def _patchify(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            embeddings = F.pad(embeddings, (0, self.pad))
        patches = embeddings.view(embeddings.size(0), self.num_patches, self.patch_size)
        return patches

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        embeddings = patches.view(patches.size(0), -1)
        if self.pad > 0:
            embeddings = embeddings[:, :-self.pad]
        return embeddings

    def _build_mask(
        self,
        batch: int,
        num_patches: int,
        device: torch.device,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask_indices is None:
            shuffle = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
            num_mask = max(1, int(num_patches * self.mask_ratio))
            masked_idx = shuffle[:, :num_mask]
            visible_idx = shuffle[:, num_mask:]
        else:
            masked_idx = mask_indices
            all_indices = torch.arange(num_patches, device=device).unsqueeze(0).expand(batch, -1)
            visible_mask = torch.ones(batch, num_patches, dtype=torch.bool, device=device)
            visible_mask.scatter_(1, masked_idx, False)
            visible_idx = torch.stack(
                [all_indices[b][visible_mask[b]] for b in range(batch)]
            )

        mask = torch.zeros(batch, num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_idx, True)
        return mask, visible_idx, masked_idx.long()

