
import torch
import torch.nn.functional as F
from torch import nn

from .fusion_dmae import FusionEmbeddingMAE


class FusionMAE(nn.Module):


    def __init__(
        self,
        backbone: nn.Module,
        d_model: int = 256,
        patch_size: int = 16,
        vocab_size: int = 512,
        mask_ratio: float = 0.6,
        reg_layers: int = 2,
        attn_heads: int = 8,
        momentum: float = 0.999,
        device=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.momentum = momentum
        embedding_dim = backbone.get_fusion_embedding_dim()

        self.mae_head = FusionEmbeddingMAE(
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            encoder_dim=d_model,
            encoder_depth=reg_layers,
            decoder_dim=d_model,
            decoder_depth=2,
            heads=attn_heads,
            mlp_dim=512,
            mask_ratio=mask_ratio,
            dropout=0.1,
        )
        if device is not None:
            self.mae_head = self.mae_head.to(device)

    @property
    def mask_ratio(self):
        return self.mae_head.mask_ratio

    def compute_loss(self, signals, age, labels=None, ce_weight=0.2):
        """
        signals: 模型输入（dict 或 tensor）
        age: 年龄
        labels: 可选，分类标签，用于辅助 CE 损失
        ce_weight: CE 辅助损失权重
        Returns:
            loss: 标量
            log_dict: {"total_loss", "mse_loss", "ce_loss"}
        """
        with torch.set_grad_enabled(self.training):
            embeddings = self.backbone.forward_embeddings(signals, age, apply_dropout=True)
        mse_loss, _, _ = self.mae_head(embeddings)
        ce_loss = torch.tensor(0.0, device=mse_loss.device, dtype=mse_loss.dtype)
        if labels is not None and self.training:
            logits = self.backbone(signals, age)
            ce_loss = F.cross_entropy(logits, labels.long(), reduction="mean")
        total_loss = mse_loss + ce_weight * ce_loss
        log_dict = {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "ce_loss": ce_loss,
        }
        return total_loss, log_dict

    def momentum_update(self):
        """预留：可在此更新 momentum encoder，当前为 no-op。"""
        pass
