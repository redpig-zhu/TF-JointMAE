import gc
import os
from copy import deepcopy

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch

# ------------------------------
# 数据与模型相关模块（已隐藏）
# ------------------------------
from datasets.caueeg_script import build_dataset_for_train
from train.pretrain_script import pretrain_script
from models.fusion_dmae import FusionEmbeddingMAE


def check_device_env(config):
    """
    设备与运行环境检查
    """
    if not torch.cuda.is_available():
        raise ValueError("ERROR: No GPU is available.")

    config["device"] = torch.device(config.get("device", "cuda"))
    device_name = torch.cuda.get_device_name(0)

    # -------- minibatch size 自动推断逻辑 --------
    # --------------------------------------------------
    # | 根据 GPU 型号动态调整 minibatch |
    # | 3090 / 2080 / 1080 / 1070 等 |
    # --------------------------------------------------
    # ..................................................
    # ..................................................
    # ..................................................


def prepare_and_run_pretrain(config):
    """
    预训练主流程封装
    """
    # -------- 资源清理 --------
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # -------- 随机种子控制 --------
    # ..................................................
    # ..................................................

    # -------- 构建数据集 --------
    train_loader, val_loader, _, _ = build_dataset_for_train(config)

    # -------- 模型实例化 --------
    model = hydra.utils.instantiate(config)
    model = model.to(config["device"])

    # -------- MAE 预训练头 --------
    mae_cfg = deepcopy(config.get("pretrain_head", {}))
    # ..................................................
    # ..................................................
    mae_head = FusionEmbeddingMAE(**mae_cfg).to(config["device"])

    # -------- 调用核心预训练逻辑 --------
    pretrain_script(
        config=config,
        model=model,
        mae_head=mae_head,
        train_loader=train_loader,
        val_loader=val_loader,
        preprocess_train=config["preprocess_train"],
        preprocess_val=config["preprocess_test"],
    )


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydra 入口：配置组装
    """
    # -------- 配置展开与合并 --------
    config = {
        # ..................................................
        # ..................................................
        # ..................................................
        "cwd": HydraConfig.get().runtime.cwd,
    }

    check_device_env(config)
    prepare_and_run_pretrain(config)


if __name__ == "__main__":
    main()
