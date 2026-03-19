import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os

from models.vgg_fusion import VGG1D2DFusion
from models.fusion_mae import FusionMAE
from datasets.caueeg_script import build_dataset_for_train, load_caueeg_config


def pretrain_epoch(mae_model, train_loader, device, preprocess_train, scaler=None, max_grad_norm=1.0):
    """预训练一个epoch，支持混合精度和梯度裁剪"""
    mae_model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_ce_loss = 0.0
    # 核心细节暂时隐藏：****
    # （论文发表后放出）
    
    for batch in tqdm(train_loader, desc="Training"):
        # 预处理数据
        preprocess_train(batch)
        signals = batch["signal"]
        age = batch["age"]
        labels = batch.get("class_label")  # 用于分类辅助损失

        mae_model.optimizer.zero_grad()
        
        # 混合精度训练（MSE 重建 + CE 分类辅助）
        with autocast(enabled=(scaler is not None)):
            raise RuntimeError("核心训练逻辑暂时隐藏：****（论文发表后放出）。")
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪 - 稍微放宽梯度裁剪阈值，允许更大的梯度更新
            scaler.unscale_(mae_model.optimizer)
            torch.nn.utils.clip_grad_norm_(mae_model.parameters(), max_grad_norm)
            scaler.step(mae_model.optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(mae_model.parameters(), max_grad_norm)
            mae_model.optimizer.step()
        
        mae_model.momentum_update()

        total_loss += log_dict["total_loss"].item()
        total_mse_loss += log_dict["mse_loss"].item()
        total_ce_loss += log_dict["ce_loss"].item()
    
    n_batches = len(train_loader)
    return {
        "loss": total_loss / n_batches,
        "mse_loss": total_mse_loss / n_batches,
        "ce_loss": total_ce_loss / n_batches,
    }


@torch.no_grad()
def evaluate_backbone(backbone, val_loader, device, config):
    """评估backbone在下游任务上的性能"""
    backbone.eval()
    correct = 0
    total = 0
    
    preprocess_test = config.get("preprocess_test")
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        # 使用测试预处理
        if preprocess_test is not None:
            preprocess_test(batch)
        else:
            # 如果没有preprocess_test，使用简单的预处理
            batch["signal"] = batch["signal"].to(device)
            batch["age"] = batch["age"].to(device)
            batch["class_label"] = batch["class_label"].to(device)
        
        signals = batch["signal"]
        age = batch["age"]
        labels = batch["class_label"]
        
        # 使用backbone进行预测
        output = backbone(signals, age)
        pred = output.argmax(dim=-1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """带warmup的cosine学习率调度器，支持min_lr防止衰减过小"""
    def lr_lambda(current_step):
        # 核心调度策略暂时隐藏：****
        # （论文发表后放出）
        raise RuntimeError("核心LR调度逻辑暂时隐藏：****（论文发表后放出）。")
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FusionMAE 预训练（4 分类时 out_dims 由数据集 config 决定）")
    parser.add_argument("--dataset_path", type=str, default="datasets/caueeg-dataset", help="数据集根路径")
    parser.add_argument("--file_format", type=str, default="feather", choices=["edf", "feather"], help="信号文件格式")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ===== 1) 构造 config 并用官方接口拿 loader =====
    dataset_path = args.dataset_path
    
    # 加载数据集配置（包含 signal_header 等）
    config = load_caueeg_config(dataset_path)
    
    # 添加 build_dataset_for_train 需要的必需字段
    config["dataset_path"] = dataset_path
    config["task"] = "dementia"  # 或 "abnormal"，根据你的任务选择
    # 具体2D-VGG-19模型配置暂时隐藏：****（论文发表后放出）
    config["model"] = "****"
    config["seq_length"] = 2000  # 200Hz * 10s = 2000
    config["file_format"] = args.file_format
    config["load_event"] = False
    config["device"] = device
    config["EKG"] = "O"  # "O" 表示使用, "X" 表示不使用
    config["photic"] = "X"  # "O" 表示使用, "X" 表示不使用
    config["crop_multiple"] = 1
    config["test_crop_multiple"] = 8
    config["latency"] = 2000
    config["signal_length_limit"] = 10000000
    config["input_norm"] = "dataset"
    config["minibatch"] = 32  # 根据你的 GPU 内存调整
    config["awgn"] = 0.005  # 增加数据增强强度，提高泛化能力
    config["awgn_age"] = 0.005  # 增加数据增强强度
    config["mgn"] = 0.005  # 增加数据增强强度

    train_loader, val_loader, _, _ = build_dataset_for_train(config)
    
    # build_dataset_for_train 会设置 in_channels, out_dims, seq_len_2d 等；total_channels 用 in_channels
    if "in_channels" in config:
        config["total_channels"] = config["in_channels"]
    elif "total_channels" not in config:
        config["total_channels"] = 20
    if "seq_len_1d" not in config:
        config["seq_len_1d"] = config.get("seq_length", 2000)  # 如果没有设置，使用 seq_length
    if "seq_len_2d" not in config:
        config["seq_len_2d"] = config.get("seq_len_2d", 64)
    if "use_age" not in config:
        config["use_age"] = config.get("use_age", "fc")

    # ===== 2) 构造 backbone（1D+2D 融合 VGG） =====
    backbone = VGG1D2DFusion(
        model="****",  # 原为 2D-VGG-19，暂时隐藏：论文发表后放出
        total_channels=config["total_channels"],   # 建议用 config 里的参数
        out_dims=config["out_dims"],              # 虽然预训练不用，但保持一致
        seq_len_1d=config["seq_len_1d"],
        seq_len_2d=config["seq_len_2d"],
        use_age=config.get("use_age", "fc"),
    ).to(device)

    # ===== 3) 构造 FusionMAE 包装器 =====
    mae_model = FusionMAE(
        backbone=backbone,
        d_model=256,
        patch_size=16,
        vocab_size=512,
        mask_ratio=0.6,  # 降低mask ratio，平衡预训练难度和效果（0.6-0.65通常效果更好）
        reg_layers=2,
        attn_heads=8,
        momentum=0.999,  # 保持高momentum，使target更稳定
        device=device,
    ).to(device)

    # ===== 4) 优化器和学习率调度器 =====
    # 使用更合适的学习率和weight decay
    base_lr = 1.5e-4  # 提高初始学习率，加快收敛
    weight_decay = 0.05  # 保持正则化
    
    optimizer = torch.optim.AdamW(
        mae_model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),  # 使用更标准的beta参数
    )
    
    # 学习率调度：按 epoch 步进（scheduler.step 每 epoch 调一次）
    num_epochs = 200
    num_training_steps = num_epochs  # 按 epoch 计数
    num_warmup_steps = int(0.05 * num_epochs)  # 前 10 个 epoch warmup，尽快进入有效学习率
    num_training_steps_lr = num_epochs * len(train_loader)  # 用于打印
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.05,  # 最低 LR 为 base 的 5%，避免后期几乎不更新
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 挂到 mae_model 上
    mae_model.optimizer = optimizer
    mae_model.scheduler = scheduler


    checkpoint_dir = "local/checkpoint/vggdmae"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ===== 6) 预训练循环，带验证（无早停，固定200轮） =====
    best_val_acc = 0.0
    
    print(f"开始预训练，共 {num_epochs} 个epoch（无早停）")
    print(f"LR调度: {num_training_steps} steps, Warmup {num_warmup_steps} epochs, min_lr=5%×base")
    print(f"初始学习率: {base_lr}, Weight decay: {weight_decay}")
    print(f"Mask ratio: {mae_model.mask_ratio}, Batch size: {config['minibatch']}")
    print(f"损失: MSE + CE辅助(权重0.2), Val Acc: 分类头随训练更新，可反映embedding质量")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        # 训练
        train_metrics = pretrain_epoch(
            mae_model, train_loader, device, 
            preprocess_train=config["preprocess_train"],
            scaler=scaler, 
            max_grad_norm=1.0
        )
        scheduler.step()
        
        # 每个epoch都进行验证，更及时地跟踪模型性能
        val_acc = evaluate_backbone(backbone, val_loader, device, config)
        
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Loss: {train_metrics['loss']:.4f} "
              f"(MSE: {train_metrics['mse_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f}) | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # 保存完整的checkpoint（包含所有信息，便于恢复训练）
            # 保存格式与 run_train.py 兼容
            best_checkpoint = {
                "model_state": backbone.state_dict(),  # 与 run_train.py 兼容的格式
                "mae_model_state": mae_model.state_dict(),  # 完整的FusionMAE模型（可选）
                "config": config,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "epoch": epoch + 1,
                "best_val_acc": best_val_acc,
                "train_metrics": train_metrics,
                "ddp": False,  # 添加ddp标志，与run_train.py兼容
            }
            # 保存到与 run_train.py 兼容的路径格式
            checkpoint_dir = "local/checkpoint/vgg19_dmae"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                best_checkpoint,
                os.path.join(checkpoint_dir, "checkpoint.pt")
            )
            # 同时保存backbone权重（便于直接加载）
            torch.save(
                backbone.state_dict(),
                os.path.join(checkpoint_dir, "vgg_fusion_mae_best.pth")
            )
            print(f"  -> 保存最佳模型到 {checkpoint_dir}/checkpoint.pt (Val Acc: {best_val_acc:.2f}%)")
        
        # 定期保存checkpoint（每10个epoch保存一次）
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = "local/checkpoint/vgg19_dmae"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                "model_state": backbone.state_dict(),
                "mae_model_state": mae_model.state_dict(),
                "config": config,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "epoch": epoch + 1,
                "best_val_acc": best_val_acc,
                "train_metrics": train_metrics,
                "ddp": False,  # 添加ddp标志，与run_train.py兼容
            }
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, "checkpoint.pt")
            )
            # 同时保存带epoch编号的checkpoint
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
            )
            torch.save(
                backbone.state_dict(),
                os.path.join(checkpoint_dir, f"vgg_fusion_mae_epoch{epoch+1}.pth")
            )
    
    print("-" * 80)
    print(f"预训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    checkpoint_dir = "local/checkpoint/vgg19_dmae"
    print(f"最佳模型已保存到:")
    print(f"  - {checkpoint_dir}/checkpoint.pt (完整checkpoint，包含优化器状态)")
    print(f"  - {checkpoint_dir}/vgg_fusion_mae_best.pth (仅backbone权重)")



if __name__ == "__main__":
    main()
