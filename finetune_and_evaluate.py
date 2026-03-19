
import os
from collections import OrderedDict

import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from datasets.caueeg_script import build_dataset_for_train, load_caueeg_config
from train.evaluate import (
    check_accuracy, check_accuracy_extended,
    check_accuracy_multicrop_extended,
    calculate_class_wise_metrics
)
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# 统一控制所有输出结果的根目录
BASE_OUTPUT_DIR = "outputs_all1"


def focal_loss(logits, targets, alpha=None, gamma=2.0):

    raise RuntimeError("核心 focal_loss 暂时隐藏：****（论文发表后放出）。")


def load_pretrained_weights(model, checkpoint_path, device, strict=False):
    """加载预训练权重"""
    print(f"加载预训练权重: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # 处理不同的checkpoint格式
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            # 完整checkpoint格式
            pretrained_state = ckpt["model_state"]
            print(f"从完整checkpoint加载，epoch: {ckpt.get('epoch', 'unknown')}, "
                  f"best_val_acc: {ckpt.get('best_val_acc', 'unknown')}")
        elif "backbone" in ckpt:
            pretrained_state = ckpt["backbone"]
        else:
            # 直接是state_dict
            pretrained_state = ckpt
    else:
        pretrained_state = ckpt


    has_vgg_backbone = any(k.startswith("vgg_backbone.") for k in pretrained_state.keys())

    if has_vgg_backbone:
        # 提取vgg_backbone的权重
        vgg_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k.startswith("vgg_backbone."):
                new_key = k[len("vgg_backbone."):]
                vgg_state[new_key] = v
        print(f"从PretrainMAE提取权重: {len(vgg_state)} 个参数")
        pretrained_state = vgg_state

    # 获取当前模型的state_dict
    current_state = model.state_dict()

    raise RuntimeError("核心预训练权重加载逻辑暂时隐藏：****（论文发表后放出）。")

    for k, v in pretrained_state.items():
        # 如果通道数不匹配，跳过第一层
        if k in skip_first_layers:
            if k in current_state and current_state[k].shape != v.shape:
                skipped_layers.append(f"{k} (跳过第一层，通道数不匹配: {v.shape} vs {current_state[k].shape})")
                continue

        if k in current_state:
            if current_state[k].shape == v.shape:
                filtered_state[k] = v
                loaded_layers.append(k)
            else:
                skipped_layers.append(f"{k} (shape mismatch: {v.shape} vs {current_state[k].shape})")
        else:
            skipped_layers.append(f"{k} (not in model)")

    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=strict)

    print(f"成功加载 {len(loaded_layers)} 个层的权重")
    if missing_keys:
        print(f"缺失的层 ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"意外的层 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    if skipped_layers:
        print(f"跳过的层 ({len(skipped_layers)}): {skipped_layers[:5]}...")
        if len(skipped_layers) > 5:
            print(f"  ... 还有 {len(skipped_layers) - 5} 个层被跳过")

    # 检查是否有第一层被跳过（通道数不匹配的情况）
    first_layer_skipped = any("第一层" in s or "conv_stage1" in s for s in skipped_layers)
    if first_layer_skipped:
        print("\n" + "=" * 60)

        print("=" * 60 + "\n")

    # 警告：如果跳过的层太多，说明模型结构不匹配
    total_layers = len(current_state)
    skip_ratio = len(skipped_layers) / total_layers if total_layers > 0 else 0


    return model


def calculate_class_weights(train_loader, num_classes, device, class_label_to_name=None):
    """计算类别权重（处理类别不平衡）"""
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    total_samples = 0

    print("\n计算类别权重（处理类别不平衡）...")
    print("遍历训练集统计类别分布...")

    for batch in tqdm(train_loader, desc="统计类别"):
        labels = batch["class_label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu()
        else:
            labels = torch.tensor(labels).cpu()

        # 统计每个类别的样本数
        unique_labels, counts = torch.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_counts[label.item()] += count.item()
            total_samples += count.item()

    class_counts_float = class_counts.float()
    class_counts_float[class_counts_float == 0] = 1.0  # 避免除零

    class_weights = total_samples / (num_classes * class_counts_float)
    # 归一化权重（让权重更平滑，总和等于类别数）
    class_weights = class_weights / class_weights.sum() * num_classes

    # 打印类别分布和权重
    print(f"\n类别分布（总样本数: {total_samples}）:")
    for i in range(num_classes):
        class_name = class_label_to_name[i] if class_label_to_name and i < len(class_label_to_name) else f"Class {i}"
        print(
            f"  {class_name}: {class_counts[i].item()} 个样本 ({100.0 * class_counts[i].item() / total_samples:.1f}%)")

    print(f"\n类别权重:")
    for i in range(num_classes):
        class_name = class_label_to_name[i] if class_label_to_name and i < len(class_label_to_name) else f"Class {i}"
        print(f"  {class_name}: {class_weights[i].item():.4f}")

    print(f"\n权重说明: 权重越大，该类别在损失函数中的重要性越高")
    print("=" * 60 + "\n")

    return class_weights.to(device)


def finetune_epoch(model, train_loader, optimizer, scheduler, device, preprocess_train,
                   scaler=None, max_grad_norm=1.0, class_weights=None, use_focal=True, gamma=2.0):
    """微调一个epoch
    使用Focal Loss + 类别权重，更好地处理类别不平衡和难样本
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Fine-tuning"):
        preprocess_train(batch)
        signals = batch["signal"]
        age = batch["age"]
        labels = batch["class_label"]

        optimizer.zero_grad()

        with autocast(enabled=(scaler is not None)):
            output = model(signals, age)
            # 使用Focal Loss + 类别权重（更关注难样本和少数类）
            if use_focal:
                loss = focal_loss(output, labels, alpha=class_weights, gamma=gamma)
            else:
                # 降级到加权交叉熵
                if class_weights is not None:
                    loss = F.cross_entropy(output, labels, weight=class_weights)
                else:
                    loss = F.cross_entropy(output, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # 计算准确率
        pred = output.argmax(dim=-1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    # 注意：scheduler.step()应该在epoch结束后调用，而不是每个batch
    # 这里不调用，让主函数在每个epoch后调用

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def extract_latent_features(model, data_loader, preprocess_fn, device, add_noise=False, noise_std=0.1):
    """
    提取模型倒数第二层（分类头前）的潜在特征，用于 UMAP 可视化

    Args:
        model: 模型
        data_loader: 数据加载器
        preprocess_fn: 预处理函数
        device: 设备
        add_noise: 是否添加高斯噪声
        noise_std: 噪声标准差
    Returns:
        features: [N, D] 潜在特征
        labels: [N] 类别标签
    """
    model.eval()
    features_list = []
    labels_list = []

    for batch in data_loader:
        preprocess_fn(batch)
        signals = batch["signal"].to(device)
        age = batch.get("age", None)
        if age is not None:
            age = age.to(device)
        labels = batch["class_label"]

        # 添加噪声
        if add_noise:
            noise = torch.randn_like(signals) * noise_std
            signals = signals + noise

        # 提取分类头前的特征（forward_embeddings）
        if hasattr(model, "forward_embeddings"):
            feat = model.forward_embeddings(signals, age)
        else:
            # fallback: 直接使用 forward 输出
            feat = model(signals, age)

        features_list.append(feat.cpu())
        labels_list.append(labels.cpu())

    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    return features, labels



@torch.no_grad()
def sample_random_predictions_by_class(
    model,
    data_loader,
    preprocess_fn,
    device,
    class_label_to_name,
    samples_per_class=None,
):

    import random
    from collections import defaultdict

    model.eval()

    if samples_per_class is None:
        # 只关心 Normal / MCI / Dementia，SCD 不抽样
        samples_per_class = {0: 3, 2: 2, 3: 3}


    collected = defaultdict(list)

    def _is_done():
        for cls, n in samples_per_class.items():
            if len(collected[cls]) < n:
                return False
        return True

    for batch in data_loader:
        preprocess_fn(batch)
        signals = batch["signal"].to(device)
        age = batch.get("age", None)
        if age is not None:
            age = age.to(device)
        labels = batch["class_label"].to(device)

        logits = model(signals, age)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        batch_size = labels.size(0)
        indices = list(range(batch_size))
        random.shuffle(indices)

        for idx in indices:
            true_label = int(labels[idx].item())
            if true_label not in samples_per_class:
                continue
            if len(collected[true_label]) >= samples_per_class[true_label]:
                continue

            sample_info = {
                "true_label": true_label,
                "pred_label": int(preds[idx].item()),
                "probs": probs[idx].detach().cpu().numpy().tolist(),
            }


            collected[true_label].append(sample_info)

        if _is_done():
            break





@torch.no_grad()
def evaluate_model(
    model,
    val_loader,
    test_loader,
    multicrop_test_loader,
    preprocess_test,
    config,
    device,
    output_dir=BASE_OUTPUT_DIR,

    run_test_eval=True,
    run_multicrop_test_eval=True,
):

    model.eval()

    print("\n" + "=" * 80)
    print("开始评估模型...")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    class_label_to_name = config.get("class_label_to_name", ["Normal", "SCD", "MCI", "Dementia"])


    val_acc = check_accuracy(model, val_loader, preprocess_test, config, repeat=1)
    val_score, val_target, val_confusion_matrix, val_throughput = (None, None, None, None)

    # ========== 测试集完整评估 ==========
    test_acc = None
    test_score = None
    test_target = None
    test_confusion_matrix = None
    test_throughput = None
    if run_test_eval:
        print("\n" + "=" * 80)
        print("[测试集完整评估]")
        print("=" * 80)

        test_acc, test_score, test_target, test_confusion_matrix, test_throughput = check_accuracy_extended(
            model, test_loader, preprocess_test, config, repeat=1, dummy=0
        )

        print(f"\n测试集准确率: {test_acc:.2f}%")
        print(f"吞吐量: {test_throughput:.2f} samples/sec")
        print("\n测试集混淆矩阵:")
        print(test_confusion_matrix)

    # ========== 多裁剪测试集完整评估（可选） ==========
    multicrop_test_acc = None
    multicrop_score = None
    multicrop_target = None
    multicrop_confusion_matrix = None
    multicrop_throughput = None
    if run_multicrop_test_eval:
        print("\n" + "=" * 80)
        print("[多裁剪测试集完整评估]")
        print("=" * 80)

        multicrop_test_acc, multicrop_score, multicrop_target, multicrop_confusion_matrix, multicrop_throughput = (
            check_accuracy_multicrop_extended(
                model, multicrop_test_loader, preprocess_test, config, repeat=1, dummy=0
            )
        )

        print(f"\n多裁剪测试集准确率: {multicrop_test_acc:.2f}%")
        print(f"吞吐量: {multicrop_throughput:.2f} samples/sec")
        print("\n多裁剪测试集混淆矩阵:")
        print(multicrop_confusion_matrix)

        # 计算多裁剪测试集 Class-wise metrics
        multicrop_class_wise_metrics = calculate_class_wise_metrics(multicrop_confusion_matrix)
        multicrop_sensitivity = multicrop_class_wise_metrics["Sensitivity"]
        multicrop_specificity = multicrop_class_wise_metrics["Specificity"]
        multicrop_precision = multicrop_class_wise_metrics["Precision"]
        multicrop_f1_score = multicrop_class_wise_metrics["F1-score"]

        print("\n多裁剪测试集类别级别指标:")
        for i, name in enumerate(class_label_to_name):
            print(f"  {name}:")
            print(f"    Sensitivity: {multicrop_sensitivity[i]:.4f}")
            print(f"    Specificity: {multicrop_specificity[i]:.4f}")
            print(f"    Precision:   {multicrop_precision[i]:.4f}")
            print(f"    F1-score:    {multicrop_f1_score[i]:.4f}")
    

    candidates = {}
    if run_test_eval and test_acc is not None:
        candidates["test"] = {
            "tag": "test",
            "acc": test_acc,
            "score": test_score,
            "target": test_target,
            "confusion": test_confusion_matrix,
        }
    if run_multicrop_test_eval and multicrop_test_acc is not None:
        candidates["multicrop_test"] = {
            "tag": "multicrop_test",
            "acc": multicrop_test_acc,
            "score": multicrop_score,
            "target": multicrop_target,
            "confusion": multicrop_confusion_matrix,
        }






    print("\n" + "=" * 80)
    print("评估完成！(已移除所有可视化绘图，仅保留数值结果)")
    print("=" * 80)
    if test_acc is not None:
        print(f"测试集准确率: {test_acc:.2f}%")
    if multicrop_test_acc is not None:
        print(f"多裁剪测试集准确率: {multicrop_test_acc:.2f}%")

    print("=" * 80)
    return {
        "val_acc": val_acc,
        "val_confusion_matrix": val_confusion_matrix,
        "val_throughput": val_throughput,
        "test_acc": test_acc,
        "test_confusion_matrix": test_confusion_matrix,
        "test_throughput": test_throughput,
        "multicrop_test_acc": multicrop_test_acc,
        "multicrop_test_confusion_matrix": multicrop_confusion_matrix,
        "multicrop_test_throughput": multicrop_throughput,

    }








def main_finetune_and_evaluate(
        pretrain_path,
        dataset_path="datasets/caueeg-dataset",
        task="dementia",
        file_format=None,
        finetune_epochs=10,
        finetune_lr=1e-4,
        finetune_batch_size=32,
        device="cuda",
        skip_finetune=False,
        n_monte_carlo_runs=1,
        random_seeds=None,

        skip_multicrop_test_eval=False,
):

    device = torch.device(device)

    print("=" * 80)
    print("加载预训练权重并评估")
    print("=" * 80)
    print(f"预训练权重路径: {pretrain_path}")
    print(f"数据集路径: {dataset_path}")
    print(f"任务: {task}")
    print(f"微调: {'跳过' if skip_finetune else f'{finetune_epochs} epochs'}")
    if n_monte_carlo_runs > 1:
        print(f"运行次数: {n_monte_carlo_runs}")
    print("=" * 80)

    # 生成随机种子列表
    if random_seeds is None:
        # 自动生成随机种子
        random_seeds = [42 + i * 100 for i in range(n_monte_carlo_runs)]
    else:
        # 确保随机种子数量与运行次数匹配
        if len(random_seeds) != n_monte_carlo_runs:
            print(f"警告: 随机种子数量({len(random_seeds)})与运行次数({n_monte_carlo_runs})不匹配")
            random_seeds = random_seeds[:n_monte_carlo_runs]
    
    # 用于存储所有运行的结果
    all_run_results = []
    
    os.makedirs("local/checkpoints", exist_ok=True)
    # 汇总结果仍放在固定目录；每次运行的图/曲线统一放到该 run 的 viz 目录里
    curve_dir = os.path.join(BASE_OUTPUT_DIR, "finetune_curves")
    os.makedirs(curve_dir, exist_ok=True)


    # ===== 开始训练循环 =====
    for run_idx, current_seed in enumerate(random_seeds):
        print("\n" + "=" * 80)
        print(f"运行 [{run_idx + 1}/{n_monte_carlo_runs}] - 随机种子: {current_seed}")
        print("=" * 80 + "\n")
        
        # 重置每次运行的指标
        best_val_acc = 0.0
        best_epoch = 0
        
        # 为每次运行创建独立的输出目录（即使只有一次运行，也统一按 run_1_seed_xxx 命名）
        run_output_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{run_idx + 1}_seed_{current_seed}")
        os.makedirs(run_output_dir, exist_ok=True)



        os.makedirs(run_viz_dir, exist_ok=True)
    
        # ===== 1. 准备配置和数据 =====
        config = load_caueeg_config(dataset_path)
        config["dataset_path"] = dataset_path
        config["task"] = task

        config["model"] = "****"
        config["seq_length"] = 2000
        # 统一设置痴呆任务的类别名称顺序：
        # 0 -> Normal, 1 -> SCD/MCI, 2 -> AD
        if task == "dementia":
            config["class_label_to_name"] = ["Normal", "SCD/MCI", "AD"]
        # 设置文件格式：优先使用用户指定的，否则自动检测
        if file_format is not None:
            config["file_format"] = file_format
            print(f"✓ 使用指定的文件格式: {file_format}")
        elif "file_format" not in config:
            # 自动检测文件格式：优先使用feather（更快），如果不存在则使用edf
            feather_dir = os.path.join(dataset_path, "signal", "feather")
            edf_dir = os.path.join(dataset_path, "signal", "edf")
            if os.path.exists(feather_dir) and len(os.listdir(feather_dir)) > 0:
                config["file_format"] = "feather"
                print(f"✓ 检测到feather格式数据，使用: feather")
            elif os.path.exists(edf_dir) and len(os.listdir(edf_dir)) > 0:
                config["file_format"] = "edf"
                print(f"✓ 检测到edf格式数据，使用: edf")
            else:
                config["file_format"] = "edf"  # 默认
                print(f"⚠ 无法检测文件格式，默认使用: edf")
        else:
            print(f"✓ 使用配置中的文件格式: {config['file_format']}")
        config["load_event"] = False
        config["device"] = device
        config["EKG"] = "O"
        config["photic"] = "X"
        config["crop_multiple"] = 1
        # 多裁剪次数（测试时对同一条记录裁多少段再平均）
        # 原来是 8，这里改大一点，比如 16
        config["test_crop_multiple"] = 16
        config["latency"] = 2000
        config["signal_length_limit"] = 10000000
        config["input_norm"] = "dataset"
        config["minibatch"] = finetune_batch_size
        config["awgn"] = 0.001
        config["awgn_age"] = 0.001
        config["mgn"] = 0.001
        config["criterion"] = "cross-entropy"  # 评估需要的损失函数类型
        # 先关闭 MCI 回退策略，直接使用 argmax 预测，以避免所有样本过度被归为 MCI
        # 如需重新启用，可将 use_mci_fallback 设为 True 并调整 mci_fallback_threshold
        config["use_mci_fallback"] = False
        config["mci_fallback_threshold"] = 0.48

        # 构建数据集，使用分层随机划分策略
        print(f"正在构建数据集（随机种子: {current_seed}）...")
        train_loader, val_loader, test_loader, multicrop_test_loader = build_dataset_for_train(
            config, 
            random_seed=current_seed
        )

        # build_dataset_for_train 会设置 out_dims, in_channels, seq_len_2d 等
        # 使用实际数据的通道数
        if "in_channels" in config:
            actual_total_channels = config["in_channels"]
            print(f"✓ 使用实际数据的通道数: {actual_total_channels}")
            config["total_channels"] = actual_total_channels
        elif "total_channels" not in config:
            # 如果都没有，使用默认值
            config["total_channels"] = 20
            print(f"⚠ 使用默认total_channels: {config['total_channels']}")

        # 现在从checkpoint加载其他配置（但不覆盖通道数）
        checkpoint_config = None
        if os.path.exists(pretrain_path):
            try:
                ckpt = torch.load(pretrain_path, map_location="cpu")
                if isinstance(ckpt, dict) and "config" in ckpt:
                    checkpoint_config = ckpt["config"]
                    print(f"\n从checkpoint加载配置信息...")
                    # 检查通道数是否匹配
                    if "total_channels" in checkpoint_config:
                        checkpoint_channels = checkpoint_config["total_channels"]
                        if checkpoint_channels != config["total_channels"]:
                            print(f"⚠ 警告: checkpoint中的total_channels ({checkpoint_channels}) "
                                  f"与实际数据通道数 ({config['total_channels']}) 不匹配")
                            print(f"   将使用实际数据通道数: {config['total_channels']}")
                        else:
                            print(f"✓ 通道数匹配: {config['total_channels']}")

                    # 加载其他配置（不覆盖通道数）
                    if "seq_len_1d" in checkpoint_config:
                        config["seq_len_1d"] = checkpoint_config["seq_len_1d"]
                        print(f"✓ seq_len_1d: {config['seq_len_1d']}")
                    if "seq_len_2d" in checkpoint_config:
                        config["seq_len_2d"] = checkpoint_config["seq_len_2d"]
                        print(f"✓ seq_len_2d: {config['seq_len_2d']}")
                    if "use_age" in checkpoint_config:
                        config["use_age"] = checkpoint_config["use_age"]
                        print(f"✓ use_age: {config['use_age']}")
            except Exception as e:
                print(f"⚠ 无法从checkpoint加载配置: {e}")

        # 设置默认值（如果checkpoint中没有）
        if "seq_len_1d" not in config:
            config["seq_len_1d"] = config.get("seq_length", 2000)
        if "seq_len_2d" not in config:
            # seq_len_2d 应该由 build_dataset_for_train 设置，如果没有则使用默认值
            config["seq_len_2d"] = config.get("seq_len_2d", 64)  # 默认值，实际应该从数据中获取
        if "use_age" not in config:
            config["use_age"] = config.get("use_age", "fc")

        print(f"\n最终配置:")
        print(f"  - total_channels: {config['total_channels']}")
        print(f"  - in_channels: {config.get('in_channels', 'N/A')}")
        print(f"  - seq_len_1d: {config['seq_len_1d']}")
        print(f"  - seq_len_2d: {config.get('seq_len_2d', 'N/A')}")
        print(f"  - out_dims: {config.get('out_dims', 'N/A')}")
        print(f"  - use_age: {config.get('use_age', 'N/A')}")

        # ===== 2. 构建模型（使用实际数据的通道数） =====
        from models.vgg_fusion import VGG1D2DFusion

        model = VGG1D2DFusion(
        model="****",
        total_channels=config["total_channels"],  # 使用实际数据的通道数
        out_dims=config["out_dims"],
        seq_len_1d=config["seq_len_1d"],
        seq_len_2d=config["seq_len_2d"],
        use_age=config.get("use_age", "fc"),
        ).to(device)

        # ===== 3. 加载预训练权重 =====
        model = load_pretrained_weights(model, pretrain_path, device, strict=False)
        history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "lr": []}

        # ===== 4. 可选微调 =====
        if not skip_finetune and finetune_epochs > 0:
            print("\n" + "=" * 80)
        print("\n" + "=" * 80)
        print(f"开始微调，共 {finetune_epochs} 个epoch")
        print("=" * 80)


        backbone_params = []
        fc_params = []
        first_layer_params = []

        for name, param in model.named_parameters():
            if 'fc_stage' in name or 'classifier' in name.lower():
                fc_params.append(param)
            elif 'conv_stage1' in name:  # 第一层（随机初始化，需要更多学习）
                first_layer_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': finetune_lr * 0.3, 'weight_decay': 0.01},  # Backbone小学习率
            {'params': first_layer_params, 'lr': finetune_lr * 1.5, 'weight_decay': 0.01},  # 第一层大学习率
            {'params': fc_params, 'lr': finetune_lr, 'weight_decay': 0.01}  # FC层正常学习率
        ])

        print(f"分层学习率设置:")
        print(f"  Backbone: {finetune_lr * 0.3:.2e}")
        print(f"  第一层: {finetune_lr * 1.5:.2e}")
        print(f"  FC层: {finetune_lr:.2e}")

        # 学习率调度器（改进版：不衰减到0，保持最小学习率）
        from torch.optim.lr_scheduler import CosineAnnealingLR
        # 设置eta_min，让学习率不要衰减到0，保持一个最小学习率继续学习
        min_lr = finetune_lr * 0.1  # 最小学习率是初始学习率的10%
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=min_lr)

        # 混合精度
        scaler = GradScaler()

        # 计算类别权重（处理类别不平衡）
        num_classes = config["out_dims"]
        class_label_to_name = config.get("class_label_to_name", None)
        class_weights = calculate_class_weights(train_loader, num_classes, device, class_label_to_name)

        with torch.no_grad():
            if num_classes == 4:  # Normal, SCD, MCI, Dementia
                class_weights[0] = class_weights[0] * 1.0
                class_weights[1] = class_weights[1] * 3.0
                class_weights[2] = class_weights[2] * 0.3
                class_weights[3] = class_weights[3] * 2.0
                class_weights = class_weights / class_weights.sum() * num_classes
                print(f"\n调整后的类别权重（四分类: Normal, SCD, MCI, Dementia）:")
                for i in range(num_classes):
                    class_name = class_label_to_name[i] if class_label_to_name and i < len(
                        class_label_to_name) else f"Class {i}"
                    print(f"  {class_name}: {class_weights[i].item():.4f}")

        # 微调循环（带早停）
        patience = 200  # 增加耐心值：20个epoch没有提升才停止
        patience_counter = 0


        use_focal_loss = False  # 改为False，使用稳定的加权交叉熵
        print(
            f"\n使用{'Focal Loss (gamma=1.5)' if use_focal_loss else '加权交叉熵'} + 类别权重 + 早停机制 (patience={patience})")
        print("=" * 80 + "\n")
        for epoch in range(finetune_epochs):
            loss, train_acc = finetune_epoch(
                model, train_loader, optimizer, scheduler, device,
                config["preprocess_train"], scaler=scaler,
                class_weights=class_weights, use_focal=use_focal_loss, gamma=1.5
            )

            # epoch结束更新学习率
            scheduler.step()

            # ===== 每一轮都算val_acc（保证曲线每轮都有点）=====
            val_acc = check_accuracy(model, val_loader, config["preprocess_test"], config, repeat=1)

            # 打印（只保留关键KPI）
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch + 1}/{finetune_epochs}] "
                  f"Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {cur_lr:.2e}")

            # Early stopping / best model（仍用val_acc）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), "local/checkpoints/finetuned_best.pth")
                print(f"  -> 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发！最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
                    print(f"已训练 {epoch + 1} 个epoch，加载最佳模型继续评估...")
                    model.load_state_dict(torch.load("local/checkpoints/finetuned_best.pth"))
                    break

            # ===== 记录history（每轮都有val_acc）=====
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(float(loss))
            history["train_acc"].append(float(train_acc))
            history["val_acc"].append(float(val_acc))
            history["lr"].append(float(cur_lr))

        # 确保加载最佳模型进行评估
        if os.path.exists("local/checkpoints/finetuned_best.pth"):
            print("加载最佳模型进行评估...")
            model.load_state_dict(torch.load("local/checkpoints/finetuned_best.pth"))
        else:
            print("\n跳过微调，直接评估...")

        # ===== 5. 评估模型 =====
        results = evaluate_model(
            model, val_loader, test_loader, multicrop_test_loader,
            config["preprocess_test"], config, device,
            output_dir=run_viz_dir,

            run_test_eval=not skip_test_eval,
            run_multicrop_test_eval=not skip_multicrop_test_eval,
        )
        
        # 保存本次运行的checkpoint（统一采用 run_{idx}_seed_{seed} 命名）
        run_ckpt_path = f"local/checkpoints/finetuned_run_{run_idx + 1}_seed_{current_seed}.pt"
        
        run_checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "results": results,
            "run_idx": run_idx,
            "random_seed": current_seed,
        }
        torch.save(run_checkpoint, run_ckpt_path)
        print(f"✓ 已保存本次运行的checkpoint: {run_ckpt_path}")
        
        # 收集本次运行的结果
        run_result = {
            "run_idx": run_idx + 1,
            "random_seed": current_seed,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "val_acc": results["val_acc"],
            "test_acc": results["test_acc"],
            "multicrop_test_acc": results["multicrop_test_acc"],
            "test_confusion_matrix": results["test_confusion_matrix"],
            "multicrop_test_confusion_matrix": results["multicrop_test_confusion_matrix"],
        }
        all_run_results.append(run_result)
        
        print(f"\n运行 {run_idx + 1} 完成:")
        print(f"  验证集准确率: {results['val_acc']:.2f}%")
        print(f"  测试集准确率: {results['test_acc']:.2f}%")
        print(f"  多裁剪测试集准确率: {results['multicrop_test_acc']:.2f}%")
    
    # ===== 训练循环结束 =====
    # ===== 6. 汇总所有运行的结果（统一按多次 Monte Carlo 视角处理，即使只运行一次） =====
    print("\n" + "=" * 80)
    print("多次运行结果汇总")
    print("=" * 80)
    
    # 计算统计量
    val_accs = [r["val_acc"] for r in all_run_results]
    test_accs = [r["test_acc"] for r in all_run_results]
    multicrop_test_accs = [r["multicrop_test_acc"] for r in all_run_results]
    
    print(f"\n验证集准确率:")
    print(f"  均值: {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")
    print(f"  范围: [{np.min(val_accs):.2f}%, {np.max(val_accs):.2f}%]")
    
    print(f"\n测试集准确率:")
    print(f"  均值: {np.mean(test_accs):.2f}% ± {np.std(test_accs):.2f}%")
    print(f"  范围: [{np.min(test_accs):.2f}%, {np.max(test_accs):.2f}%]")
    
    print(f"\n多裁剪测试集准确率:")
    print(f"  均值: {np.mean(multicrop_test_accs):.2f}% ± {np.std(multicrop_test_accs):.2f}%")
    print(f"  范围: [{np.min(multicrop_test_accs):.2f}%, {np.max(multicrop_test_accs):.2f}%]")
    print("=" * 80)
    return all_run_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="加载预训练权重并进行微调和评估")
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help="预训练权重路径（best_checkpoint.pt或vgg_fusion_mae_best.pth）。如果不指定，默认使用 local/checkpoint/vgg19_dmae/checkpoint.pt")
    parser.add_argument("--dataset_path", type=str, default="datasets/caueeg-dataset",
                        help="数据集路径")
    parser.add_argument("--file_format", type=str, default=None, choices=["edf", "feather", None],
                        help="文件格式（edf或feather）。如果不指定，将自动检测")
    parser.add_argument("--task", type=str, default="dementia", choices=["dementia", "abnormal"],
                        help="任务类型")
    parser.add_argument("--finetune_epochs", type=int, default=10,
                        help="微调epoch数（0表示不微调）")
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                        help="微调学习率")
    parser.add_argument("--finetune_batch_size", type=int, default=32,
                        help="微调批次大小")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="跳过微调，直接评估")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--n_monte_carlo_runs", type=int, default=1,
                        help="运行次数（默认1，使用随机种子42）")
    parser.add_argument("--random_seeds", type=int, nargs="+", default=None,
                        help="随机种子列表（如果不指定，自动生成：42, 142, 242, ...）")
    parser.add_argument(

        type=str,
        default="test",
        choices=["val", "test", "multicrop_test"],

    )
    parser.add_argument(
        "--skip_test_eval",
        action="store_true",

    )
    parser.add_argument(
        "--skip_multicrop_test_eval",
        action="store_true",

    )

    args = parser.parse_args()


    if args.pretrain_path is None:
        default_path = "local/checkpoint/vgg19_dmae/checkpoint.pt"
        if os.path.exists(default_path):
            args.pretrain_path = default_path
            print(f"使用默认预训练权重路径: {default_path}")
        else:
            raise FileNotFoundError(
                f"未找到默认预训练权重: {default_path}\n"
                f"请使用 --pretrain_path 指定预训练权重路径，或先运行预训练脚本。"
            )





    main_finetune_and_evaluate(
        pretrain_path=args.pretrain_path,
        dataset_path=args.dataset_path,
        task=args.task,
        file_format=args.file_format,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_batch_size=args.finetune_batch_size,
        device=args.device,
        skip_finetune=args.skip_finetune,
        n_monte_carlo_runs=args.n_monte_carlo_runs,
        random_seeds=args.random_seeds,

        skip_test_eval=args.skip_test_eval,
        skip_multicrop_test_eval=args.skip_multicrop_test_eval,
    )

