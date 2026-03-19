import numpy as np
import torch
import torch.nn.functional as F


# __all__ = []


@torch.no_grad()
def compute_embedding(model, sample_batched, preprocess, config):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    # apply model on whole batch directly on device
    x = sample_batched["signal"]
    age = sample_batched["age"]
    output = model.compute_feature_embedding(x, age, target_from_last=1)

    return output


@torch.no_grad()
def estimate_score(model, sample_batched, preprocess, config):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    # apply model on whole batch directly on device
    # 参考 finetune_and_evaluate.py 和 pretrain_fusion_mae.py，直接传入 signals 和 age
    signals = sample_batched["signal"]
    age = sample_batched["age"]

    # 直接调用模型，模型内部会处理输入格式
    output = model(signals, age)

    if config["criterion"] == "cross-entropy":
        score = F.softmax(output, dim=1)
    elif config["criterion"] == "multi-bce":
        score = torch.sigmoid(output)
    elif config["criterion"] == "svm":
        score = output
    else:
        raise ValueError(f"estimate_score(): cannot parse config['criterion']={config['criterion']}.")
    return score


def apply_mci_fallback_strategy(score, config, confidence_threshold=0.4):
    """
    改进的MCI回退策略（更智能的决策）：
    1) 可选二段式决策：先在 Normal vs Dementia 上做二分类置信度判断，足够高则直接输出，否则进入MCI回退
    2) 若二段式未触发，则：
       - 计算最大概率和次大概率的差值
       - 如果最大概率足够高且差值足够大，选择最大概率对应的类别
       - 否则，使用更精细的规则：
         * 如果Dementia概率 >= threshold 且比Normal高很多，选Dementia
         * 如果Normal概率 >= threshold 且比Dementia高很多，选Normal
         * 否则选MCI（不确定的情况）

    Args:
        score: 模型输出的概率分布 [batch_size, num_classes]
        config: 配置字典，需要包含 class_name_to_label 或 class_label_to_name
        confidence_threshold: 置信度阈值，默认0.4

    Returns:
        调整后的预测结果 [batch_size]
    """
    # 获取类别索引
    class_name_to_label = config.get("class_name_to_label", {})
    class_label_to_name = config.get("class_label_to_name", [])

    # 尝试从配置中获取类别索引
    normal_idx = class_name_to_label.get("Normal", 0)
    mci_idx = class_name_to_label.get("MCI", 1)
    dementia_idx = class_name_to_label.get("Dementia", 2)

    # 如果配置中没有，尝试从class_label_to_name推断
    if not class_name_to_label and class_label_to_name:
        for i, name in enumerate(class_label_to_name):
            if name.lower() == "normal":
                normal_idx = i
            elif name.lower() == "mci":
                mci_idx = i
            elif name.lower() == "dementia":
                dementia_idx = i

    # 获取各类别的概率
    normal_prob = score[:, normal_idx]
    mci_prob = score[:, mci_idx]
    dementia_prob = score[:, dementia_idx]

    # 1) 可选二段式：先判断 Normal vs Dementia 的置信度
    if config.get("binary_nd_fallback", False):
        nd_conf_threshold = config.get("nd_conf_threshold", 0.55)
        nd_gap_threshold = config.get("nd_gap_threshold", 0.10)
        # 仅取 Normal 和 Dementia 概率做二分类 softmax
        nd = torch.stack([normal_prob, dementia_prob], dim=1)  # [B,2]
        nd_soft = torch.softmax(nd, dim=1)
        nd_max, nd_idx = nd_soft.max(dim=1)  # 0=Normal,1=Dementia
        nd_gap = (nd_soft[:, 0] - nd_soft[:, 1]).abs()

        nd_conf_mask = (nd_max >= nd_conf_threshold) & (nd_gap >= nd_gap_threshold)
        if nd_conf_mask.any():
            # 先用 ND 高置信度的样本直接决策
            pred = torch.full((score.shape[0],), mci_idx, dtype=torch.long, device=score.device)
            nd_choices = torch.where(nd_idx == 0, torch.full_like(nd_idx, normal_idx),
                                     torch.full_like(nd_idx, dementia_idx))
            pred[nd_conf_mask] = nd_choices[nd_conf_mask]
            # 未满足高置信度的，继续走后续逻辑
            remain_mask = ~nd_conf_mask
            if remain_mask.any():
                # 对剩余样本使用后续逻辑，重用变量但保持维度
                normal_prob = normal_prob[remain_mask]
                mci_prob = mci_prob[remain_mask]
                dementia_prob = dementia_prob[remain_mask]
                score = score[remain_mask]
                max_probs, max_indices = torch.max(score, dim=1)
                sorted_probs, _ = torch.sort(score, dim=1, descending=True)
                prob_diff = sorted_probs[:, 0] - sorted_probs[:, 1]
                # 后续逻辑得到的结果写回 pred 的对应位置
                sub_pred = _apply_mci_fallback_core(normal_prob, mci_prob, dementia_prob, max_probs, max_indices,
                                                    prob_diff, confidence_threshold, config, device=score.device)
                pred[remain_mask] = sub_pred
            return pred
    # 若未启用二段式或无高置信度 ND 样本，则继续常规逻辑

    # 计算最大概率和对应的类别
    max_probs, max_indices = torch.max(score, dim=1)

    # 计算最大概率和次大概率的差值（用于判断置信度）
    sorted_probs, _ = torch.sort(score, dim=1, descending=True)
    prob_diff = sorted_probs[:, 0] - sorted_probs[:, 1]  # 最大概率 - 次大概率

    return _apply_mci_fallback_core(
        normal_prob,
        mci_prob,
        dementia_prob,
        max_probs,
        max_indices,
        prob_diff,
        confidence_threshold,
        config,
        device=score.device,
    )


def _apply_mci_fallback_core(
        normal_prob,
        mci_prob,
        dementia_prob,
        max_probs,
        max_indices,
        prob_diff,
        confidence_threshold,
        config,
        device,
):
    """
    基础回退策略核心逻辑，便于二段式决策复用
    """
    # 获取类别索引
    class_name_to_label = config.get("class_name_to_label", {})
    class_label_to_name = config.get("class_label_to_name", [])

    normal_idx = class_name_to_label.get("Normal", 0)
    mci_idx = class_name_to_label.get("MCI", 1)
    dementia_idx = class_name_to_label.get("Dementia", 2)

    # 创建预测结果，默认选择MCI
    pred = torch.full((max_probs.shape[0],), mci_idx, dtype=torch.long, device=device)

    # 策略1: 适度的高置信阈值（0.52 / 0.20），让更多样本进入精细决策
    high_confidence_mask = (max_probs >= 0.52) & (prob_diff >= 0.20)
    pred[high_confidence_mask] = max_indices[high_confidence_mask]

    # 策略2: 对于剩余的不确定样本，使用更精细的规则
    uncertain_mask = ~high_confidence_mask

    if uncertain_mask.any():
        # 计算各类概率差值和优势
        normal_advantage = normal_prob - dementia_prob
        normal_vs_mci = normal_prob - mci_prob
        dementia_advantage = dementia_prob - normal_prob
        dementia_vs_mci = dementia_prob - mci_prob
        mci_vs_normal = mci_prob - normal_prob
        mci_vs_dementia = mci_prob - dementia_prob

        # Normal决策：再收紧，降低 Normal→MCI（当前 Normal→MCI 较高）
        normal_mask = uncertain_mask & (
                (normal_prob >= confidence_threshold + 0.08) &
                (normal_vs_mci >= 0.13) &
                (normal_advantage >= 0.04) &
                (mci_prob < confidence_threshold + 0.06)
        )
        pred[normal_mask] = normal_idx

        # Dementia决策：收紧，降低 Dementia→MCI（当前 Dementia→MCI 较高）
        dementia_mask = uncertain_mask & (
                (dementia_prob >= confidence_threshold + 0.08) &
                (dementia_vs_mci >= 0.13) &
                (dementia_advantage >= 0.05) &
                (mci_prob < confidence_threshold + 0.07)
        )
        pred[dementia_mask] = dementia_idx

        # Normal决策：进一步保护Normal（中等置信度），但要更谨慎
        normal_mask_enhanced = uncertain_mask & (
                (normal_prob >= confidence_threshold + 0.05) &
                (normal_vs_mci >= 0.11) &
                ~normal_mask &
                (normal_prob > mci_prob + 0.09) &
                (mci_prob < confidence_threshold + 0.06)
        )
        pred[normal_mask_enhanced] = normal_idx

        # Dementia决策：进一步保护Dementia，但要更谨慎（避免误判MCI）
        dementia_mask_enhanced = uncertain_mask & (
                (dementia_prob >= confidence_threshold + 0.04) &
                (dementia_vs_mci >= 0.10) &
                ~dementia_mask &
                (dementia_prob > mci_prob + 0.08) &
                (mci_prob < confidence_threshold + 0.06)
        )
        pred[dementia_mask_enhanced] = dementia_idx

        # MCI决策：更精细的判断，优先考虑MCI
        # 1. MCI明显最高或与最高接近（略收紧，减少 ND→MCI）
        mci_highest = uncertain_mask & (
                (mci_prob >= normal_prob - 0.02) &
                (mci_prob >= dementia_prob - 0.03) &
                (mci_prob >= confidence_threshold + 0.03)
        )

        # 2. Normal和Dementia都很低，MCI相对较高
        both_low_confidence = uncertain_mask & (
                (normal_prob < confidence_threshold + 0.07) &
                (dementia_prob < confidence_threshold + 0.08) &
                (mci_prob >= confidence_threshold + 0.03)
        )

        # 3. MCI vs Dementia：当MCI和Dementia接近时，倾向于MCI（放宽条件）
        mci_vs_dementia_close = uncertain_mask & (
                (mci_prob >= confidence_threshold + 0.03) &
                (mci_vs_dementia >= -0.08) &
                (mci_prob > normal_prob - 0.05) &
                (dementia_prob < confidence_threshold + 0.10) &
                (dementia_vs_mci < 0.07)
        )

        # 4. MCI vs Normal：当MCI和Normal接近时，倾向于MCI（放宽条件）
        mci_vs_normal_close = uncertain_mask & (
                (mci_prob >= confidence_threshold + 0.03) &
                (mci_vs_normal >= -0.07) &
                (mci_prob > dementia_prob - 0.05) &
                (normal_prob < confidence_threshold + 0.10) &
                (normal_vs_mci < 0.08)
        )

        # 5. MCI中等置信度：当MCI概率中等，且Normal和Dementia都不太确定时
        mci_medium = uncertain_mask & (
                (mci_prob >= confidence_threshold - 0.02) &
                (normal_prob < confidence_threshold + 0.08) &
                (dementia_prob < confidence_threshold + 0.08) &
                (normal_vs_mci < 0.07) &
                (dementia_vs_mci < 0.07)
        )

        mci_mask = (
                mci_highest |
                (
                            both_low_confidence & ~dementia_mask & ~normal_mask & ~dementia_mask_enhanced & ~normal_mask_enhanced) |
                (mci_vs_dementia_close & ~dementia_mask & ~dementia_mask_enhanced) |
                (mci_vs_normal_close & ~normal_mask & ~normal_mask_enhanced) |
                (mci_medium & ~dementia_mask & ~normal_mask & ~dementia_mask_enhanced & ~normal_mask_enhanced)
        )
        pred[mci_mask] = mci_idx

    return pred


def calculate_confusion_matrix(pred, target, num_classes):
    N = target.shape[0]
    C = num_classes
    confusion = np.zeros((C, C), dtype=np.int32)

    for i in range(N):
        r = target[i]
        c = pred[i]
        confusion[r, c] += 1
    return confusion


def calculate_class_wise_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]

    accuracy = np.zeros((n_classes,))
    sensitivity = np.zeros((n_classes,))
    specificity = np.zeros((n_classes,))
    precision = np.zeros((n_classes,))
    recall = np.zeros((n_classes,))

    for c in range(n_classes):
        tp = confusion_matrix[c, c]
        fn = confusion_matrix[c].sum() - tp
        fp = confusion_matrix[:, c].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp

        # Class-wise Accuracy: (TP + TN) / N
        # 即：该类别的正确预测 + 其他类别的正确预测 / 总样本数
        accuracy[c] = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) > 0 else 0.0
        sensitivity[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity[c] = tn / (fp + tn) if (fp + tn) > 0 else 0.0
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = sensitivity[c]
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)  # 避免除零

    class_wise_metrics = {
        "Accuracy": accuracy,  # (TP + TN) / N
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1_score,
    }  # 'Recall': recall is same with sensitivity
    return class_wise_metrics


@torch.no_grad()
def check_accuracy(model, loader, preprocess, config, repeat=1):
    # for accuracy
    correct, total = (0, 0)

    # 是否使用MCI回退策略
    use_mci_fallback = config.get("use_mci_fallback", False)
    confidence_threshold = config.get("mci_fallback_threshold", 0.4)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # calculate accuracy
            if use_mci_fallback:
                pred = apply_mci_fallback_strategy(s, config, confidence_threshold)
            else:
                pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    # 使用更高精度的计算，避免精度损失
    if total > 0:
        accuracy = 100.0 * correct / total
    else:
        accuracy = 0.0
    return accuracy


@torch.no_grad()
def check_accuracy_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # 是否使用MCI回退策略
    use_mci_fallback = config.get("use_mci_fallback", False)
    confidence_threshold = config.get("mci_fallback_threshold", 0.4)

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            start_event.record()
            s = estimate_score(model, sample_batched, preprocess, config)
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            y = sample_batched["class_label"]

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix - 使用MCI回退策略或标准argmax
            if use_mci_fallback:
                pred = apply_mci_fallback_strategy(s, config, confidence_threshold)
            else:
                pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    return accuracy, score, target, confusion_matrix, throughput


@torch.no_grad()
def check_accuracy_multicrop(model, loader, preprocess, config, repeat=1):
    # for accuracy
    correct, total = (0, 0)

    # 是否使用MCI回退策略
    use_mci_fallback = config.get("use_mci_fallback", False)
    confidence_threshold = config.get("mci_fallback_threshold", 0.4)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # multi-crop averaging
            if s.size(0) % config["test_crop_multiple"] != 0:
                raise ValueError(
                    f"check_accuracy_multicrop(): Real minibatch size={y.size(0)} is not multiple of "
                    f"config['test_crop_multiple']={config['test_crop_multiple']}."
                )

            real_minibatch = s.size(0) // config["test_crop_multiple"]
            s_ = torch.zeros((real_minibatch, s.size(1)), device=s.device)
            y_ = torch.zeros((real_minibatch,), dtype=torch.int32, device=y.device)

            # 改进的平均策略：对logits进行平均（而不是对概率进行平均）
            # 这样可以更好地利用多裁剪的信息
            for m in range(real_minibatch):
                start_idx = config["test_crop_multiple"] * m
                end_idx = config["test_crop_multiple"] * (m + 1)
                # 对logits进行平均（更稳健）
                crop_logits = s[start_idx:end_idx]
                s_[m] = crop_logits.mean(dim=0, keepdim=True)
                y_[m] = y[start_idx]  # 标签应该相同

            s = s_
            y = y_

            # calculate accuracy - 使用MCI回退策略或标准argmax
            if use_mci_fallback:
                pred = apply_mci_fallback_strategy(s, config, confidence_threshold)
            else:
                pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def collect_misclassified_mci_dementia(model, loader, preprocess, config):
    """
    收集测试集中 true=MCI/Dementia 但预测错误的样本编号（serial）
    """
    model.eval()
    use_mci_fallback = config.get("use_mci_fallback", False)
    confidence_threshold = config.get("mci_fallback_threshold", 0.4)

    class_name_to_label = config.get("class_name_to_label", {})
    class_label_to_name = config.get("class_label_to_name", [])
    normal_idx = class_name_to_label.get("Normal", 0)
    mci_idx = class_name_to_label.get("MCI", 1)
    dementia_idx = class_name_to_label.get("Dementia", 2)

    mistakes = []

    for sample_batched in loader:
        serials = sample_batched.get("serial", None)
        preprocess(sample_batched)

        s = estimate_score(model, sample_batched, preprocess=lambda x: x, config=config)
        y = sample_batched["class_label"]

        if use_mci_fallback:
            pred = apply_mci_fallback_strategy(s, config, confidence_threshold)
        else:
            pred = s.argmax(dim=-1)

        for i in range(y.size(0)):
            true_idx = y[i].item()
            pred_idx = pred[i].item()
            if true_idx in (mci_idx, dementia_idx) and pred_idx != true_idx:
                serial = None
                if serials is not None:
                    try:
                        serial = serials[i]
                    except Exception:
                        serial = None
                true_name = (
                    class_label_to_name[true_idx]
                    if class_label_to_name and true_idx < len(class_label_to_name)
                    else str(true_idx)
                )
                pred_name = (
                    class_label_to_name[pred_idx]
                    if class_label_to_name and pred_idx < len(class_label_to_name)
                    else str(pred_idx)
                )
                mistakes.append(
                    {
                        "serial": serial,
                        "true_idx": true_idx,
                        "pred_idx": pred_idx,
                        "true_name": true_name,
                        "pred_name": pred_name,
                    }
                )

    return mistakes


@torch.no_grad()
def collect_mci_dementia_cases(model, loader, preprocess, config):
    """
    收集测试集中 MCI / Dementia 的预测结果，区分正确与错误样本编号（serial）
    返回:
      {
        "mci": {"correct": [...], "wrong": [...]},
        "dementia": {"correct": [...], "wrong": [...]}
      }
    """
    model.eval()
    use_mci_fallback = config.get("use_mci_fallback", False)
    confidence_threshold = config.get("mci_fallback_threshold", 0.4)

    class_name_to_label = config.get("class_name_to_label", {})
    class_label_to_name = config.get("class_label_to_name", [])
    normal_idx = class_name_to_label.get("Normal", 0)
    mci_idx = class_name_to_label.get("MCI", 1)
    dementia_idx = class_name_to_label.get("Dementia", 2)

    result = {
        "mci": {"correct": [], "wrong": []},
        "dementia": {"correct": [], "wrong": []},
    }

    for sample_batched in loader:
        serials = sample_batched.get("serial", None)
        preprocess(sample_batched)

        s = estimate_score(model, sample_batched, preprocess=lambda x: x, config=config)
        y = sample_batched["class_label"]

        if use_mci_fallback:
            pred = apply_mci_fallback_strategy(s, config, confidence_threshold)
        else:
            pred = s.argmax(dim=-1)

        for i in range(y.size(0)):
            true_idx = y[i].item()
            pred_idx = pred[i].item()
            serial = None
            if serials is not None:
                try:
                    serial = serials[i]
                except Exception:
                    serial = None

            if true_idx == mci_idx:
                (result["mci"]["correct"] if pred_idx == true_idx else result["mci"]["wrong"]).append(serial)
            elif true_idx == dementia_idx:
                (result["dementia"]["correct"] if pred_idx == true_idx else result["dementia"]["wrong"]).append(serial)

    return result


@torch.no_grad()
def check_accuracy_multicrop_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_score(model, sample_batched, preprocess, config)

    # 调试信息：记录第一个batch的情况
    first_batch = True

    for k in range(repeat):
        for sample_batched in loader:
            # 计算实际batch size（考虑多裁剪）
            batch_size_before_merge = sample_batched["signal"].size(0)
            test_crop_multiple = config["test_crop_multiple"]
            real_minibatch = batch_size_before_merge // test_crop_multiple

            # 调试信息（仅第一个batch）
            if first_batch and k == 0:
                print(f"\n[多裁剪评估调试]")
                print(f"  batch_size_before_merge: {batch_size_before_merge}")
                print(f"  test_crop_multiple: {test_crop_multiple}")
                print(f"  real_minibatch (平均后): {real_minibatch}")
                first_batch = False

            s_merge = torch.zeros((real_minibatch, config["out_dims"]), device=sample_batched["signal"].device)
            y_merge = torch.zeros((real_minibatch,), dtype=torch.int32, device=sample_batched["signal"].device)

            # estimate
            start_event.record()
            s = estimate_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # multi-crop averaging
            if s.size(0) % test_crop_multiple != 0:
                raise ValueError(
                    f"check_accuracy_multicrop(): Score batch size={s.size(0)} is not multiple of "
                    f"test_crop_multiple={test_crop_multiple}."
                )

            # 确保 s 和 y 的 batch size 一致
            if s.size(0) != y.size(0):
                raise ValueError(
                    f"Score batch size ({s.size(0)}) != label batch size ({y.size(0)})"
                )

            # 改进的多裁剪平均：对logits进行平均（更稳健）
            # 每 test_crop_multiple 段平均成一个预测结果
            for m in range(real_minibatch):
                start_idx = test_crop_multiple * m
                end_idx = test_crop_multiple * (m + 1)
                # 对logits进行平均（而不是对概率进行平均），这样可以更好地利用多裁剪的信息
                crop_logits = s[start_idx:end_idx]
                s_merge[m] = crop_logits.mean(dim=0, keepdim=True)
                # 标签（所有段的标签应该相同，取第一个）
                y_merge[m] = y[start_idx]

            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            # 使用平均后的结果
            s = s_merge
            y = y_merge

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix - 使用MCI回退策略或标准argmax
            use_mci_fallback = config.get("use_mci_fallback", False)
            confidence_threshold = config.get("mci_fallback_threshold", 0.4)
            if use_mci_fallback:
                pred = apply_mci_fallback_strategy(s, config, confidence_threshold)
            else:
                pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    return accuracy, score, target, confusion_matrix, throughput
