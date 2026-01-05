# -------------------- 隐藏 --------------------
# import numpy as np
# -------------------- 隐藏 --------------------
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

    # -------------------- 隐藏：embedding 计算细节 --------------------
    # ..................................................
    # output = model.compute_feature_embedding(...)
    # ..................................................
    return None


@torch.no_grad()
def estimate_score(model, sample_batched, preprocess, config):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    signals = sample_batched["signal"]
    age = sample_batched["age"]

    # -------------------- 隐藏：模型前向细节 --------------------
    # ..................................................
    # output = model(signals, age)
    # ..................................................
    output = None

    # -------------------- 隐藏：criterion -> score 映射细节 --------------------
    # ..................................................
    # score = softmax/sigmoid/raw
    # ..................................................
    return None


def apply_mci_fallback_strategy(score, config, confidence_threshold=0.4):
    """
    应用 MCI 回退策略（预测规则已隐藏，仅保留函数边界）
    """
    # -------------------- 隐藏：类索引推断与阈值逻辑 --------------------
    # ..................................................
    # ..................................................
    return None


def calculate_confusion_matrix(pred, target, num_classes):
    # -------------------- 隐藏 --------------------
    # ..................................................
    return None


def calculate_class_wise_metrics(confusion_matrix):
    # -------------------- 隐藏：Accuracy/Sensitivity/Specificity/Precision/F1 --------------------
    # ..................................................
    return {}


@torch.no_grad()
def check_accuracy(model, loader, preprocess, config, repeat=1):
    # -------------------- 隐藏：循环、score、fallback/argmax、统计 --------------------
    # ..................................................
    return 0.0


@torch.no_grad()
def check_accuracy_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # -------------------- 隐藏：confusion/ROC缓存/throughput计时/warmup等 --------------------
    # ..................................................
    return 0.0, None, None, None, 0.0


@torch.no_grad()
def check_accuracy_multicrop(model, loader, preprocess, config, repeat=1):
    # -------------------- 隐藏：multi-crop merge 与 fallback/argmax --------------------
    # ..................................................
    return 0.0


@torch.no_grad()
def check_accuracy_multicrop_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # -------------------- 隐藏：multi-crop extended 指标与throughput --------------------
    # ..................................................
    return 0.0, None, None, None, 0.0
