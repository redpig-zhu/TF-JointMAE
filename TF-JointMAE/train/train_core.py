# import numpy as np   # -------------------- 隐藏 --------------------
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


def train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps, class_weights=None):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()
            preprocess(sample_batched)

            # 直接使用4D图像数据
            x = sample_batched["signal"]
            age = sample_batched["age"]
            y = sample_batched["class_label"]

            # 分割通道作为1D和2D数据
            if hasattr(model, "in_channels_1d") and hasattr(model, "in_channels_2d"):
                # -----------------------------------------------
                # | 1D/2D 输入拆分与 reshape 细节（隐藏）          |
                # -----------------------------------------------
                # .................................................
                # model_input = {"1d": x_1d, "2d": x_2d}
                pass
            else:
                model_input = x

            with autocast(enabled=config.get("mixed_precision", False)):
                # forward pass
                output = model(model_input, age)

                # -----------------------------------------------
                # | loss 计算细节：criterion/权重/one-hot 等（隐藏）|
                # -----------------------------------------------
                # .................................................
                # s = ...
                # loss = ...
                pass

            # -----------------------------------------------
            # | backward + step + scheduler + grad clip（隐藏）|
            # -----------------------------------------------
            # .................................................
            # .................................................
            pass

            # -----------------------------------------------
            # | accuracy / loss 累积统计（隐藏）               |
            # -----------------------------------------------
            # .................................................
            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    # -----------------------------------------------
    # | 计算 avg_loss / train_acc（隐藏）             |
    # -----------------------------------------------
    # .................................................
    return 0.0, 0.0


def train_mixup_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps, class_weights=None):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()
            preprocess(sample_batched)

            # -----------------------------------------------
            # | mixup：permute / beta采样 / 混合输入与标签（隐藏）|
            # -----------------------------------------------
            # .................................................
            # x, age, y1, y2, lam = ...
            pass

            # 准备模型输入
            if hasattr(model, "in_channels_1d") and hasattr(model, "in_channels_2d"):
                # -----------------------------------------------
                # | mixup 下 1D/2D 输入切分与 reshape 细节（隐藏）  |
                # -----------------------------------------------
                # .................................................
                # model_input = {"1d": x_1d, "2d": x_2d}
                pass
            else:
                model_input = x

            with autocast(enabled=config.get("mixed_precision", False)):
                output = model(model_input, age)

                # -----------------------------------------------
                # | mixup loss：两份标签加权/类别权重等（隐藏）     |
                # -----------------------------------------------
                # .................................................
                # s = ...
                # loss = ...
                pass

            # -----------------------------------------------
            # | backward + step + scheduler + grad clip（隐藏）|
            # -----------------------------------------------
            # .................................................
            pass

            # -----------------------------------------------
            # | mixup accuracy 统计（隐藏）                    |
            # -----------------------------------------------
            # .................................................
            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    # -----------------------------------------------
    # | 计算 avg_loss / train_acc（隐藏）             |
    # -----------------------------------------------
    # .................................................
    return 0.0, 0.0
