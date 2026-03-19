import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

# from .utils import TimeElapsed

# __all__ = []
def train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps, class_weights=None):
    import time
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)
    t0 = time.perf_counter()
    last_t = t0

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()
            preprocess(sample_batched)

            # 直接使用4D图像数据
            x = sample_batched["signal"]  # [64, 40, 64, 63]
            age = sample_batched["age"]
            y = sample_batched["class_label"]

            # 分割通道作为1D和2D数据
            if hasattr(model, 'in_channels_1d') and hasattr(model, 'in_channels_2d'):
                # 假设前N个通道是1D数据，后M个通道是2D数据
                x_1d = x[:, :model.in_channels_1d, :, :]  # 使用空间维度作为1D序列
                x_2d = x[:, model.in_channels_1d:, :, :]  # 剩余的作为2D图像

                # 将1D数据展平为 [batch, channels, seq_len]
                x_1d = x_1d.reshape(x_1d.shape[0], x_1d.shape[1], -1)  # [64, ch1, 64*63]


                model_input = {"1d": x_1d, "2d": x_2d}
            else:
                model_input = x


                # mixed precision training if needed
            with autocast(enabled=config.get("mixed_precision", False)):
                # forward pass
                output = model(model_input, age)

                # loss function (支持类别权重)
                if config["criterion"] == "cross-entropy":
                    # 统一使用 cross_entropy（内部包含 log_softmax + nll_loss）
                    # 支持 label smoothing：对四分类更稳，尤其在样本较少/噪声较多时
                    label_smoothing = float(config.get("label_smoothing", 0.0) or 0.0)
                    if class_weights is not None and isinstance(class_weights, torch.Tensor):
                        class_weights = class_weights.to(y.device)
                    loss = F.cross_entropy(
                        output,
                        y,
                        weight=class_weights if class_weights is not None else None,
                        label_smoothing=label_smoothing,
                    )
                    s = output
                elif config["criterion"] == "multi-bce":
                    y_oh = F.one_hot(y, num_classes=output.size(dim=1))
                    s = torch.sigmoid(output)
                    loss = F.binary_cross_entropy_with_logits(output, y_oh.float())
                elif config["criterion"] == "svm":
                    s = output
                    loss = F.multi_margin_loss(output, y)
                else:
                    raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce', 'svm']")



            # backward and update
            if config.get("mixed_precision", False):
                amp_scaler.scale(loss).backward()
                if "clip_grad_norm" in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if "clip_grad_norm" in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            inner_log = int(config.get("inner_log_interval", 0) or 0)
            if inner_log > 0 and (i % inner_log == 0 or i == steps):
                now = time.perf_counter()
                avg_loss_so_far = cumu_loss / max(1, i)
                acc_so_far = 100.0 * correct / max(1, total)
                it_s = (now - last_t) / max(1, inner_log)
                last_t = now
                print(
                    f"[train_step] {i:>5}/{steps} "
                    f"loss={avg_loss_so_far:.4f} acc={acc_so_far:.2f}% "
                    f"{it_s:.3f}s/it",
                    flush=True,
                )
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps
    _ = time.perf_counter() - t0

    return avg_loss, train_acc


def train_mixup_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps, class_weights=None):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # load and mixup the mini-batched data
            x1 = sample_batched["signal"]
            age1 = sample_batched["age"]
            y1 = sample_batched["class_label"]

            index = torch.randperm(x1.shape[0]).cuda()
            x2 = x1[index]
            age2 = age1[index]
            y2 = y1[index]

            mixup_alpha = config["mixup"]
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            x = lam * x1 + (1.0 - lam) * x2
            age = lam * age1 + (1.0 - lam) * age2

            # 准备模型输入
            if hasattr(model, 'in_channels_1d') and hasattr(model, 'in_channels_2d'):
                seq_len_1d = config.get("seq_len_1d", 2000)
                img_size = config.get("img_size", 32)
                img_channels = model.in_channels_2d

                # 分割1D数据
                x_1d = x[:, :model.in_channels_1d, :seq_len_1d]

                # 分割并reshape 2D数据
                expected_2d_elements = img_channels * img_size * img_size
                x_2d_flat = x[:, :img_channels, seq_len_1d:seq_len_1d + expected_2d_elements]
                x_2d = x_2d_flat.reshape(-1, img_channels, img_size, img_size)

                model_input = {"1d": x_1d, "2d": x_2d}
            else:
                model_input = x


            with autocast(enabled=config.get("mixed_precision", False)):
                # forward pass
                output = model(model_input, age)  # 修改这里


                # loss function (支持类别权重)
                if config["criterion"] == "cross-entropy":
                    label_smoothing = float(config.get("label_smoothing", 0.0) or 0.0)
                    if class_weights is not None and isinstance(class_weights, torch.Tensor):
                        class_weights = class_weights.to(y1.device)
                    loss1 = F.cross_entropy(
                        output,
                        y1,
                        weight=class_weights if class_weights is not None else None,
                        label_smoothing=label_smoothing,
                    )
                    loss2 = F.cross_entropy(
                        output,
                        y2,
                        weight=class_weights if class_weights is not None else None,
                        label_smoothing=label_smoothing,
                    )
                    loss = lam * loss1 + (1 - lam) * loss2
                    s = output
                elif config["criterion"] == "multi-bce":
                    y1_oh = F.one_hot(y1, num_classes=output.size(dim=1))
                    y2_oh = F.one_hot(y2, num_classes=output.size(dim=1))
                    y_oh = lam * y1_oh + (1.0 - lam) * y2_oh
                    s = torch.sigmoid(output)
                    loss = F.binary_cross_entropy_with_logits(output, y_oh)
                elif config["criterion"] == "svm":
                    s = output
                    loss1 = F.multi_margin_loss(output, y1)
                    loss2 = F.multi_margin_loss(output, y2)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce', 'svm']")

            # backward and update
            if config.get("mixed_precision", False):
                amp_scaler.scale(loss).backward()
                if "clip_grad_norm" in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if "clip_grad_norm" in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = s.argmax(dim=-1)
            correct1 = pred.squeeze().eq(y1).sum().item()
            correct2 = pred.squeeze().eq(y2).sum().item()
            correct += lam * correct1 + (1.0 - lam) * correct2
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc
