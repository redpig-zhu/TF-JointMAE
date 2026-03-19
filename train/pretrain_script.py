import os
import pprint
from copy import deepcopy

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

from optim import get_lr_scheduler


def _prepare_model_input(sample_batched):
    return sample_batched["signal"]


def _mae_multistep(
    model,
    mae_head,
    loader,
    preprocess,
    optimizer,
    scheduler,
    scaler,
    config,
    steps,
):
    model.train()
    mae_head.train()

    i = 0
    cum_loss = 0.0
    data_iter = iter(loader)

    while i < steps:
        try:
            sample_batched = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            sample_batched = next(data_iter)

        optimizer.zero_grad()
        preprocess(sample_batched)

        x = _prepare_model_input(sample_batched)
        age = sample_batched["age"]
        sample_ids = sample_batched.get("serial", None)

        with autocast(enabled=config.get("mixed_precision", False)):
            embeddings = model.forward_embeddings(x, age, sample_ids=sample_ids, apply_dropout=True)
            loss, _, _ = mae_head(embeddings)

        if scaler is not None:
            scaler.scale(loss).backward()
            if "clip_grad_norm" in config:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if "clip_grad_norm" in config:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            optimizer.step()

        scheduler.step()
        cum_loss += loss.item()
        i += 1

    return cum_loss / steps


@torch.no_grad()
def _mae_eval(model, mae_head, loader, preprocess):
    model.eval()
    mae_head.eval()

    losses = []
    for sample_batched in loader:
        preprocess(sample_batched)
        x = _prepare_model_input(sample_batched)
        age = sample_batched["age"]
        sample_ids = sample_batched.get("serial", None)
        embeddings = model.forward_embeddings(x, age, sample_ids=sample_ids, apply_dropout=False)
        loss, _, _ = mae_head(embeddings)
        losses.append(loss.item())

    return sum(losses) / max(len(losses), 1)


def pretrain_script(
    config,
    model,
    mae_head,
    train_loader,
    val_loader,
    preprocess_train,
    preprocess_val,
):
    main_process = True  # 当前实现仅支持单GPU

    if main_process:
        print(f"\n{'*'*30} {'Configurations for Pretrain':^30} {'*'*30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*'*92}\n")

    if config.get("search_lr", False):
        raise ValueError("Learning rate search is not supported for pretraining.")

    config["iterations"] = round(config["total_samples"] / config["minibatch"])
    config["warmup_steps"] = max(round(config["iterations"] * config["warmup_ratio"]), config["warmup_min"])
    history_interval = max(config["iterations"] // config["num_history"], 1)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(mae_head.parameters()),
        lr=config["base_lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = get_lr_scheduler(
        optimizer,
        config["lr_scheduler_type"],
        iterations=config["iterations"],
        warmup_steps=config["warmup_steps"],
    )
    scaler = GradScaler() if config.get("mixed_precision", False) else None

    if config["save_model"]:
        save_path = os.path.join(config.get("cwd", ""), f"local/checkpoint/{config.get('pretrain_run_name', 'mae')}/")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = None

    best_val = float("inf")
    best_state = None
    i_step = 0

    while i_step < config["iterations"]:
        steps = min(history_interval, config["iterations"] - i_step)
        i_step += steps

        train_loss = _mae_multistep(
            model=model,
            mae_head=mae_head,
            loader=train_loader,
            preprocess=preprocess_train,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            steps=steps,
        )

        val_loss = _mae_eval(
            model=model,
            mae_head=mae_head,
            loader=val_loader,
            preprocess=preprocess_val,
        )

        if main_process:
            print(
                f"Step {i_step:>7} / {config['iterations']:>7} - "
                f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state": deepcopy(model.state_dict()),
                "mae_state": deepcopy(mae_head.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": deepcopy(config),
            }
            if save_path is not None:
                torch.save(best_state, os.path.join(save_path, "checkpoint.pt"))

    if main_process and best_state is not None and save_path is not None:
        torch.save(best_state, os.path.join(save_path, "checkpoint.pt"))


