"""
Training loop for DPT²F model.
Optimized for Google Colab T4 GPU with mixed precision and gradient accumulation.
"""
import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict
from pathlib import Path

from config import TrainingConfig, ModelConfig, AudioConfig, get_device, UNIFIED_GENRES
from model import DPT2F, build_model, model_summary
from utils import (
    set_seed, count_parameters, format_params,
    AverageMeter, EarlyStopping, CosineWarmupScheduler,
    save_checkpoint, load_checkpoint,
    label_smoothing_loss, mixup_data, mixup_criterion,
    TrainingLogger
)


def train_one_epoch(
    model: DPT2F,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 0,
) -> Tuple[float, float]:
    """
    Train for one epoch with AMP and gradient accumulation.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    loss_meter = AverageMeter("loss")
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Train E{epoch}", leave=False)
    for step, (features, labels) in enumerate(pbar):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp augmentation
        use_mixup = np.random.random() < 0.5
        if use_mixup:
            features, y_a, y_b, lam = mixup_data(
                features, labels, alpha=config.label_smoothing
            )

        # Forward pass with AMP
        if config.use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(features)
                if use_mixup:
                    loss = mixup_criterion(
                        None, logits, y_a, y_b, lam,
                        smoothing=config.label_smoothing
                    )
                else:
                    loss = label_smoothing_loss(
                        logits, labels, smoothing=config.label_smoothing
                    )
                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(features)
            if use_mixup:
                loss = mixup_criterion(
                    None, logits, y_a, y_b, lam,
                    smoothing=config.label_smoothing
                )
            else:
                loss = label_smoothing_loss(
                    logits, labels, smoothing=config.label_smoothing
                )
            loss = loss / config.accumulation_steps
            loss.backward()

            if (step + 1) % config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                optimizer.step()
                optimizer.zero_grad()

        # Metrics
        loss_meter.update(loss.item() * config.accumulation_steps, features.size(0))
        preds = logits.argmax(dim=-1)
        if not use_mixup:
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    accuracy = correct / max(total, 1)
    return loss_meter.avg, accuracy


@torch.no_grad()
def validate(
    model: DPT2F,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
) -> Tuple[float, float]:
    """
    Validate model.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    correct = 0
    total = 0

    for features, labels in val_loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if config.use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(features)
                loss = label_smoothing_loss(logits, labels, smoothing=0.0)
        else:
            logits = model(features)
            loss = label_smoothing_loss(logits, labels, smoothing=0.0)

        loss_meter.update(loss.item(), features.size(0))
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / max(total, 1)
    return loss_meter.avg, accuracy


def train(
    model: DPT2F,
    train_loader,
    val_loader,
    config: TrainingConfig,
    save_dir: str = "./checkpoints",
    experiment_name: str = "dpt2f",
    resume_path: Optional[str] = None,
) -> Dict:
    """
    Full training loop.

    Returns:
        Training history dict
    """
    device = get_device()
    model = model.to(device)

    # Print model summary
    print(model_summary(model))
    total, trainable = count_parameters(model)
    print(f"\nTrainable parameters: {format_params(trainable)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    # Scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.max_epochs,
        min_lr=config.min_lr,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if config.use_amp and device.type == "cuda" else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode="max"
    )

    # Logger
    logger = TrainingLogger(save_dir)

    # Resume from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    if resume_path and os.path.exists(resume_path):
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("best_acc", 0.0)
        print(f"📂 Resumed from epoch {start_epoch - 1}, best acc: {best_val_acc:.4f}")

    # Save directory
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"{experiment_name}_best.pth")
    last_path = os.path.join(save_dir, f"{experiment_name}_last.pth")

    print(f"\n🚀 Starting training for {config.max_epochs} epochs...")
    print(f"   Device: {device}")
    print(f"   Batch size: {config.batch_size} × {config.accumulation_steps} = "
          f"{config.batch_size * config.accumulation_steps} effective")
    print(f"   AMP: {config.use_amp}")
    print(f"   Learning rate: {config.learning_rate}")
    print()

    for epoch in range(start_epoch, config.max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, config, scaler, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, device, config)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_lr()

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "epoch_time": epoch_time,
        }
        logger.log_epoch(epoch, metrics)
        logger.print_epoch(epoch, config.max_epochs, metrics)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_path, model, optimizer, epoch, best_val_acc,
                config={"experiment": experiment_name},
                extra={"genres": UNIFIED_GENRES}
            )
            print(f"  ✅ New best: {best_val_acc:.4f}")

        # Save last model
        save_checkpoint(
            last_path, model, optimizer, epoch, best_val_acc,
            config={"experiment": experiment_name},
            extra={"genres": UNIFIED_GENRES}
        )

        # Early stopping
        if early_stopping(val_acc):
            print(f"\n⚠️ Early stopping at epoch {epoch} (best: {best_val_acc:.4f})")
            break

    print(f"\n🏁 Training complete! Best validation accuracy: {best_val_acc:.4f}")
    return logger.get_history()
