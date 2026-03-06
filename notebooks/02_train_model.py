"""
Notebook 2: Model Training
============================
Train the DPT²F model on multi-dataset features.
Run on Google Colab with T4 GPU.

Usage: Copy this to a Colab notebook, or run cells marked with # %%
"""

# %% [markdown]
# # 🚀 Model Training — DPT²F
# Trains the Dual-Path Temporal-Frequency Fusion Transformer
# on harmonized multi-dataset features.

# %% — Setup
from google.colab import drive
drive.mount('/content/drive')

import os
import sys

BASE_DIR = "/content/drive/MyDrive/music_genre_classification"
REPO_DIR = f"{BASE_DIR}/repo"
sys.path.insert(0, f"{REPO_DIR}/src")

os.system("pip install -q librosa soundfile tqdm pandas scikit-learn torch torchaudio matplotlib")

# %% — Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% — Load configurations and data
import pandas as pd
from config import get_configs, get_device

audio_cfg, model_cfg, train_cfg, aug_cfg, data_cfg = get_configs()
data_cfg.base_dir = BASE_DIR
data_cfg.__post_init__()

device = get_device()
print(f"Device: {device}")

# Load manifests
train_df = pd.read_csv(f"{data_cfg.manifests_dir}/train_manifest.csv")
val_df = pd.read_csv(f"{data_cfg.manifests_dir}/val_manifest.csv")
test_df = pd.read_csv(f"{data_cfg.manifests_dir}/test_manifest.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"\nGenre distribution (train):\n{train_df['genre'].value_counts().to_string()}")

# %% — Create dataloaders
from dataset import create_dataloaders
from utils import set_seed

set_seed(train_cfg.seed)

train_loader, val_loader, test_loader, global_mean, global_std = create_dataloaders(
    train_df, val_df, test_df,
    audio_config=audio_cfg,
    aug_config=aug_cfg,
    batch_size=train_cfg.batch_size,
    num_workers=train_cfg.num_workers,
    balanced_sampling=True,
)

print(f"\nNormalization stats: mean={global_mean:.6f}, std={global_std:.6f}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# %% — Build model
from model import build_model, model_summary

model = build_model(
    n_mels=model_cfg.n_mels,
    target_frames=model_cfg.target_frames,
    num_classes=model_cfg.num_classes,
    d_model=model_cfg.d_model,
    nhead=model_cfg.nhead,
    num_encoder_layers=model_cfg.num_encoder_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    transformer_dropout=model_cfg.transformer_dropout,
    fusion_nhead=model_cfg.fusion_nhead,
    fusion_layers=model_cfg.fusion_layers,
    classifier_hidden=model_cfg.classifier_hidden,
    classifier_dropout=model_cfg.classifier_dropout,
    use_gradient_checkpointing=train_cfg.use_gradient_checkpointing,
)

print(model_summary(model))

# Quick forward pass test
with torch.no_grad():
    dummy = torch.randn(2, model_cfg.target_frames, model_cfg.n_mels)
    out = model(dummy)
    print(f"\n✅ Forward pass test: input {dummy.shape} → output {out.shape}")

# %% [markdown]
# ## Training
# Training with:
# - Mixed precision (FP16) for T4 efficiency
# - Gradient accumulation (effective batch 32)
# - Cosine warmup learning rate schedule
# - Label smoothing + MixUp augmentation
# - Early stopping on validation accuracy

# %% — Train model
from train import train

history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=train_cfg,
    save_dir=data_cfg.checkpoint_dir,
    experiment_name="dpt2f_multi_dataset",
)

# %% — Plot training curves
from evaluate import plot_training_history

import matplotlib.pyplot as plt

fig = plot_training_history(
    history,
    save_path=f"{data_cfg.results_dir}/training_curves.png"
)
plt.show()

# %% [markdown]
# ## Quick Evaluation on Test Set

# %% — Evaluate
from evaluate import full_evaluation
from utils import load_checkpoint

# Load best checkpoint
best_path = f"{data_cfg.checkpoint_dir}/dpt2f_multi_dataset_best.pth"
checkpoint = load_checkpoint(best_path, model, device=device)
model = model.to(device)

metrics = full_evaluation(
    model=model,
    test_loader=test_loader,
    save_dir=data_cfg.results_dir,
    experiment_name="dpt2f_multi_dataset",
    device=device,
)

print(f"\n🏆 Final Test Accuracy: {metrics['accuracy']:.4f}")
print(f"   Macro F1: {metrics['macro_f1']:.4f}")

# %% — Save normalization stats with checkpoint
import json

stats = {
    "global_mean": float(global_mean),
    "global_std": float(global_std),
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "best_accuracy": float(metrics["accuracy"]),
    "macro_f1": float(metrics["macro_f1"]),
}

with open(f"{data_cfg.results_dir}/training_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n💾 Stats saved to {data_cfg.results_dir}/training_stats.json")
print("\n✅ Training complete! Proceed to notebook 03_evaluation.py for detailed analysis.")
