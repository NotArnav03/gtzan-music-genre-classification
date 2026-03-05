"""
Notebook 3: Evaluation & Ablation Studies
==========================================
Comprehensive evaluation, ablation studies, and visualization
for the research paper.

Usage: Copy this to a Colab notebook, or run cells marked with # %%
"""

# %% [markdown]
# # 📊 Evaluation & Ablation Studies
# Detailed analysis for the research paper including:
# 1. Full test metrics & confusion matrices
# 2. Per-dataset evaluation (cross-dataset generalization)
# 3. Ablation studies
# 4. t-SNE visualization
# 5. Attention weight analysis

# %% — Setup
from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = "/content/drive/MyDrive/music_genre_classification"
REPO_DIR = f"{BASE_DIR}/repo"
sys.path.insert(0, f"{REPO_DIR}/src")

os.system("pip install -q librosa soundfile tqdm pandas scikit-learn torch torchaudio matplotlib")

from config import get_configs, get_device, UNIFIED_GENRES, GENRE_EMOTIONS
from model import build_model, model_summary
from dataset import GenreDataset, create_dataloaders
from evaluate import (
    predict_all, compute_metrics, plot_confusion_matrix,
    plot_per_class_metrics, plot_tsne, get_embeddings, full_evaluation
)
from utils import load_checkpoint, set_seed

set_seed(42)
device = get_device()

audio_cfg, model_cfg, train_cfg, aug_cfg, data_cfg = get_configs()
data_cfg.base_dir = BASE_DIR
data_cfg.__post_init__()

# %% — Load model and data
model = build_model(
    n_mels=model_cfg.n_mels,
    target_frames=model_cfg.target_frames,
    num_classes=model_cfg.num_classes,
    d_model=model_cfg.d_model,
    use_gradient_checkpointing=False,  # No need during eval
)

best_path = f"{data_cfg.checkpoint_dir}/dpt2f_multi_dataset_best.pth"
load_checkpoint(best_path, model, device=device)
model = model.to(device)
model.eval()

test_df = pd.read_csv(f"{data_cfg.manifests_dir}/test_manifest.csv")
stats = json.load(open(f"{data_cfg.results_dir}/training_stats.json"))

print(model_summary(model))

# %% [markdown]
# ## 1. Full Test Set Evaluation

# %% — Run evaluation
from augmentations import MusicAugmentor

test_dataset = GenreDataset(
    test_df,
    target_frames=model_cfg.target_frames,
    mode="eval",
    normalize=True,
    global_mean=stats["global_mean"],
    global_std=stats["global_std"],
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

metrics = full_evaluation(
    model=model,
    test_loader=test_loader,
    save_dir=data_cfg.results_dir,
    experiment_name="dpt2f_final",
    device=device,
)

# %% [markdown]
# ## 2. Per-Dataset Evaluation (Cross-Dataset Generalization)
# Evaluate the unified model on each dataset's test set separately.
# This measures how well the model generalizes across datasets.

# %% — Per-dataset evaluation
results_per_dataset = {}

for dataset_name in ["gtzan", "fma", "mtt"]:
    subset = test_df[test_df["dataset"] == dataset_name]
    if len(subset) == 0:
        print(f"\n⚠️ No {dataset_name} samples in test set, skipping...")
        continue

    subset_dataset = GenreDataset(
        subset,
        target_frames=model_cfg.target_frames,
        mode="eval",
        normalize=True,
        global_mean=stats["global_mean"],
        global_std=stats["global_std"],
    )
    subset_loader = torch.utils.data.DataLoader(
        subset_dataset, batch_size=16, shuffle=False, num_workers=2
    )

    preds, labels, probs = predict_all(model, subset_loader, device)
    subset_metrics = compute_metrics(preds, labels)

    results_per_dataset[dataset_name] = subset_metrics

    print(f"\n{'='*40}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"  Samples: {len(subset)}")
    print(f"  Accuracy: {subset_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {subset_metrics['macro_f1']:.4f}")

    plot_confusion_matrix(
        preds, labels,
        save_path=f"{data_cfg.results_dir}/cm_{dataset_name}.png",
        title=f"Confusion Matrix — {dataset_name.upper()}"
    )

# Summary table
print(f"\n{'='*60}")
print(f"{'Dataset':<15} {'Accuracy':<12} {'Macro F1':<12} {'Samples':<10}")
print(f"{'-'*60}")
for ds, m in results_per_dataset.items():
    n = len(test_df[test_df["dataset"] == ds])
    print(f"{ds.upper():<15} {m['accuracy']:<12.4f} {m['macro_f1']:<12.4f} {n:<10}")
print(f"{'-'*60}")
print(f"{'COMBINED':<15} {metrics['accuracy']:<12.4f} {metrics['macro_f1']:<12.4f} {len(test_df):<10}")

# %% [markdown]
# ## 3. Ablation Studies
# Compare DPT²F variants to understand contribution of each component.

# %% — Ablation: Architecture variants
# Uncomment and run after training each variant

ablation_results = {
    "DPT²F (Full)": metrics,
}

# Save ablation results
print("\n📊 Ablation Study Results:")
print(f"{'Method':<30} {'Accuracy':<12} {'Macro F1':<12}")
print(f"{'-'*55}")
for method, m in ablation_results.items():
    print(f"{method:<30} {m['accuracy']:<12.4f} {m['macro_f1']:<12.4f}")

# %% [markdown]
# ## 4. Generate Research Paper Tables

# %% — LaTeX table: Comparison with SOTA
print("\n📄 LaTeX Table: Comparison with State-of-the-Art")
print("="*60)

sota_results = {
    "CNN (Modified, 2024)": {"gtzan": 92.7},
    "AST (2024)": {"gtzan": 85.5},
    "WavLM (2024)": {"gtzan": 84.6},
    "wav2vec 2.0 (2024)": {"gtzan": 81.2},
    "HuBERT (2024)": {"gtzan": 81.4},
    "EfficientNet-B3 (2023)": {"fma": 65.6},
    "CRNN (2023)": {"fma": 65.2},
}

print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Comparison with state-of-the-art methods}")
print(r"\begin{tabular}{lccc}")
print(r"\hline")
print(r"Method & GTZAN (\%) & FMA (\%) & Multi-Dataset (\%) \\")
print(r"\hline")

for method, accs in sota_results.items():
    gtzan = f"{accs.get('gtzan', '-')}"
    fma = f"{accs.get('fma', '-')}"
    print(f"{method} & {gtzan} & {fma} & - \\\\")

# Our results
our_gtzan = results_per_dataset.get("gtzan", {}).get("accuracy", 0) * 100
our_fma = results_per_dataset.get("fma", {}).get("accuracy", 0) * 100
our_combined = metrics["accuracy"] * 100
print(r"\hline")
print(f"\\textbf{{DPT²F (Ours)}} & \\textbf{{{our_gtzan:.1f}}} & "
      f"\\textbf{{{our_fma:.1f}}} & \\textbf{{{our_combined:.1f}}} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

# %% — Save all results
all_results = {
    "overall": metrics,
    "per_dataset": results_per_dataset,
    "ablation": {k: v for k, v in ablation_results.items()},
}

with open(f"{data_cfg.results_dir}/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n💾 All results saved to {data_cfg.results_dir}/all_results.json")
print("\n✅ Evaluation complete!")
