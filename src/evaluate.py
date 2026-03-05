"""
Evaluation module for DPT²F model.
Computes metrics, confusion matrices, and generates visualizations for research paper.
"""
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import UNIFIED_GENRES, IDX_TO_GENRE, GENRE_EMOTIONS, get_device


@torch.no_grad()
def predict_all(
    model,
    data_loader,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all predictions from a dataloader.

    Returns:
        (all_preds, all_labels, all_probs) arrays
    """
    if device is None:
        device = get_device()

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for features, labels in data_loader:
        features = features.to(device, non_blocking=True)
        logits = model(features)
        probs = torch.softmax(logits, dim=-1)

        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs)
    )


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    genre_names: list = None,
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Returns:
        Dict with accuracy, per-class metrics, macro/weighted averages
    """
    if genre_names is None:
        genre_names = UNIFIED_GENRES

    accuracy = accuracy_score(labels, preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(len(genre_names)), zero_division=0
    )

    # Macro and weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    # Build per-class dict
    per_class = {}
    for i, genre in enumerate(genre_names):
        per_class[genre] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]) if i < len(support) else 0,
        }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    genre_names: list = None,
    save_path: str = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Generate publication-quality confusion matrix."""
    if genre_names is None:
        genre_names = UNIFIED_GENRES

    cm = confusion_matrix(labels, preds, labels=range(len(genre_names)))

    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_display = cm_norm
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=figsize)

    # Color map
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels
    ax.set(
        xticks=np.arange(len(genre_names)),
        yticks=np.arange(len(genre_names)),
        xticklabels=genre_names,
        yticklabels=genre_names,
        ylabel="True Genre",
        xlabel="Predicted Genre",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Text annotations
    thresh = cm_display.max() / 2.0
    for i in range(len(genre_names)):
        for j in range(len(genre_names)):
            val = cm_display[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Confusion matrix saved to {save_path}")

    return fig


def plot_training_history(
    history: Dict,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot training curves for research paper."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=1.5)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Learning rate
    axes[2].plot(epochs, history["lr"], "g-", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule", fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📈 Training history saved to {save_path}")

    return fig


def plot_per_class_metrics(
    metrics: Dict,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot per-class precision, recall, F1 bar chart."""
    per_class = metrics["per_class"]
    genres = list(per_class.keys())
    precision = [per_class[g]["precision"] for g in genres]
    recall = [per_class[g]["recall"] for g in genres]
    f1 = [per_class[g]["f1"] for g in genres]

    x = np.arange(len(genres))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#667eea", alpha=0.8)
    bars2 = ax.bar(x, recall, width, label="Recall", color="#2EC4B6", alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#FF006E", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Metrics", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in genres], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Per-class metrics saved to {save_path}")

    return fig


def get_embeddings(
    model,
    data_loader,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature embeddings before classifier for t-SNE visualization.

    Returns:
        (embeddings, labels) arrays
    """
    if device is None:
        device = get_device()

    model.eval()
    embeddings = []
    labels = []

    # Hook into attention pooling output
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output["embedding"] = output.detach().cpu()

    handle = model.attention_pool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for features, batch_labels in data_loader:
            features = features.to(device, non_blocking=True)
            _ = model(features)
            embeddings.append(hook_output["embedding"].numpy())
            labels.append(batch_labels.numpy())

    handle.remove()

    return np.concatenate(embeddings), np.concatenate(labels)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    genre_names: list = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
) -> plt.Figure:
    """
    t-SNE visualization of learned embeddings for research paper.
    """
    from sklearn.manifold import TSNE

    if genre_names is None:
        genre_names = UNIFIED_GENRES

    # Compute t-SNE
    print("  Computing t-SNE embeddings...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Get colors from genre emotions
    colors = [GENRE_EMOTIONS[g]["color"] for g in genre_names]

    fig, ax = plt.subplots(figsize=figsize)

    for i, genre in enumerate(genre_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[i],
            label=genre.capitalize(),
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_title("t-SNE Visualization of Learned Embeddings", fontweight="bold", fontsize=14)
    ax.legend(loc="best", fontsize=9, markerscale=2)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  🎯 t-SNE plot saved to {save_path}")

    return fig


def full_evaluation(
    model,
    test_loader,
    save_dir: str = "./results",
    experiment_name: str = "dpt2f",
    device: torch.device = None,
) -> Dict:
    """
    Run full evaluation pipeline: metrics, confusion matrix, t-SNE.

    Returns:
        Complete metrics dict
    """
    if device is None:
        device = get_device()

    print(f"\n📊 Running full evaluation ({experiment_name})...")

    # Get predictions
    preds, labels, probs = predict_all(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(preds, labels)

    # Print report
    print(f"\n{'='*50}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"{'='*50}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(
        labels, preds, target_names=UNIFIED_GENRES, zero_division=0
    ))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    plot_confusion_matrix(
        preds, labels,
        save_path=str(save_dir / f"{experiment_name}_confusion_matrix.png"),
        title=f"Confusion Matrix — {experiment_name}"
    )

    # Per-class metrics
    plot_per_class_metrics(
        metrics,
        save_path=str(save_dir / f"{experiment_name}_per_class_metrics.png")
    )

    # t-SNE
    embeddings, emb_labels = get_embeddings(model, test_loader, device)
    plot_tsne(
        embeddings, emb_labels,
        save_path=str(save_dir / f"{experiment_name}_tsne.png")
    )

    # Save metrics to JSON
    metrics_path = save_dir / f"{experiment_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  📁 Metrics saved to {metrics_path}")

    return metrics
