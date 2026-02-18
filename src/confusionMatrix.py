import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.train import UltimateNet, GTZANFeatureDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
CHECKPOINT_PATH = "artifacts/sample_results/best_model.pth"
VAL_MANIFEST = "data/gtzan/manifests/val_manifest_fixed.csv"
FEATURE_DIR = "data/gtzan/features"
SAVE_PATH = "artifacts/results/confusion_matrix.png"

# Load dataset
val_ds = GTZANFeatureDataset(
    VAL_MANIFEST,
    mode="val",
    normalize=True
)

val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=32, shuffle=False
)

# Load model
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

model = UltimateNet(
    time_frames=ckpt["target_frames"],
    n_mels=ckpt["n_mels"],
    num_classes=len(ckpt["label_map"])
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Restore normalization
val_ds.global_mean = ckpt["global_mean"]
val_ds.global_std = ckpt["global_std"]

# Inference
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y.numpy())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=val_ds.inv_label_map.values(),
    yticklabels=val_ds.inv_label_map.values()
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - GTZAN")
plt.tight_layout()
plt.savefig(SAVE_PATH)
plt.close()

print(f"✅ Confusion matrix saved to {SAVE_PATH}")
