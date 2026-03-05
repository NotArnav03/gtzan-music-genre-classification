"""
Notebook 1: Data Preparation
=============================
Download, preprocess, and harmonize GTZAN + FMA-small + MagnaTagATune datasets.
Run this notebook first on Google Colab.

Usage: Copy this to a Colab notebook, or run cells marked with # %%
"""

# %% [markdown]
# # 📀 Data Preparation — Multi-Dataset Music Genre Classification
# Downloads and preprocesses GTZAN, FMA-small, and MagnaTagATune.
# Extracts mel spectrogram features and creates harmonized manifests.

# %% — Setup and Imports
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import sys

# Configuration — EDIT THIS
BASE_DIR = "/content/drive/MyDrive/music_genre_classification"
os.makedirs(BASE_DIR, exist_ok=True)

# Clone repo or copy src files
REPO_DIR = f"{BASE_DIR}/repo"
if not os.path.exists(f"{REPO_DIR}/src/config.py"):
    os.system(f"git clone https://github.com/NotArnav03/gtzan-music-genre-classification.git {REPO_DIR}")

sys.path.insert(0, f"{REPO_DIR}/src")

# Install dependencies
os.system("pip install -q librosa soundfile tqdm pandas scikit-learn torch torchaudio matplotlib")

# %% — Import modules
from config import AudioConfig, DataConfig, get_configs
from dataset import MultiDatasetBuilder, AudioFeatureExtractor

audio_cfg, model_cfg, train_cfg, aug_cfg, data_cfg = get_configs()

# Override paths for this Colab session
data_cfg.base_dir = BASE_DIR
data_cfg.__post_init__()

print(f"Base directory: {BASE_DIR}")
print(f"Audio config: SR={audio_cfg.sample_rate}, n_mels={audio_cfg.n_mels}, "
      f"target_frames={audio_cfg.target_frames}")

# %% [markdown]
# ## 1. Download Datasets
#
# ### GTZAN Dataset
# The GTZAN dataset contains 1000 audio tracks of 30 seconds each,
# distributed across 10 genres.

# %% — Download GTZAN
GTZAN_DIR = data_cfg.gtzan_dir
os.makedirs(GTZAN_DIR, exist_ok=True)

if not os.path.exists(f"{GTZAN_DIR}/genres"):
    print("📥 Downloading GTZAN dataset...")
    os.system(f"""
    cd /tmp && \
    wget -q http://opihi.cs.uvic.ca/sound/genres.tar.gz && \
    tar -xzf genres.tar.gz -C {GTZAN_DIR} && \
    rm genres.tar.gz
    """)
    print("✅ GTZAN downloaded!")
else:
    print("✅ GTZAN already exists")

# Count files
import glob
gtzan_files = glob.glob(f"{GTZAN_DIR}/genres/**/*.wav", recursive=True)
print(f"   Total GTZAN audio files: {len(gtzan_files)}")

# %% [markdown]
# ### FMA-small Dataset
# FMA-small contains 8,000 tracks of 30 seconds across 8 genres.

# %% — Download FMA-small
FMA_DIR = data_cfg.fma_dir
os.makedirs(FMA_DIR, exist_ok=True)

if not os.path.exists(f"{FMA_DIR}/fma_small") and not os.path.exists(f"{FMA_DIR}/000"):
    print("📥 Downloading FMA-small dataset...")
    print("   (This may take a while — ~7.2 GB)")
    os.system(f"""
    cd /tmp && \
    wget -q https://os.unil.cloud.switch.ch/fma/fma_small.zip && \
    unzip -q fma_small.zip -d {FMA_DIR} && \
    rm fma_small.zip
    """)
    # Also download metadata
    os.system(f"""
    cd /tmp && \
    wget -q https://os.unil.cloud.switch.ch/fma/fma_metadata.zip && \
    unzip -q -o fma_metadata.zip -d {FMA_DIR} && \
    rm fma_metadata.zip
    """)
    print("✅ FMA-small downloaded!")
else:
    print("✅ FMA-small already exists")

# %% [markdown]
# ### MagnaTagATune Dataset
# MagnaTagATune contains ~25,000 audio clips with tag annotations.

# %% — Download MagnaTagATune
MTT_DIR = data_cfg.mtt_dir
os.makedirs(MTT_DIR, exist_ok=True)

if not os.path.exists(f"{MTT_DIR}/annotations.csv") and not os.path.exists(f"{MTT_DIR}/annotations_final.csv"):
    print("📥 Downloading MagnaTagATune dataset...")
    print("   (Annotations + audio — this takes a while)")
    # Download annotations
    os.system(f"""
    cd {MTT_DIR} && \
    wget -q https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset/annotations_final.csv && \
    mv annotations_final.csv annotations.csv
    """)
    # Download audio (3 parts)
    for part in range(1, 4):
        part_str = chr(96 + part)  # a, b, c
        url = f"https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.00{part}"
        print(f"   Downloading part {part}/3...")
        os.system(f"cd {MTT_DIR} && wget -q {url}")
    # Combine and extract
    os.system(f"""
    cd {MTT_DIR} && \
    cat mp3.zip.* > mp3.zip && \
    unzip -q mp3.zip && \
    rm mp3.zip mp3.zip.*
    """)
    print("✅ MagnaTagATune downloaded!")
else:
    print("✅ MagnaTagATune already exists")

# %% [markdown]
# ## 2. Build Manifests and Extract Features

# %% — Build unified manifests
builder = MultiDatasetBuilder(
    audio_config=audio_cfg,
    features_dir=data_cfg.features_dir,
    manifests_dir=data_cfg.manifests_dir,
)

# Build per-dataset manifests
gtzan_manifest = builder.build_gtzan_manifest(GTZAN_DIR)
fma_manifest = builder.build_fma_manifest(FMA_DIR)
mtt_manifest = builder.build_mtt_manifest(MTT_DIR)

print(f"\n📊 Dataset sizes:")
print(f"   GTZAN: {len(gtzan_manifest)} tracks")
print(f"   FMA-small: {len(fma_manifest)} tracks")
print(f"   MagnaTagATune: {len(mtt_manifest)} tracks")

# %% — Extract features
import pandas as pd

# Combine all manifests
all_manifests = []
if len(gtzan_manifest) > 0:
    all_manifests.append(gtzan_manifest)
if len(fma_manifest) > 0:
    all_manifests.append(fma_manifest)
if len(mtt_manifest) > 0:
    all_manifests.append(mtt_manifest)

combined = pd.concat(all_manifests, ignore_index=True)
print(f"\n📊 Combined manifest: {len(combined)} tracks")
print(f"   Genre distribution:\n{combined['genre'].value_counts().to_string()}")

# Extract features (this is the longest step - ~30-60 min for all datasets)
combined = builder.extract_all_features(combined, force=False)

# Save combined manifest
combined.to_csv(f"{data_cfg.manifests_dir}/combined_manifest.csv", index=False)
print(f"\n💾 Saved combined manifest to {data_cfg.manifests_dir}/combined_manifest.csv")

# %% [markdown]
# ## 3. Create Train/Val/Test Splits

# %% — Artist-stratified splits
from utils import set_seed
set_seed(42)

train_df, val_df, test_df = builder.create_splits(
    combined,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    artist_stratified=True,
)

# Save splits
train_df.to_csv(f"{data_cfg.manifests_dir}/train_manifest.csv", index=False)
val_df.to_csv(f"{data_cfg.manifests_dir}/val_manifest.csv", index=False)
test_df.to_csv(f"{data_cfg.manifests_dir}/test_manifest.csv", index=False)

print(f"\n💾 Saved split manifests:")
print(f"   Train: {len(train_df)} ({data_cfg.manifests_dir}/train_manifest.csv)")
print(f"   Val:   {len(val_df)} ({data_cfg.manifests_dir}/val_manifest.csv)")
print(f"   Test:  {len(test_df)} ({data_cfg.manifests_dir}/test_manifest.csv)")

# %% [markdown]
# ## 4. Verify Features

# %% — Quick sanity check
import numpy as np

sample_path = train_df.iloc[0]["feature_path"]
sample = np.load(sample_path)
print(f"Sample feature shape: {sample.shape}")
print(f"Sample stats: mean={sample.mean():.4f}, std={sample.std():.4f}, "
      f"min={sample.min():.4f}, max={sample.max():.4f}")

# Visualize a sample spectrogram
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(sample.T, aspect="auto", origin="lower", cmap="magma")
ax.set_xlabel("Time Frames")
ax.set_ylabel("Mel Bins")
ax.set_title(f"Sample: {train_df.iloc[0]['filename']} ({train_df.iloc[0]['genre']})")
plt.tight_layout()
plt.show()

print("\n✅ Data preparation complete! Proceed to notebook 02_train_model.py")
