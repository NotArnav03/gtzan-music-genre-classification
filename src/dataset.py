"""
Multi-dataset loader with harmonized genre taxonomy.
Supports GTZAN, FMA-small, and MagnaTagATune with unified labels.
"""
import os
import math
import random
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

from config import (
    AudioConfig, AugmentationConfig, UNIFIED_GENRES, GENRE_TO_IDX,
    GTZAN_GENRE_MAP, FMA_GENRE_MAP, MTT_TAG_MAP
)
from augmentations import MusicAugmentor, AugConfig


class AudioFeatureExtractor:
    """Extract log-mel spectrogram features from audio files."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.cfg = config or AudioConfig()

    def extract(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract log-mel spectrogram from audio file.

        Returns:
            np.ndarray of shape (T, n_mels) or None if extraction fails
        """
        try:
            y, sr = librosa.load(
                audio_path,
                sr=self.cfg.sample_rate,
                duration=self.cfg.duration,
                mono=True
            )

            # Pad if shorter than expected
            target_length = int(self.cfg.duration * self.cfg.sample_rate)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')

            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
                fmin=self.cfg.fmin,
                fmax=self.cfg.fmax,
            )

            # Convert to log scale (add small epsilon to avoid log(0))
            log_mel = np.log(mel_spec + 1e-9)

            # Transpose to (T, F) from (F, T)
            log_mel = log_mel.T

            return log_mel.astype(np.float32)

        except Exception as e:
            print(f"  [WARN] Failed to extract features from {audio_path}: {e}")
            return None


class MultiDatasetBuilder:
    """
    Build unified manifests from multiple datasets.
    Handles genre mapping, artist-aware splitting, and feature extraction.
    """

    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        features_dir: str = "./features",
        manifests_dir: str = "./manifests"
    ):
        self.audio_cfg = audio_config or AudioConfig()
        self.extractor = AudioFeatureExtractor(self.audio_cfg)
        self.features_dir = Path(features_dir)
        self.manifests_dir = Path(manifests_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def build_gtzan_manifest(self, gtzan_dir: str) -> pd.DataFrame:
        """
        Build manifest from GTZAN dataset.
        Handles artist-aware splitting to prevent data leakage.
        """
        print("\n📀 Processing GTZAN dataset...")
        gtzan_path = Path(gtzan_dir)
        records = []

        # GTZAN structure: gtzan/genres/<genre>/<genre>.<id>.wav
        genres_dir = gtzan_path / "genres"
        if not genres_dir.exists():
            # Try alternative structure
            genres_dir = gtzan_path
            if not any(genres_dir.iterdir()):
                print(f"  [ERROR] No genres found in {gtzan_dir}")
                return pd.DataFrame()

        for genre_dir in sorted(genres_dir.iterdir()):
            if not genre_dir.is_dir():
                continue
            genre_name = genre_dir.name.lower()
            mapped = GTZAN_GENRE_MAP.get(genre_name)
            if mapped is None:
                continue

            for audio_file in sorted(genre_dir.glob("*.wav")):
                # Extract artist from filename if possible
                # GTZAN format: genre.NNNNN.wav
                parts = audio_file.stem.split(".")
                artist_id = parts[0] if len(parts) >= 2 else "unknown"

                feat_path = self.features_dir / f"gtzan_{audio_file.stem}.npy"

                records.append({
                    "audio_path": str(audio_file),
                    "feature_path": str(feat_path),
                    "genre": mapped,
                    "dataset": "gtzan",
                    "artist": f"gtzan_{artist_id}",
                    "filename": audio_file.name,
                })

        df = pd.DataFrame(records)
        print(f"  Found {len(df)} tracks across {df['genre'].nunique()} genres")
        return df

    def build_fma_manifest(self, fma_dir: str) -> pd.DataFrame:
        """Build manifest from FMA-small dataset."""
        print("\n🎵 Processing FMA-small dataset...")
        fma_path = Path(fma_dir)
        records = []

        # FMA structure: fma_small/<track_id_prefix>/<track_id>.mp3
        # Metadata in tracks.csv (may be in subdirectory after extraction)
        tracks_csv = None
        for candidate in [
            fma_path / "tracks.csv",
            fma_path / "fma_metadata" / "tracks.csv",
            fma_path / "metadata" / "tracks.csv",
        ]:
            if candidate.exists():
                tracks_csv = candidate
                break

        if tracks_csv is not None:
            print(f"  [INFO] Found metadata at {tracks_csv}")
            # Load FMA metadata (multi-level header)
            tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
            for track_id, row in tracks.iterrows():
                try:
                    genre = row[("track", "genre_top")]
                    if pd.isna(genre) or not isinstance(genre, str):
                        continue
                    genre = genre.lower().strip()
                except (KeyError, AttributeError):
                    continue

                mapped = FMA_GENRE_MAP.get(genre)
                if mapped is None:
                    continue

                # Construct audio path — try multiple locations
                tid = str(track_id).zfill(6)
                audio_file = None
                for audio_candidate in [
                    fma_path / "fma_small" / tid[:3] / f"{tid}.mp3",
                    fma_path / tid[:3] / f"{tid}.mp3",
                ]:
                    if audio_candidate.exists():
                        audio_file = audio_candidate
                        break

                if audio_file is None:
                    continue

                try:
                    artist = str(row[("artist", "name")])
                except (KeyError, AttributeError):
                    artist = f"fma_{track_id}"

                feat_path = self.features_dir / f"fma_{tid}.npy"

                records.append({
                    "audio_path": str(audio_file),
                    "feature_path": str(feat_path),
                    "genre": mapped,
                    "dataset": "fma",
                    "artist": f"fma_{artist}",
                    "filename": f"{tid}.mp3",
                })
        else:
            # No metadata → skip FMA (can't classify without genre labels)
            print("  [WARN] No tracks.csv found anywhere in FMA directory.")
            print("  [WARN] FMA requires metadata for genre labels. Skipping FMA.")
            print("  [WARN] Download metadata: wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip")

        df = pd.DataFrame(records)
        if len(df) > 0:
            print(f"  Found {len(df)} tracks across {df['genre'].nunique()} genres")
        else:
            print("  [WARN] No FMA tracks processed")
        return df

    def build_mtt_manifest(self, mtt_dir: str) -> pd.DataFrame:
        """Build manifest from MagnaTagATune dataset."""
        print("\n🎶 Processing MagnaTagATune dataset...")
        mtt_path = Path(mtt_dir)
        records = []

        # MTT structure: annotations.csv + audio in subdirectories
        annotations = mtt_path / "annotations.csv"
        clip_info = mtt_path / "clip_info_final.csv"

        if annotations.exists():
            ann_df = pd.read_csv(annotations, sep="\t") if annotations.suffix == ".csv" else pd.read_csv(annotations)

            # MTT has binary tag columns
            for idx, row in ann_df.iterrows():
                # Find genre tags that are active
                mapped_genre = None
                max_score = 0

                for tag, unified in MTT_TAG_MAP.items():
                    if tag in row and row[tag] > max_score:
                        max_score = row[tag]
                        mapped_genre = unified

                if mapped_genre is None or max_score == 0:
                    continue

                # Get audio path
                if "mp3_path" in row:
                    audio_file = mtt_path / row["mp3_path"]
                elif "clip_id" in row:
                    clip_id = str(row["clip_id"])
                    # MTT audio: <prefix>/<clip_id>.mp3
                    audio_file = mtt_path / f"{clip_id[:1]}/{clip_id}.mp3"
                else:
                    continue

                if not audio_file.exists():
                    continue

                feat_name = f"mtt_{audio_file.stem}"
                feat_path = self.features_dir / f"{feat_name}.npy"

                records.append({
                    "audio_path": str(audio_file),
                    "feature_path": str(feat_path),
                    "genre": mapped_genre,
                    "dataset": "mtt",
                    "artist": f"mtt_{idx}",  # MTT doesn't have clean artist info
                    "filename": audio_file.name,
                })
        else:
            print(f"  [WARN] No annotations found in {mtt_dir}")

        df = pd.DataFrame(records)
        if len(df) > 0:
            print(f"  Found {len(df)} tracks across {df['genre'].nunique()} genres")
        else:
            print("  [WARN] No MagnaTagATune tracks processed")
        return df

    def extract_all_features(self, manifest: pd.DataFrame, force: bool = False):
        """Extract mel spectrogram features for all tracks in manifest."""
        print(f"\n🔊 Extracting features for {len(manifest)} tracks...")
        success, failed = 0, 0

        for idx, row in manifest.iterrows():
            feat_path = row["feature_path"]

            # Skip if already extracted
            if os.path.exists(feat_path) and not force:
                success += 1
                continue

            features = self.extractor.extract(row["audio_path"])
            if features is not None:
                os.makedirs(os.path.dirname(feat_path), exist_ok=True)
                np.save(feat_path, features)
                success += 1
            else:
                failed += 1

            if (success + failed) % 200 == 0:
                print(f"  Progress: {success + failed}/{len(manifest)} "
                      f"(✅ {success} | ❌ {failed})")

        print(f"  Done: ✅ {success} extracted | ❌ {failed} failed")
        return manifest[manifest["feature_path"].apply(os.path.exists)]

    def create_splits(
        self,
        manifest: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        artist_stratified: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/val/test splits with artist stratification.
        Ensures no artist appears in multiple splits (prevents data leakage).
        """
        print(f"\n📊 Creating splits (train={train_ratio}, val={val_ratio}, test={test_ratio})...")

        if artist_stratified:
            # Group by artist, then split artist groups
            artists = manifest["artist"].unique()
            np.random.shuffle(artists)

            n_train = int(len(artists) * train_ratio)
            n_val = int(len(artists) * val_ratio)

            train_artists = set(artists[:n_train])
            val_artists = set(artists[n_train:n_train + n_val])
            test_artists = set(artists[n_train + n_val:])

            train_df = manifest[manifest["artist"].isin(train_artists)].copy()
            val_df = manifest[manifest["artist"].isin(val_artists)].copy()
            test_df = manifest[manifest["artist"].isin(test_artists)].copy()
        else:
            # Simple stratified split by genre
            train_dfs, val_dfs, test_dfs = [], [], []
            for genre in manifest["genre"].unique():
                genre_df = manifest[manifest["genre"] == genre].sample(frac=1.0)
                n = len(genre_df)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)

                train_dfs.append(genre_df.iloc[:n_train])
                val_dfs.append(genre_df.iloc[n_train:n_train + n_val])
                test_dfs.append(genre_df.iloc[n_train + n_val:])

            train_df = pd.concat(train_dfs).reset_index(drop=True)
            val_df = pd.concat(val_dfs).reset_index(drop=True)
            test_df = pd.concat(test_dfs).reset_index(drop=True)

        print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            dist = split_df["genre"].value_counts().to_dict()
            print(f"  {split_name} distribution: {dict(sorted(dist.items()))}")

        return train_df, val_df, test_df


class GenreDataset(Dataset):
    """
    PyTorch Dataset for music genre classification.
    Loads pre-extracted features and applies augmentations.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        target_frames: int = 1292,
        mode: str = "train",
        augmentor: Optional[MusicAugmentor] = None,
        normalize: bool = True,
        global_mean: float = 0.0,
        global_std: float = 1.0,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.target_frames = target_frames
        self.mode = mode
        self.augmentor = augmentor if mode == "train" else None
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std

        # Validate features exist
        valid_mask = self.manifest["feature_path"].apply(os.path.exists)
        if not valid_mask.all():
            n_missing = (~valid_mask).sum()
            print(f"  [WARN] {n_missing} feature files missing, filtering out")
            self.manifest = self.manifest[valid_mask].reset_index(drop=True)

        # Build label encoding
        self.label_map = GENRE_TO_IDX

    def compute_stats(self, sample_frac: float = 0.25) -> Tuple[float, float]:
        """Compute global mean and std from a sample of the dataset."""
        paths = self.manifest["feature_path"].tolist()
        n = max(1, int(len(paths) * sample_frac))
        sampled = random.sample(paths, n)

        running_sum = 0.0
        running_sq_sum = 0.0
        count = 0

        for p in sampled:
            arr = np.load(p).astype(np.float32)
            arr = MusicAugmentor.pad_or_truncate(arr, self.target_frames, mode="eval")
            running_sum += arr.sum()
            running_sq_sum += (arr ** 2).sum()
            count += arr.size

        self.global_mean = running_sum / count
        self.global_std = math.sqrt(max(1e-12, running_sq_sum / count - self.global_mean ** 2))
        print(f"  Dataset stats: mean={self.global_mean:.6f}, std={self.global_std:.6f} "
              f"(from {n} samples)")
        return self.global_mean, self.global_std

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.manifest.iloc[idx]
        feat_path = row["feature_path"]
        genre = row["genre"]
        label = self.label_map[genre]

        # Load features
        feat = np.load(feat_path).astype(np.float32)

        # Pad or truncate
        feat = MusicAugmentor.pad_or_truncate(feat, self.target_frames, mode=self.mode)

        # Apply augmentations (train only)
        if self.augmentor is not None:
            feat = self.augmentor(feat)

        # Normalize
        if self.normalize:
            feat = (feat - self.global_mean) / (self.global_std + 1e-8)

        return torch.tensor(feat, dtype=torch.float32), label


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    audio_config: AudioConfig,
    aug_config: AugmentationConfig,
    batch_size: int = 8,
    num_workers: int = 2,
    balanced_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    Create train/val/test dataloaders with optional balanced sampling.

    Returns:
        (train_loader, val_loader, test_loader, global_mean, global_std)
    """
    target_frames = audio_config.target_frames
    augmentor = MusicAugmentor(AugConfig(
        time_mask_num=aug_config.time_mask_num,
        time_mask_max_pct=aug_config.time_mask_max_pct,
        freq_mask_num=aug_config.freq_mask_num,
        freq_mask_max_pct=aug_config.freq_mask_max_pct,
        time_shift_max_pct=aug_config.time_shift_max_pct,
        noise_prob=aug_config.noise_prob,
        noise_std=aug_config.noise_std,
        gain_prob=aug_config.gain_prob,
        gain_range=aug_config.gain_range,
        freq_shift_prob=aug_config.freq_shift_prob,
        freq_shift_max=aug_config.freq_shift_max,
    ))

    # Create datasets
    train_dataset = GenreDataset(
        train_df, target_frames=target_frames,
        mode="train", augmentor=augmentor
    )

    # Compute normalization stats from training set
    global_mean, global_std = train_dataset.compute_stats(sample_frac=0.3)

    val_dataset = GenreDataset(
        val_df, target_frames=target_frames,
        mode="eval", normalize=True,
        global_mean=global_mean, global_std=global_std
    )
    test_dataset = GenreDataset(
        test_df, target_frames=target_frames,
        mode="eval", normalize=True,
        global_mean=global_mean, global_std=global_std
    )

    # Balanced sampling for training
    sampler = None
    shuffle = True
    if balanced_sampling:
        labels = [GENRE_TO_IDX[row["genre"]] for _, row in train_df.iterrows()
                  if os.path.exists(row["feature_path"])]
        label_counts = Counter(labels)
        weights = [1.0 / label_counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, global_mean, global_std
