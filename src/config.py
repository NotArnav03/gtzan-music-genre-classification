"""
Configuration for Multi-Dataset Music Genre Classification.
All hyperparameters and paths centralized here.
"""
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio processing parameters."""
    sample_rate: int = 22050
    duration: float = 30.0          # seconds per clip
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 8000.0

    @property
    def target_frames(self) -> int:
        """Number of spectrogram frames for target duration."""
        return int(self.duration * self.sample_rate / self.hop_length) + 1  # ~1292 for 30s


@dataclass
class AugmentationConfig:
    """Music-aware augmentation parameters."""
    # SpecAugment
    time_mask_num: int = 2
    time_mask_max_pct: float = 0.10
    freq_mask_num: int = 2
    freq_mask_max_pct: float = 0.15

    # MixUp
    mixup_alpha: float = 0.3
    mixup_prob: float = 0.5

    # Time shift
    time_shift_max_pct: float = 0.03

    # Gaussian noise
    noise_prob: float = 0.3
    noise_std: float = 0.005

    # Random gain
    gain_prob: float = 0.3
    gain_range: tuple = (-6, 6)     # dB

    # Pitch-preserving frequency shift
    freq_shift_prob: float = 0.2
    freq_shift_max: int = 4         # mel bins


@dataclass
class ModelConfig:
    """DPT²F Model architecture parameters."""
    # Input
    n_mels: int = 128
    target_frames: int = 1292       # will be computed from AudioConfig

    # CNN Feature Extraction (per path)
    cnn_channels: List[int] = field(default_factory=lambda: [1, 32, 64, 128])
    cnn_kernel_temporal: tuple = (5, 1)     # convolution along time
    cnn_kernel_spectral: tuple = (1, 5)     # convolution along freq
    cnn_pool: tuple = (2, 2)
    cnn_dropout: float = 0.2

    # Transformer Encoder (per path)
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 512
    transformer_dropout: float = 0.2

    # Cross-Attention Fusion
    fusion_nhead: int = 4
    fusion_layers: int = 2
    fusion_dropout: float = 0.2

    # Classifier
    classifier_hidden: int = 256
    classifier_dropout: float = 0.4
    num_classes: int = 10


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for T4 GPU."""
    # Batch
    batch_size: int = 8              # actual batch on T4
    accumulation_steps: int = 4      # effective batch = 32
    num_workers: int = 2

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # Scheduler
    warmup_epochs: int = 5
    max_epochs: int = 100
    min_lr: float = 1e-6

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Regularization
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Reproducibility
    seed: int = 42

    # Logging
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1


@dataclass
class DataConfig:
    """Dataset paths and configuration."""
    # Base directory (on Colab, use /content/drive/MyDrive/...)
    base_dir: str = "/content/drive/MyDrive/music_genre_classification"

    # Raw audio directories
    gtzan_dir: str = ""
    fma_dir: str = ""
    mtt_dir: str = ""

    # Feature directories
    features_dir: str = ""

    # Manifest files
    manifests_dir: str = ""

    # Results
    results_dir: str = ""
    checkpoint_dir: str = ""

    def __post_init__(self):
        self.gtzan_dir = f"{self.base_dir}/data/gtzan"
        self.fma_dir = f"{self.base_dir}/data/fma_small"
        self.mtt_dir = f"{self.base_dir}/data/magnatagatune"
        self.features_dir = f"{self.base_dir}/features"
        self.manifests_dir = f"{self.base_dir}/manifests"
        self.results_dir = f"{self.base_dir}/results"
        self.checkpoint_dir = f"{self.base_dir}/checkpoints"


# ─── Unified Genre Taxonomy ────────────────────────────────────────

UNIFIED_GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

NUM_CLASSES = len(UNIFIED_GENRES)

GENRE_TO_IDX = {g: i for i, g in enumerate(UNIFIED_GENRES)}
IDX_TO_GENRE = {i: g for i, g in enumerate(UNIFIED_GENRES)}

# GTZAN mapping (direct 1:1)
GTZAN_GENRE_MAP = {
    "blues": "blues", "classical": "classical", "country": "country",
    "disco": "disco", "hiphop": "hiphop", "jazz": "jazz",
    "metal": "metal", "pop": "pop", "reggae": "reggae", "rock": "rock"
}

# FMA-small mapping (8 genres → 10 unified, some may not map)
FMA_GENRE_MAP = {
    "electronic": "disco",       # closest dance/electronic
    "experimental": None,        # exclude — no clear mapping
    "folk": "country",           # closest acoustic/roots
    "hip-hop": "hiphop",
    "instrumental": None,        # exclude — cross-genre
    "international": None,       # exclude — cross-genre
    "pop": "pop",
    "rock": "rock",
}

# MagnaTagATune tag → genre mapping (filter to genre-relevant tags)
MTT_TAG_MAP = {
    "blues": "blues",
    "classical": "classical", "classic": "classical", "opera": "classical",
    "country": "country",
    "dance": "disco", "techno": "disco", "electronic": "disco",
    "hip hop": "hiphop", "rap": "hiphop",
    "jazz": "jazz",
    "metal": "metal", "hard rock": "metal", "heavy": "metal",
    "pop": "pop",
    "reggae": "reggae",
    "rock": "rock", "alternative": "rock", "punk": "rock",
    "indie": "rock",
}

# Genre → Emotion mapping for interpretability
GENRE_EMOTIONS = {
    "blues":     {"emotion": "Melancholic",    "description": "Deep, soulful, introspective",     "color": "#4A5899"},
    "classical": {"emotion": "Serene",         "description": "Elegant, peaceful, refined",       "color": "#8B7355"},
    "country":   {"emotion": "Nostalgic",      "description": "Heartfelt, warm, sentimental",     "color": "#D4A574"},
    "disco":     {"emotion": "Euphoric",       "description": "Groovy, danceable, energetic",     "color": "#FF6B9D"},
    "hiphop":    {"emotion": "Confident",      "description": "Bold, rhythmic, expressive",       "color": "#FF9F1C"},
    "jazz":      {"emotion": "Sophisticated",  "description": "Smooth, improvisational, cool",    "color": "#2EC4B6"},
    "metal":     {"emotion": "Intense",        "description": "Powerful, aggressive, raw",        "color": "#E71D36"},
    "pop":       {"emotion": "Joyful",         "description": "Upbeat, catchy, bright",           "color": "#FF006E"},
    "reggae":    {"emotion": "Chill",          "description": "Laid-back, groovy, positive",      "color": "#06FFA5"},
    "rock":      {"emotion": "Energetic",      "description": "Dynamic, powerful, passionate",    "color": "#FB5607"},
}


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_configs():
    """Return all configuration objects."""
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig(
        n_mels=audio_cfg.n_mels,
        target_frames=audio_cfg.target_frames
    )
    train_cfg = TrainingConfig()
    aug_cfg = AugmentationConfig()
    data_cfg = DataConfig()
    return audio_cfg, model_cfg, train_cfg, aug_cfg, data_cfg
