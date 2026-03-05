"""
Music-aware augmentation pipeline for spectrogram features.
Designed specifically for music genre classification, not speech.
"""
import numpy as np
import random
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AugConfig:
    """Augmentation parameters (mirror of config.AugmentationConfig)."""
    time_mask_num: int = 2
    time_mask_max_pct: float = 0.10
    freq_mask_num: int = 2
    freq_mask_max_pct: float = 0.15
    time_shift_max_pct: float = 0.03
    noise_prob: float = 0.3
    noise_std: float = 0.005
    gain_prob: float = 0.3
    gain_range: tuple = (-6, 6)
    freq_shift_prob: float = 0.2
    freq_shift_max: int = 4
    mixup_alpha: float = 0.3
    mixup_prob: float = 0.5


class MusicAugmentor:
    """
    Music-aware augmentation pipeline.

    Unlike standard SpecAugment (designed for speech), this pipeline
    considers music-specific characteristics:
    - Harmonic structure (frequency masking respects harmonic bands)
    - Temporal periodicity (time masking considers beat-scale patterns)
    - Timbral variation (random EQ-like gain adjustments)
    """

    def __init__(self, config: Optional[AugConfig] = None):
        self.cfg = config or AugConfig()

    def __call__(self, feat: np.ndarray, p: float = 1.0) -> np.ndarray:
        """
        Apply full augmentation pipeline.

        Args:
            feat: (T, F) spectrogram, typically (1292, 128)
            p: overall probability of applying augmentations

        Returns:
            Augmented spectrogram of same shape
        """
        if random.random() > p:
            return feat

        feat = feat.copy()  # don't modify original

        # 1. Time shift (slight temporal displacement)
        if random.random() < 0.5:
            feat = self.time_shift(feat)

        # 2. SpecAugment-style masking (music-adapted)
        if random.random() < 0.7:
            feat = self.spec_augment(feat)

        # 3. Additive noise (simulates recording conditions)
        if random.random() < self.cfg.noise_prob:
            feat = self.add_noise(feat)

        # 4. Random gain (simulates volume/EQ variation)
        if random.random() < self.cfg.gain_prob:
            feat = self.random_gain(feat)

        # 5. Frequency shift (pitch-like augmentation in mel space)
        if random.random() < self.cfg.freq_shift_prob:
            feat = self.frequency_shift(feat)

        return feat

    def spec_augment(self, feat: np.ndarray) -> np.ndarray:
        """
        Music-adapted SpecAugment.

        Differences from speech SpecAugment:
        - Frequency masks target musical sub-bands (bass/mid/treble)
        - Time masks are proportional to typical musical phrase lengths
        - Mask values use mean instead of zero (preserves energy distribution)
        """
        T, F = feat.shape
        fill_value = feat.mean()

        # Time masking
        max_t = max(1, int(self.cfg.time_mask_max_pct * T))
        for _ in range(self.cfg.time_mask_num):
            t = random.randint(1, max_t)
            start = random.randint(0, max(0, T - t))
            feat[start:start+t, :] = fill_value

        # Frequency masking — split into sub-bands for music
        # Low (0-F//3), Mid (F//3-2F//3), High (2F//3-F)
        max_f = max(1, int(self.cfg.freq_mask_max_pct * F))
        for _ in range(self.cfg.freq_mask_num):
            f = random.randint(1, max_f)
            # Choose a sub-band region with weighted probability
            # Music: more information in mid frequencies
            band = random.choices(
                ["low", "mid", "high"],
                weights=[0.2, 0.5, 0.3],
                k=1
            )[0]

            if band == "low":
                region_start, region_end = 0, F // 3
            elif band == "mid":
                region_start, region_end = F // 3, 2 * F // 3
            else:
                region_start, region_end = 2 * F // 3, F

            f = min(f, region_end - region_start)
            start = random.randint(region_start, max(region_start, region_end - f))
            feat[:, start:start+f] = fill_value

        return feat

    def time_shift(self, feat: np.ndarray) -> np.ndarray:
        """Circular time shift."""
        T = feat.shape[0]
        max_shift = max(1, int(self.cfg.time_shift_max_pct * T))
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            feat = np.roll(feat, shift, axis=0)
        return feat

    def add_noise(self, feat: np.ndarray) -> np.ndarray:
        """Add Gaussian noise scaled to feature magnitude."""
        noise = np.random.normal(0, 1, feat.shape).astype(np.float32)
        # Scale noise relative to feature std for magnitude-awareness
        feat_std = max(feat.std(), 1e-6)
        scaled_noise = noise * self.cfg.noise_std * feat_std
        return feat + scaled_noise

    def random_gain(self, feat: np.ndarray) -> np.ndarray:
        """
        Apply random frequency-band gain (like a random EQ).
        Simulates different playback/recording conditions.
        """
        F = feat.shape[1]
        lo, hi = self.cfg.gain_range

        # Random per-band gain curve (smooth)
        n_bands = 8
        gains_db = np.random.uniform(lo, hi, n_bands).astype(np.float32)
        # Interpolate to full frequency resolution
        x_bands = np.linspace(0, F - 1, n_bands)
        x_full = np.arange(F)
        gains_full = np.interp(x_full, x_bands, gains_db)

        # Convert dB to linear scale and apply
        gains_linear = 10 ** (gains_full / 20.0)
        feat = feat * gains_linear[np.newaxis, :]
        return feat

    def frequency_shift(self, feat: np.ndarray) -> np.ndarray:
        """
        Shift spectrogram along frequency axis (mel-space pitch shift).
        More musically meaningful than random frequency warping.
        """
        shift = random.randint(-self.cfg.freq_shift_max, self.cfg.freq_shift_max)
        if shift == 0:
            return feat

        result = np.zeros_like(feat)
        if shift > 0:
            result[:, shift:] = feat[:, :-shift]
        else:
            result[:, :shift] = feat[:, -shift:]
        return result

    @staticmethod
    def pad_or_truncate(
        feat: np.ndarray,
        target_frames: int,
        mode: str = "train"
    ) -> np.ndarray:
        """
        Pad or truncate spectrogram to target length.

        Args:
            feat: (T, F) spectrogram
            target_frames: target number of time frames
            mode: 'train' for random crop, 'eval' for center crop

        Returns:
            (target_frames, F) spectrogram
        """
        T, F = feat.shape

        if T < target_frames:
            # Pad with zeros (silence)
            pad = np.zeros((target_frames - T, F), dtype=feat.dtype)
            feat = np.concatenate([feat, pad], axis=0)
        elif T > target_frames:
            if mode == "train":
                start = random.randint(0, T - target_frames)
            else:
                start = (T - target_frames) // 2
            feat = feat[start:start + target_frames]

        return feat
