"""
Audio processing utilities for feature extraction.
"""
import librosa
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class AudioProcessor:
    """Handles audio loading and feature extraction."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel bands
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def load_audio(self, file_path: Path, duration: Optional[float] = None, max_duration: float = 30.0) -> np.ndarray:
        """
        Load audio file with optimized duration for faster processing.
        
        Args:
            file_path: Path to audio file
            duration: Maximum duration to load (None = use max_duration)
            max_duration: Default maximum duration for fast processing (30s)
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            # Limit to max_duration for faster processing (30s is enough for genre classification)
            load_duration = duration if duration is not None else max_duration
            
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                duration=load_duration,
                mono=True,
                res_type='kaiser_fast'  # Faster resampling
            )
            return audio
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {str(e)}")
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel spectrogram from audio.
        IMPORTANT: Must match training extraction exactly!
        
        Args:
            audio: Audio waveform
            
        Returns:
            Log-mel spectrogram of shape (time_frames, n_mels)
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale using np.log (MATCHES TRAINING!)
        # NOTE: Training used np.log(mel + 1e-6), NOT librosa.power_to_db!
        log_mel_spec = np.log(mel_spec + 1e-6)
        
        # Transpose to (time, freq) format
        log_mel_spec = log_mel_spec.T
        
        return log_mel_spec.astype(np.float32)
    
    def extract_features(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Extract features from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (log_mel_spectrogram, duration_seconds)
        """
        # Load audio
        audio = self.load_audio(file_path)
        
        # Get duration
        duration = librosa.get_duration(y=audio, sr=self.sample_rate)
        
        # Extract mel spectrogram
        features = self.extract_mel_spectrogram(audio)
        
        return features, int(duration)
    
    @staticmethod
    def pad_or_truncate(
        features: np.ndarray,
        target_frames: int,
        mode: str = 'center'
    ) -> np.ndarray:
        """
        Pad or truncate features to target length.
        
        Args:
            features: Feature matrix of shape (time, freq)
            target_frames: Target number of time frames
            mode: Truncation mode ('center', 'random', 'start', 'end')
            
        Returns:
            Padded or truncated features
        """
        current_frames = features.shape[0]
        
        if current_frames < target_frames:
            # Pad with zeros
            pad_amount = target_frames - current_frames
            pad = np.zeros((pad_amount, features.shape[1]), dtype=features.dtype)
            features = np.concatenate([features, pad], axis=0)
        elif current_frames > target_frames:
            # Truncate
            if mode == 'center':
                start = (current_frames - target_frames) // 2
            elif mode == 'start':
                start = 0
            elif mode == 'end':
                start = current_frames - target_frames
            else:  # random
                start = np.random.randint(0, current_frames - target_frames + 1)
            
            features = features[start:start + target_frames, :]
        
        return features
    
    @staticmethod
    def normalize_features(
        features: np.ndarray,
        mean: float,
        std: float
    ) -> np.ndarray:
        """
        Normalize features using provided statistics.
        
        Args:
            features: Feature matrix
            mean: Global mean for normalization
            std: Global std for normalization
            
        Returns:
            Normalized features
        """
        return (features - mean) / (std + 1e-8)
