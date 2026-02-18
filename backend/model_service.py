"""
Model serving layer for music genre classification.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import threading
import time

from config import settings


# Import model architecture from existing training code
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), padding=1, pool=(2,2), dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(pool),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)


class UltimateNet(nn.Module):
    def __init__(self, time_frames, n_mels, num_classes, lstm_hidden=256, transformer_heads=4, transformer_layers=2, dropout=0.3):
        super().__init__()
        # conv stack
        self.conv1 = ConvBlock(1, 32, dropout=0.15)
        self.conv2 = ConvBlock(32, 64, dropout=0.2)
        self.conv3 = ConvBlock(64, 128, dropout=0.25)
        self.conv4 = ConvBlock(128, 256, dropout=0.3)

        # compute shape after convs
        with torch.no_grad():
            dummy = torch.zeros(1,1,time_frames,n_mels)
            x = self._forward_convs(dummy)
            _, c, t, f = x.shape
            transformer_input_dim = c * f
            transformer_time = t

        # BiLSTM
        self.lstm = nn.LSTM(input_size=transformer_input_dim, hidden_size=lstm_hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)

        # small Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden*2, nhead=transformer_heads, dim_feedforward=512, dropout=0.2, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # attention pooling
        self.attn_w = nn.Linear(lstm_hidden*2, 1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def _forward_convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward(self, x):
        # x: (B, T, F)
        x = x.unsqueeze(1)              # (B,1,T,F)
        x = self._forward_convs(x)      # (B, C, T', F')
        b, c, t, f = x.shape
        x = x.permute(0,2,1,3).contiguous().view(b, t, c * f)  # (B, T', C*F)
        # LSTM
        lstm_out, _ = self.lstm(x)      # (B, T', H*2)
        # Transformer expects (T, B, C)
        trans_in = lstm_out.permute(1,0,2)
        trans_out = self.transformer(trans_in).permute(1,0,2)  # (B,T',H*2)
        # attention pooling
        attn_logits = self.attn_w(trans_out)  # (B, T', 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = (attn_weights * trans_out).sum(dim=1)  # (B, H*2)
        logits = self.classifier(context)
        return logits


class ModelService:
    """Singleton service for model inference."""
    
    _instance: Optional['ModelService'] = None
    _lock = threading.Lock()
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize model service.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        
        # Set device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Load checkpoint
        self._load_model()
    
    @classmethod
    def get_instance(cls, checkpoint_path: Optional[str] = None, device: Optional[str] = None) -> 'ModelService':
        """
        Get singleton instance of ModelService.
        
        Args:
            checkpoint_path: Path to model checkpoint (only used on first call)
            device: Device to run on (only used on first call)
            
        Returns:
            ModelService instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    checkpoint_path = checkpoint_path or settings.model_checkpoint_path
                    device = device or settings.device
                    cls._instance = cls(checkpoint_path, device)
        return cls._instance
    
    def _load_model(self):
        """Load model checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Extract metadata
        self.label_map = checkpoint['label_map']
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.target_frames = checkpoint['target_frames']
        self.n_mels = checkpoint['n_mels']
        self.global_mean = checkpoint['global_mean']
        self.global_std = checkpoint['global_std']
        
        # Debug: Print loaded statistics
        print(f"📊 Loaded normalization stats:")
        print(f"   Global mean: {self.global_mean:.6f}")
        print(f"   Global std: {self.global_std:.6f}")
        print(f"   Genres: {sorted(self.label_map.keys())}")
        
        # Check if normalization stats are invalid (identity transform)
        if abs(self.global_mean) < 1e-6 and abs(self.global_std - 1.0) < 1e-6:
            print(f"⚠️  WARNING: Checkpoint has identity normalization stats!")
            print(f"   This suggests model was trained WITHOUT normalization")
            print(f"   Disabling normalization for inference...")
            self.use_normalization = False
        else:
            self.use_normalization = True
        
        # Initialize model
        num_classes = len(self.label_map)
        self.model = UltimateNet(
            time_frames=self.target_frames,
            n_mels=self.n_mels,
            num_classes=num_classes
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Apply optimizations for faster inference
        if self.device.type == 'cuda':
            # Enable cudnn benchmarking for faster inference
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Try to compile model for faster inference (PyTorch 2.0+) - completely optional
        try:
            import sys
            if hasattr(torch, 'compile') and sys.version_info >= (3, 8) and sys.version_info < (3, 13):
                # Only attempt compilation on supported Python versions
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print(f"   ⚡ Model compiled for faster inference")
        except Exception as e:
            # Compilation failed - not critical, model will still work
            print(f"   ℹ️  Model compilation skipped (using eager mode): {type(e).__name__}")
            pass
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Classes: {num_classes}")
        print(f"   Target frames: {self.target_frames}")
        print(f"   N mels: {self.n_mels}")
    
    @torch.inference_mode()  # Faster than @torch.no_grad()
    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Perform inference on features (optimized for speed).
        
        Args:
            features: Log-mel spectrogram of shape (time, n_mels)
            
        Returns:
            Tuple of (predicted_genre, confidence, all_probabilities, inference_time)
        """
        # Ensure features are the right shape
        if features.shape != (self.target_frames, self.n_mels):
            raise ValueError(
                f"Expected features of shape ({self.target_frames}, {self.n_mels}), "
                f"got {features.shape}"
            )
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Debug: Print feature statistics
        print(f"🔍 Feature stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}, std={features.std():.3f}")
        
        # Inference
        start_time = time.time()
        logits = self.model(features_tensor)
        inference_time = time.time() - start_time
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Debug: Print top predictions
        top_3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
        print(f"🎯 Top predictions: {[(self.inv_label_map[i], f'{p*100:.1f}%') for i, p in top_3]}")
        
        # Get prediction
        pred_idx = int(np.argmax(probs))
        genre = self.inv_label_map[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Create probability dictionary
        all_probs = {
            self.inv_label_map[i]: float(probs[i])
            for i in range(len(probs))
        }
        
        return genre, confidence, all_probs, inference_time
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'num_classes': len(self.label_map),
            'genres': sorted(self.label_map.keys(), key=lambda x: self.label_map[x]),
            'model_architecture': 'CNN + BiLSTM + Transformer + Attention',
            'target_frames': self.target_frames,
            'n_mels': self.n_mels,
            'device': str(self.device)
        }
