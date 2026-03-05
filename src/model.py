"""
DPT²F: Dual-Path Temporal-Frequency Fusion Transformer
for Music Genre Classification.

Novel architecture that explicitly separates temporal and frequency feature
processing through dual parallel paths, then fuses them via cross-attention.
This mirrors how musicians perceive music — rhythm/tempo (temporal) and
timbre/harmony (frequency) are processed independently then integrated.

Paper: "Multi-Dataset Music Genre Classification with Dual-Path
       Temporal-Frequency Fusion Transformer"

Architecture:
    Input (B, T, F) log-mel spectrogram
        ↓
    ┌──────────┐  ┌──────────┐
    │ Temporal  │  │ Frequency│
    │ Path CNN  │  │ Path CNN │
    │ (1D time) │  │ (1D freq)│
    │     ↓     │  │     ↓    │
    │ Temporal  │  │ Spectral │
    │ Trans.Enc │  │ Trans.Enc│
    └─────┬─────┘  └─────┬────┘
          │              │
          ▼              ▼
    ┌─────────────────────────┐
    │   Cross-Attention Fusion │
    │   (T→F and F→T queries)  │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │   Fusion Transformer     │
    │   Encoder (2 layers)     │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │   Attention Pooling →    │
    │   Classification Head    │
    └─────────────────────────┘
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ─── Building Blocks ────────────────────────────────────────────────

class ConvBlock1D(nn.Module):
    """1D Convolutional block with BatchNorm, GELU, and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        pool_size: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttention(nn.Module):
    """
    Bidirectional cross-attention between temporal and frequency paths.
    Temporal features query frequency features and vice versa.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        # T → F cross-attention (temporal queries, frequency keys/values)
        self.cross_attn_tf = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # F → T cross-attention (frequency queries, temporal keys/values)
        self.cross_attn_ft = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm_t = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)

        # Feed-forward for fused output
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        temporal: torch.Tensor,
        frequency: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            temporal: (B, T_len, D) - temporal path features
            frequency: (B, F_len, D) - frequency path features

        Returns:
            fused: (B, T_len + F_len, D) - fused cross-attended features
        """
        # T queries F
        t_enhanced, _ = self.cross_attn_tf(
            query=temporal, key=frequency, value=frequency
        )
        t_enhanced = self.norm_t(temporal + t_enhanced)

        # F queries T
        f_enhanced, _ = self.cross_attn_ft(
            query=frequency, key=temporal, value=temporal
        )
        f_enhanced = self.norm_f(frequency + f_enhanced)

        # Concatenate along sequence dimension
        fused = torch.cat([t_enhanced, f_enhanced], dim=1)  # (B, T+F, D)

        return fused


class AttentionPooling(nn.Module):
    """Attention-based pooling to aggregate sequence into single vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        attn_weights = self.attention(x)           # (B, L, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (attn_weights * x).sum(dim=1)    # (B, D)
        return context


# ─── Main Model ─────────────────────────────────────────────────────

class DPT2F(nn.Module):
    """
    Dual-Path Temporal-Frequency Fusion Transformer (DPT²F).

    Novel architecture for music genre classification that:
    1. Processes temporal and frequency dimensions through separate CNN+Transformer paths
    2. Fuses them via bidirectional cross-attention
    3. Refines with a fusion Transformer encoder
    4. Aggregates with attention pooling for classification

    Designed to fit within 6GB VRAM on T4 GPU with gradient checkpointing.
    """

    def __init__(
        self,
        n_mels: int = 128,
        target_frames: int = 1292,
        # CNN params
        cnn_channels: list = None,
        cnn_dropout: float = 0.2,
        # Transformer params
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        transformer_dropout: float = 0.2,
        # Fusion params
        fusion_nhead: int = 4,
        fusion_layers: int = 2,
        fusion_dropout: float = 0.2,
        # Classifier params
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.4,
        num_classes: int = 10,
        # Optimization
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [1, 32, 64, 128]

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.d_model = d_model
        self.n_mels = n_mels
        self.target_frames = target_frames

        # ─── Temporal Path ─────────────────────────────────────────
        # Process along time axis: input (B, F, T) → conv1d along T
        self.temporal_cnn = nn.Sequential(
            ConvBlock1D(n_mels, cnn_channels[1], kernel_size=7, pool_size=4, dropout=cnn_dropout),
            ConvBlock1D(cnn_channels[1], cnn_channels[2], kernel_size=5, pool_size=4, dropout=cnn_dropout),
            ConvBlock1D(cnn_channels[2], cnn_channels[3], kernel_size=3, pool_size=2, dropout=cnn_dropout),
        )

        # Compute temporal path output size
        with torch.no_grad():
            dummy_t = torch.zeros(1, n_mels, target_frames)
            t_out = self.temporal_cnn(dummy_t)
            self.temporal_seq_len = t_out.shape[2]
            temporal_feat_dim = t_out.shape[1]

        # Project to d_model
        self.temporal_proj = nn.Linear(temporal_feat_dim, d_model)
        self.temporal_pos = PositionalEncoding(d_model, max_len=self.temporal_seq_len + 100)

        # Temporal Transformer encoder
        t_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            t_enc_layer, num_layers=num_encoder_layers
        )

        # ─── Frequency (Spectral) Path ─────────────────────────────
        # Process along frequency axis: input (B, T, F) → conv1d along F
        self.spectral_cnn = nn.Sequential(
            ConvBlock1D(target_frames, cnn_channels[1], kernel_size=7, pool_size=2, dropout=cnn_dropout),
            ConvBlock1D(cnn_channels[1], cnn_channels[2], kernel_size=5, pool_size=2, dropout=cnn_dropout),
            ConvBlock1D(cnn_channels[2], cnn_channels[3], kernel_size=3, pool_size=2, dropout=cnn_dropout),
        )

        # Compute spectral path output size
        with torch.no_grad():
            dummy_f = torch.zeros(1, target_frames, n_mels)
            f_out = self.spectral_cnn(dummy_f)
            self.spectral_seq_len = f_out.shape[2]
            spectral_feat_dim = f_out.shape[1]

        # Project to d_model
        self.spectral_proj = nn.Linear(spectral_feat_dim, d_model)
        self.spectral_pos = PositionalEncoding(d_model, max_len=self.spectral_seq_len + 100)

        # Spectral Transformer encoder
        s_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.spectral_transformer = nn.TransformerEncoder(
            s_enc_layer, num_layers=num_encoder_layers
        )

        # ─── Cross-Attention Fusion ────────────────────────────────
        self.cross_attention = CrossAttention(
            d_model=d_model, nhead=fusion_nhead, dropout=fusion_dropout
        )

        # ─── Fusion Transformer ────────────────────────────────────
        fusion_total_len = self.temporal_seq_len + self.spectral_seq_len
        self.fusion_pos = PositionalEncoding(d_model, max_len=fusion_total_len + 100)

        f_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=fusion_nhead,
            dim_feedforward=dim_feedforward,
            dropout=fusion_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            f_enc_layer, num_layers=fusion_layers
        )

        # ─── Attention Pooling + Classifier ────────────────────────
        self.attention_pool = AttentionPooling(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout * 0.5),
            nn.Linear(classifier_hidden // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _temporal_path(self, x: torch.Tensor) -> torch.Tensor:
        """Process through temporal path."""
        # x: (B, T, F) → transpose to (B, F, T) for 1D conv along time
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.temporal_cnn(x)  # (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C)
        x = self.temporal_proj(x)  # (B, T', D)
        x = self.temporal_pos(x)
        x = self.temporal_transformer(x)  # (B, T', D)
        return x

    def _spectral_path(self, x: torch.Tensor) -> torch.Tensor:
        """Process through spectral/frequency path."""
        # x: (B, T, F) → use directly, conv1d along freq axis
        x = self.spectral_cnn(x)  # (B, C, F')
        x = x.transpose(1, 2)  # (B, F', C)
        x = self.spectral_proj(x)  # (B, F', D)
        x = self.spectral_pos(x)
        x = self.spectral_transformer(x)  # (B, F', D)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, F) log-mel spectrogram, e.g. (B, 1292, 128)

        Returns:
            logits: (B, num_classes) classification logits
        """
        # ─── Dual Path Processing ──────────────────────────────────
        if self.use_gradient_checkpointing and self.training:
            temporal_feat = grad_checkpoint(
                self._temporal_path, x, use_reentrant=False
            )
            spectral_feat = grad_checkpoint(
                self._spectral_path, x, use_reentrant=False
            )
        else:
            temporal_feat = self._temporal_path(x)   # (B, T', D)
            spectral_feat = self._spectral_path(x)   # (B, F', D)

        # ─── Cross-Attention Fusion ────────────────────────────────
        fused = self.cross_attention(temporal_feat, spectral_feat)  # (B, T'+F', D)

        # ─── Fusion Transformer ────────────────────────────────────
        fused = self.fusion_pos(fused)
        if self.use_gradient_checkpointing and self.training:
            fused = grad_checkpoint(
                self.fusion_transformer, fused, use_reentrant=False
            )
        else:
            fused = self.fusion_transformer(fused)    # (B, T'+F', D)

        # ─── Attention Pooling + Classification ────────────────────
        pooled = self.attention_pool(fused)           # (B, D)
        logits = self.classifier(pooled)              # (B, num_classes)

        return logits

    def get_attention_weights(self, x: torch.Tensor) -> dict:
        """
        Get attention weights for interpretability.
        Useful for visualization in the research paper.
        """
        with torch.no_grad():
            temporal_feat = self._temporal_path(x)
            spectral_feat = self._spectral_path(x)

            # Get cross-attention weights
            _, t2f_weights = self.cross_attention.cross_attn_tf(
                query=temporal_feat, key=spectral_feat, value=spectral_feat
            )
            _, f2t_weights = self.cross_attention.cross_attn_ft(
                query=spectral_feat, key=temporal_feat, value=temporal_feat
            )

            return {
                "temporal_to_freq": t2f_weights,
                "freq_to_temporal": f2t_weights,
            }


def build_model(
    n_mels: int = 128,
    target_frames: int = 1292,
    num_classes: int = 10,
    d_model: int = 256,
    use_gradient_checkpointing: bool = True,
    **kwargs
) -> DPT2F:
    """Factory function to build DPT²F model."""
    model = DPT2F(
        n_mels=n_mels,
        target_frames=target_frames,
        num_classes=num_classes,
        d_model=d_model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        **kwargs
    )
    return model


def model_summary(model: DPT2F) -> str:
    """Generate a summary of the model architecture."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        "=" * 60,
        "DPT²F: Dual-Path Temporal-Frequency Fusion Transformer",
        "=" * 60,
        f"Input: ({model.target_frames}, {model.n_mels}) log-mel spectrogram",
        f"Temporal path seq length: {model.temporal_seq_len}",
        f"Spectral path seq length: {model.spectral_seq_len}",
        f"d_model: {model.d_model}",
        f"Total parameters: {total:,} ({total/1e6:.2f}M)",
        f"Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)",
        "=" * 60,
    ]
    return "\n".join(lines)
