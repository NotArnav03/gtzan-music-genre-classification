# Multi-Dataset Music Genre Classification with DPT²F

A novel approach to music genre classification using a **Dual-Path Temporal-Frequency Fusion Transformer (DPT²F)** trained on three harmonized datasets. Designed for research-grade reproducibility and Google Colab T4 GPU training.

## Key Contributions

1. **DPT²F Architecture** — Dual-path processing that explicitly separates temporal features (rhythm, tempo, beats) from frequency features (timbre, harmonics, instrumentation), then fuses them via bidirectional cross-attention
2. **Multi-Dataset Unified Training** — Harmonized 10-class taxonomy across GTZAN, FMA-small, and MagnaTagATune with artist-stratified splits to prevent data leakage
3. **Music-Aware Augmentation** — SpecAugment adapted for music with sub-band frequency masking, random EQ simulation, and mel-space pitch shifting
4. **Rigorous Evaluation** — Cross-dataset generalization metrics, ablation studies, and comparison with state-of-the-art methods

## Architecture

```
Audio (30s) → Log-Mel Spectrogram (1292 × 128)
                    │
              ┌─────┴─────┐
              │           │
        Temporal      Frequency
        Path CNN      Path CNN
        (time-axis)   (freq-axis)
              │           │
        Temporal      Spectral
        Transformer   Transformer
              │           │
              └─────┬─────┘
                    │
           Cross-Attention
           Fusion (T↔F)
                    │
           Fusion Transformer
                    │
           Attention Pooling
                    │
           Classification (10 genres)
```

## Datasets

| Dataset | Tracks | Source Genres | Mapped Genres | Duration |
|---------|--------|-------------|---------------|----------|
| GTZAN | 1,000 | 10 | 10 (1:1) | 30s clips |
| FMA-small | 8,000 | 8 | 5 mapped | 30s clips |
| MagnaTagATune | ~25,000 | 16 tags | 10 mapped | 29s clips |

### Unified 10-Class Taxonomy
Blues · Classical · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock

## Quick Start (Google Colab)

### 1. Clone and setup
```python
!git clone https://github.com/NotArnav03/gtzan-music-genre-classification.git
%cd gtzan-music-genre-classification
!pip install -r requirements.txt
```

### 2. Run notebooks in order
1. **`notebooks/01_data_preparation.py`** — Download datasets, extract features, create splits
2. **`notebooks/02_train_model.py`** — Train DPT²F model
3. **`notebooks/03_evaluation.py`** — Full evaluation & paper figures

## Project Structure

```
├── src/
│   ├── config.py            # Centralized configuration & genre taxonomy
│   ├── model.py             # DPT²F architecture
│   ├── dataset.py           # Multi-dataset loader & feature extraction
│   ├── augmentations.py     # Music-aware augmentation pipeline
│   ├── train.py             # Training loop (AMP + gradient accumulation)
│   ├── evaluate.py          # Metrics, confusion matrices, t-SNE
│   └── utils.py             # Utilities (checkpointing, schedulers, etc.)
├── notebooks/
│   ├── 01_data_preparation.py
│   ├── 02_train_model.py
│   └── 03_evaluation.py
├── HOW_IT_WORKS.md          # Layman's explanation of the model
├── requirements.txt
└── README.md
```

## Training Configuration (T4 GPU)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 8 × 4 accumulation = 32 effective | Fits T4 16GB VRAM |
| Mixed precision | FP16 via AMP | ~2× speedup on T4 |
| Learning rate | 1e-4 (cosine warmup) | AdamW with 5-epoch warmup |
| Gradient checkpointing | Enabled | Reduces VRAM ~40% |
| Model size | ~8M parameters | Lightweight but expressive |
| Label smoothing | 0.1 | Prevents overconfidence |
| Early stopping | Patience 15 | Monitors validation accuracy |

## Literature Foundation

Based on analysis of **32 papers (2020-2025)** identifying 6 major gaps in existing research:

1. No unified multi-dataset training with harmonized taxonomy
2. Temporal and frequency features treated as single stream
3. Speech-pretrained models used without music-specific adaptation
4. GTZAN artist effect still ignored in most papers
5. No knowledge distillation for efficient music classifiers
6. Augmentation strategies not optimized for music

See `implementation_plan.md` for the full literature review and gap analysis.

## Technologies

- **PyTorch** — Model architecture and training
- **Librosa** — Audio processing and feature extraction
- **scikit-learn** — Evaluation metrics and t-SNE
- **matplotlib** — Publication-quality visualizations
- **Google Colab** — Training environment (T4 GPU)

## Citation

```bibtex
@article{dpt2f2025,
  title={Multi-Dataset Music Genre Classification with Dual-Path
         Temporal-Frequency Fusion Transformer},
  author={},
  year={2025}
}
```

## License

MIT License — See LICENSE file for details.
