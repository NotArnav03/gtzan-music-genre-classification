# Music Genre Classification on GTZAN

This project implements a deep learning system for automatic music genre
classification using the GTZAN dataset.

## Architecture
- CNN-based spectral feature extraction
- Temporal modeling using BiLSTM and Transformer
- Attention-based pooling

## Results
- ~95% validation accuracy
- Macro F1-score ≈ 0.94

## Training
```bash
python src/train.py
