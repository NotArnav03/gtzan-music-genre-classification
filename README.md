# Music Genre Classification on GTZAN

This project implements a deep learning system for automatic music genre
classification using the GTZAN dataset.

## Architecture
- CNN-based spectral feature extraction
- Temporal modeling using BiLSTM and Transformer
- Attention-based pooling

## Results

### Accuracy Curve
![Accuracy Curve](artifacts/results/accuracy.png)

### Loss Curve
![Loss Curve](artifacts/results/loss.png)

### F1 Score
![F1 Score](artifacts/results/f1.png)

### Confusion Matrix
![Confusion Matrix](artifacts/results/confusion_matrix.png)

**Performance Summary**
- Validation Accuracy: ~95%
- Macro F1-score: ~0.94


## Training
```bash
python src/train.py
