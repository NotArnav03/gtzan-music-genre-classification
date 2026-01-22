# Music Genre Classification on the GTZAN Dataset

This repository presents a deep learning–based approach for automatic music genre
classification using the GTZAN dataset. The work focuses on effective temporal
modeling of audio features and systematic evaluation following standard research
practices.

## Overview
Music genre classification is a fundamental task in Music Information Retrieval (MIR).
In this project, log-mel spectrogram representations are used as input to a deep
neural network that combines convolutional feature extraction with temporal
sequence modeling and attention-based pooling.

## Model Architecture
The proposed system consists of:
- Convolutional Neural Networks (CNNs) for spectral feature extraction
- Bi-directional LSTM layers for temporal dependency modeling
- Transformer encoder layers for long-range temporal attention
- Attention-based pooling for global feature aggregation
- Fully connected layers for final genre classification

## Dataset
Experiments are conducted on the **GTZAN music genre dataset**, which contains
audio tracks spanning ten music genres. Due to licensing restrictions, audio files
and extracted features are not included in this repository.

## Training Strategy
- Log-mel spectrogram feature extraction
- Spectrogram-level data augmentation (SpecAugment, MixUp)
- AdamW optimizer with cosine learning rate scheduling
- Early stopping based on validation performance
- Evaluation using accuracy and macro-averaged F1 score

## Results
The proposed model achieves strong and balanced performance across genres:

- Validation Accuracy: **~95%**
- Macro F1-score: **~0.94**

### Training Curves and Confusion Matrix
(See visualizations below for training dynamics and class-wise performance.)

![Accuracy Curve](artifacts/results/accuracy.png)
![Loss Curve](artifacts/results/loss.png)
![F1 Score](artifacts/results/f1.png)
![Confusion Matrix](artifacts/results/confusion_matrix.png)

## Evaluation
Performance is evaluated using:
- Overall classification accuracy
- Macro-averaged precision, recall, and F1-score
- Confusion matrix analysis for class-wise behavior

## Reproducibility
This repository contains all code required to reproduce the training and evaluation
pipeline, excluding dataset files. Feature paths and dataset handling are designed
to allow local experimentation without distributing copyrighted data.

## License
This project is released under the MIT License.
