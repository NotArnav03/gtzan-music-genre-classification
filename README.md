# Music Genre Classification API

A **production-grade REST API** for automatic music genre classification using deep learning. Built with FastAPI and a CNN + BiLSTM + Transformer model trained on the GTZAN dataset.

## 🎵 Overview

Music genre classification is a fundamental task in Music Information Retrieval (MIR). This application provides:

- **Deep Learning Model**: CNN + BiLSTM + Transformer architecture achieving ~95% accuracy
- **Professional Backend**: FastAPI-based REST API with proper error handling and validation
- **Emotion Mapping**: Genre predictions include associated emotions and descriptions

## 🏗️ Architecture

- **Model Serving**: Singleton pattern for efficient inference
- **Audio Processing**: Multi-format support (MP3, WAV, FLAC, OGG, M4A)
- **Feature Extraction**: Log-mel spectrograms matching training pipeline
- **REST API**: OpenAPI/Swagger documentation at `/docs`

## 📊 Model Architecture

The deep learning model consists of:
- Convolutional Neural Networks (CNNs) for spectral feature extraction
- Bi-directional LSTM layers for temporal dependency modeling
- Transformer encoder layers for long-range temporal attention
- Attention-based pooling for global feature aggregation
- Fully connected layers for final genre classification

**Performance**:
- Validation Accuracy: **~95%**
- Macro F1-score: **~0.94**

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Trained model checkpoint at `C:\SoundModel\artifacts\gtzan_ultimate\best_ultimate.pth`

### Setup

1. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements_backend.txt
   ```

2. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env to customize settings
   ```

3. **Start the API server**:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access API documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📖 API Documentation

### Endpoints

#### `POST /predict`
Upload an audio file and get genre prediction.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Audio file (mp3, wav, flac, ogg, m4a)

**Response**:
```json
{
  "success": true,
  "genre": "jazz",
  "confidence": 0.94,
  "emotion": "Sophisticated",
  "emotion_description": "Smooth, improvisational, and cool",
  "color": "#2EC4B6",
  "all_probabilities": {
    "jazz": 0.94,
    "blues": 0.03,
    "classical": 0.02
  },
  "processing_time": 1.23
}
```

#### `GET /health`
Check API health status.

#### `GET /model/info`
Get model information and metadata.

## 🎨 Features

- ✅ Multi-format audio support (MP3, WAV, FLAC, OGG, M4A)
- ✅ Efficient model inference with singleton pattern
- ✅ Proper error handling and validation
- ✅ CORS configuration for frontend integration
- ✅ Comprehensive API documentation (Swagger + ReDoc)
- ✅ Environment-based configuration
- ✅ Genre-to-emotion mapping with colors
- ✅ Thread-safe inference handling

## 🛠️ Configuration

Edit `.env` or set environment variables:

```env
PORT=8000
DEBUG=false
MODEL_CHECKPOINT_PATH=C:\SoundModel\artifacts\gtzan_ultimate\best_ultimate.pth
DEVICE=cuda
CORS_ORIGINS=http://localhost:3000
```

## 📁 Project Structure

```
gtzan-music-genre-classification/
├── backend/
│   ├── api.py                 # FastAPI application
│   ├── model_service.py       # Model serving layer
│   ├── audio_processor.py     # Audio processing utilities
│   ├── schemas.py             # Pydantic models
│   ├── config.py              # Configuration management
│   └── requirements_backend.txt
├── src/
│   ├── train.py               # Training script
│   ├── infer.py               # Inference script
│   └── audio_utils.py         # Audio utilities
├── data/                      # Dataset directory
├── artifacts/                 # Training artifacts
└── README.md
```

## 📝 Dataset

Experiments are conducted on the **GTZAN music genre dataset**, which contains audio tracks spanning ten music genres:
Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock.

Due to licensing restrictions, audio files are not included in this repository.

## 🎓 Training

```bash
python src/train.py
```

Training features:
- Log-mel spectrogram feature extraction
- SpecAugment and MixUp data augmentation
- AdamW optimizer with OneCycleLR scheduling
- Early stopping based on validation performance

## 📄 License

This project is released under the MIT License.

---

**Built with ❤️ using FastAPI, PyTorch, and Librosa**
