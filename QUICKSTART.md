# Quick Start Guide - Music Genre Classification API

## 🚀 Getting Started in 2 Steps

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements_backend.txt
```

### Step 2: Start the API Server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

✅ **Verify**: Open http://localhost:8000/docs to see API documentation

### Test a Prediction

```bash
# Using curl
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/your/song.mp3"

# Or use the Swagger UI at http://localhost:8000/docs
```

---

## 📋 Prerequisites

- [ ] Python 3.8+ installed
- [ ] Model checkpoint at `C:\SoundModel\artifacts\gtzan_ultimate\best_ultimate.pth`
- [ ] Sample audio files for testing

---

## 🔧 Troubleshooting

**"Model checkpoint not found"**
- Update the model path in `backend/config.py` line 18
- Or set environment variable: `MODEL_CHECKPOINT_PATH=your\path\here`

**"CUDA not available" (running on CPU)**
- Normal if you don't have a GPU
- Model will run on CPU (slower but works)
- To force CPU: Set `DEVICE=cpu` in `backend/config.py`

**Port 8000 already in use**
```bash
uvicorn api:app --reload --port 8001
```

---

## 🎵 Supported Audio Formats

- ✅ MP3 (.mp3)
- ✅ WAV (.wav)
- ✅ FLAC (.flac)
- ✅ OGG (.ogg)
- ✅ M4A (.m4a)

**Max file size**: 50 MB

---

## 💡 Tips

- **Best results**: Use full songs (30-60 seconds minimum)
- **Faster inference**: Use GPU if available (CUDA)
- **Testing**: Sample audio files are typically in the data directory
