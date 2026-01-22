import os
import torch
import numpy as np
from train_gtzan_ultimate import UltimateNet

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = r"C:\SoundModel\artifacts\gtzan_ultimate\best_ultimate.pth"
FEATURE_PATH = r"C:\SoundModel\data\gtzan\features\blues_blues.00001_chunk0.npy"  # <-- change this
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

# 🎭 Genre → Emotion mapping
GENRE_TO_EMOTION = {
    "blues": "sad / melancholic",
    "classical": "calm / peaceful",
    "country": "sentimental / nostalgic",
    "disco": "happy / energetic",
    "hiphop": "confident / excited",
    "jazz": "relaxed / chill",
    "metal": "angry / intense",
    "pop": "joyful / upbeat",
    "reggae": "carefree / cheerful",
    "rock": "energetic / passionate"
}

def load_checkpoint(path):
    print(f"Loading checkpoint: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)

def predict_genre(feature_path, checkpoint):
    label_map = checkpoint['label_map']
    inv_label_map = {v: k for k, v in label_map.items()}
    target_frames = checkpoint['target_frames']
    n_mels = checkpoint['n_mels']
    global_mean = checkpoint['global_mean']
    global_std = checkpoint['global_std']

    feat = np.load(feature_path).astype(np.float32)
    t = feat.shape[0]
    if t < target_frames:
        pad = np.zeros((target_frames - t, feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    elif t > target_frames:
        start = (t - target_frames) // 2
        feat = feat[start:start+target_frames, :]

    feat = (feat - global_mean) / (global_std + 1e-8)
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    num_classes = len(label_map)
    model = UltimateNet(time_frames=target_frames, n_mels=n_mels, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        logits = model(feat_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    genre = inv_label_map[pred_idx]
    return genre, confidence

if __name__ == "__main__":
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    genre, conf = predict_genre(FEATURE_PATH, checkpoint)
    emotion = GENRE_TO_EMOTION.get(genre, "unknown")

    print(f"\n🎵 Predicted Genre: {genre} ({conf*100:.2f}%)")
    print(f"💫 Mapped Emotion: {emotion}")
