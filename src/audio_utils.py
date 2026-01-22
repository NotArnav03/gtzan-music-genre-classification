# gtzan_emotion_wrapper.py
import torch
import torch.nn as nn
import numpy as np
import librosa

GENRE_TO_EMOTION = {
    "blues": "sadness",
    "classical": "tenderness",
    "country": "nostalgia",
    "disco": "joyful_activation",
    "hiphop": "power",
    "jazz": "calmness",
    "metal": "tension",
    "pop": "joyful_activation",
    "reggae": "calmness",
    "rock": "power"
}

class GTZANEmotionWrapper:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)

        ck = torch.load(model_path, map_location=self.device)

        # metadata
        self.n_mels = ck.get("n_mels", 128)
        self.target_frames = ck.get("target_frames", 431)
        self.label_map = ck.get("label_map", [])

        self.model = self._build_model(num_classes=len(self.label_map))
        self.model.load_state_dict(ck["model_state_dict"], strict=False)
        self.model.to(self.device).eval()

        self.global_mean = ck.get("global_mean", None)
        self.global_std = ck.get("global_std", None)

    def _build_model(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(32*4*4, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def _mel(self, audio_path):
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, n_fft=2048, hop_length=512
        )
        mel = np.log(mel + 1e-6)

        if self.global_mean is not None:
            mel = (mel - self.global_mean) / (self.global_std + 1e-6)

        # pad/crop
        if mel.shape[1] < self.target_frames:
            pad = np.zeros((self.n_mels, self.target_frames - mel.shape[1]))
            mel = np.hstack([mel, pad])
        else:
            mel = mel[:, :self.target_frames]

        mel = mel.astype(np.float32)
        return torch.tensor(mel).unsqueeze(0).unsqueeze(0)

    def predict(self, audio_path):
        mel = self._mel(audio_path).to(self.device)

        with torch.no_grad():
            logits = self.model(mel)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        genre = self.label_map[idx]

        # emotion mapped from genre
        emotion = GENRE_TO_EMOTION.get(genre.lower(), "unknown")

        return {
            "genre": genre,
            "confidence": float(probs[idx]),
            "emotion": emotion,
            "probabilities": {
                self.label_map[i]: float(p) for i, p in enumerate(probs)
            }
        }
