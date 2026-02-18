"""
Calculate expected audio duration for target frames
"""
import librosa

# Parameters from our backend
sample_rate = 22050
n_fft = 2048
hop_length = 512
target_frames = 862

# Calculate expected audio duration
# frames = (samples - n_fft) / hop_length + 1
# samples = (frames - 1) * hop_length + n_fft
expected_samples = (target_frames - 1) * hop_length + n_fft
expected_duration = expected_samples / sample_rate

print(f"Target frames: {target_frames}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Hop length: {hop_length}")
print(f"N_FFT: {n_fft}")
print(f"\nExpected audio length:")
print(f"  Samples: {expected_samples}")
print(f"  Duration: {expected_duration:.2f} seconds")

# Test with actual librosa
import numpy as np
test_audio = np.random.randn(expected_samples)
mel_spec = librosa.feature.melspectrogram(
    y=test_audio,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=256
)
print(f"\nActual mel spec shape: {mel_spec.shape}")
print(f"  (Expected: (256, {target_frames}))")

if mel_spec.shape[1] == target_frames:
    print("\n✅ MATCH! Audio duration calculation is correct!")
else:
    print(f"\n❌ MISMATCH! Got {mel_spec.shape[1]} frames, expected {target_frames}")
