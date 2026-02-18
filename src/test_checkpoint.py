"""
Quick test to verify model checkpoint integrity
"""
import torch
import numpy as np
import sys
sys.path.append('.')
from train import UltimateNet

# Load checkpoint
checkpoint_path = r"C:\SoundModel\artifacts\gtzan_ultimate\best_ultimate.pth"
print(f"Loading checkpoint: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("\n✅ Checkpoint loaded successfully!")
    
    # Check what's in the checkpoint
    print("\n📊 Checkpoint contents:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"   {key}: <state_dict with {len(checkpoint[key])} keys>")
        else:
            print(f"   {key}: {checkpoint[key]}")
    
    # Try to load model
    print("\n🔧 Testing model initialization...")
    target_frames = checkpoint['target_frames']
    n_mels = checkpoint['n_mels']
    num_classes = len(checkpoint['label_map'])
    
    model = UltimateNet(
        time_frames=target_frames,
        n_mels=n_mels,
        num_classes=num_classes
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model initialized successfully!")
    
    # Test forward pass with random data
    print("\n🧪 Testing forward pass with random features...")
    random_features = np.random.randn(1, target_frames, n_mels).astype(np.float32)
    features_tensor = torch.tensor(random_features)
    
    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    
    print(f"✅ Forward pass successful!")
    print(f"   Output shape: {logits.shape}")
    print(f"   Probabilities sum: {probs.sum():.6f}")
   
    # Show predicted probabilities
    inv_label_map = {v: k for k, v in checkpoint['label_map'].items()}
    print(f"\n🎲 Predictions for random input:")
    for i, p in enumerate(probs):
        print(f"   {inv_label_map[i]}: {p*100:.2f}%")
    
    print("\n" + "="*50)
    print("CONCLUSION: Model checkpoint appears to be valid")
    print("="*50)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
