import os
import pandas as pd

FEATURE_DIR = r"C:\gtzan-music-genre-classification\data\gtzan\features"
MANIFEST_DIR = r"C:\gtzan-music-genre-classification\data\gtzan\manifests"

for name in ["train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]:
    path = os.path.join(MANIFEST_DIR, name)
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)

    fixed_paths = []
    for p in df["feature_path"]:
        fname = os.path.basename(p)   # strip old absolute path
        fixed_paths.append(os.path.join(FEATURE_DIR, fname))

    df["feature_path"] = fixed_paths

    out = path.replace(".csv", "_fixed.csv")
    df.to_csv(out, index=False)
    print(f"✅ Fixed: {out}")
