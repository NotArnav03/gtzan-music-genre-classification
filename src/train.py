# train_gtzan_ultimate.py
import os
import math
import time
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def fix_manifest_paths(manifest_path, feature_dir):
    df = pd.read_csv(manifest_path)
    corrected_paths = []

    for path in df['feature_path']:
        # Extract just the filename (since all files are in one folder)
        filename = os.path.basename(path)
        # Build correct absolute path
        corrected_path = os.path.join(feature_dir, filename)
        corrected_paths.append(corrected_path)

    df['feature_path'] = corrected_paths
    fixed_manifest = manifest_path.replace(".csv", "_fixed.csv")
    df.to_csv(fixed_manifest, index=False)
    print(f"✅ Fixed manifest saved to: {fixed_manifest}")
    return fixed_manifest

# ----------------------- USER CONFIG -----------------------
FEATURE_DIR = r"C:\SoundModel\data\gtzan\features"       # directory containing .npy features
MANIFEST_DIR = r"C:\SoundModel\data\gtzan\manifests"     # directory containing train/val/test manifests
TRAIN_MANIFEST = os.path.join(MANIFEST_DIR, "train_manifest.csv")
VAL_MANIFEST   = os.path.join(MANIFEST_DIR, "val_manifest.csv")
TEST_MANIFEST  = os.path.join(MANIFEST_DIR, "test_manifest.csv")  # optional

ARTIFACTS_DIR = r"C:\SoundModel\artifacts\gtzan_ultimate"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Audio / feature params (should match your extracted .npy)
TARGET_FRAMES = None     # None = autodetect from first feature
N_MELS = None            # None = autodetect
# Training hyperparams
BATCH_SIZE = 32
NUM_EPOCHS = 120
MAX_LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12            # early stopping patience on val metric
CLIP_GRAD = 5.0
MIXUP_ALPHA = 0.3
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 0          # Windows: use 0
SEED = 42
USE_AMP = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------------------- Dataset (self-contained) --------------------
class GTZANFeatureDataset(Dataset):
    """
    Loads saved log-mel .npy features (shape: time_frames x n_mels) from manifest.
    Manifest must contain at least 'feature_path' and 'label' columns (case-insensitive).
    """
    def __init__(self, manifest_csv, mode="train", target_frames=None, normalize=True):
        self.mode = mode
        df = pd.read_csv(manifest_csv)
        # accept 'feature_path' or 'path' etc.
        if 'feature_path' in df.columns:
            self.df = df[['feature_path', 'label']].copy()
        elif 'path' in df.columns:
            self.df = df[['path', 'label']].rename(columns={'path': 'feature_path'})
        else:
            # fallback: take first two columns
            self.df = df.iloc[:, :2].copy()
            self.df.columns = ['feature_path', 'label']

        self.df = self.df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found in manifest {manifest_csv}")

        # auto-detect shape
        sample_path = self.df.loc[0, 'feature_path']

        if not os.path.exists(sample_path):
            # fallback to original feature directory
            sample_path = os.path.join(
                r"C:\SoundModel\data\gtzan\features",
                os.path.basename(sample_path)
            )

        feat = np.load(sample_path)


        t, f = feat.shape
        self.target_frames = int(target_frames) if target_frames is not None else t
        self.n_mels = f
        self.normalize = normalize

        # label mapping
        labels = sorted(self.df['label'].unique(), key=lambda x: str(x))
        self.label_map = {lbl: idx for idx, lbl in enumerate(labels)}
        self.inv_label_map = {v:k for k,v in self.label_map.items()}

        # compute global stats on a subset if training and normalization requested
        self.global_mean = 0.0
        self.global_std = 1.0
        if self.normalize and self.mode == "train":
            self._compute_global_stats(sample_frac=0.25)

    def _compute_global_stats(self, sample_frac=0.25):
        paths = self.df['feature_path'].tolist()
        n = max(1, int(len(paths) * sample_frac))
        sampled = random.sample(paths, n)
        s = 0.0; s2 = 0.0; count = 0
        for p in sampled:
            arr = np.load(p).astype(np.float32)
            arr = self._pad_or_truncate(arr)
            s += arr.sum(); s2 += (arr ** 2).sum(); count += arr.size
        self.global_mean = s / count
        self.global_std = math.sqrt(max(1e-12, s2 / count - self.global_mean**2))
        print(f"[dataset] Global mean={self.global_mean:.6f}, std={self.global_std:.6f} from {n} samples")

    def _pad_or_truncate(self, feat):
        t = feat.shape[0]
        if t < self.target_frames:
            pad = np.zeros((self.target_frames - t, feat.shape[1]), dtype=feat.dtype)
            feat = np.concatenate([feat, pad], axis=0)
        elif t > self.target_frames:
            if self.mode == 'train':
                start = random.randint(0, t - self.target_frames)
            else:
                start = (t - self.target_frames) // 2
            feat = feat[start:start + self.target_frames, :]
        return feat

    def _specaugment(self, feat, time_mask_num=2, time_mask_pct=0.12, freq_mask_num=2, freq_mask_pct=0.12):
        # time masks
        T, F = feat.shape
        max_t = max(1, int(time_mask_pct * T))
        for _ in range(time_mask_num):
            t = random.randint(0, max_t)
            if t == 0: continue
            start = random.randint(0, T - t)
            feat[start:start+t, :] = 0.0
        # freq masks
        max_f = max(1, int(freq_mask_pct * F))
        for _ in range(freq_mask_num):
            f = random.randint(0, max_f)
            if f == 0: continue
            start = random.randint(0, F - f)
            feat[:, start:start+f] = 0.0
        return feat

    def _time_shift(self, feat, max_pct=0.03):
        T = feat.shape[0]
        max_shift = max(1, int(max_pct * T))
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return feat
        return np.roll(feat, shift, axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['feature_path']
        label_raw = row['label']
        label = self.label_map[label_raw]

        if not os.path.exists(path):
            path = os.path.join(
                r"C:\SoundModel\data\gtzan\features",
                os.path.basename(path)
            )

        feat = np.load(path).astype(np.float32)  # shape (time, n_mels)

        feat = self._pad_or_truncate(feat)

        # augmentations on spectrogram-level (train only)
        if self.mode == "train":
            if random.random() < 0.5:
                feat = self._time_shift(feat)
            if random.random() < 0.7:
                feat = self._specaugment(feat)

            # additive gaussian noise
            if random.random() < 0.3:
                noise = np.random.normal(0, 1, feat.shape).astype(np.float32)
                feat = feat + noise * 0.003

        # normalize with global stats (or per-sample)
        if self.normalize:
            feat = (feat - self.global_mean) / (self.global_std + 1e-8)

        # return (time, n_mels), label
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ------------------ MixUp utility ------------------
def mixup_data(x, y, alpha=0.3, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b, lam)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------ Model: CNN -> BiLSTM -> Transformer -> Attention ------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), padding=1, pool=(2,2), dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(pool),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class UltimateNet(nn.Module):
    def __init__(self, time_frames, n_mels, num_classes, lstm_hidden=256, transformer_heads=4, transformer_layers=2, dropout=0.3):
        super().__init__()
        # conv stack
        self.conv1 = ConvBlock(1, 32, dropout=0.15)
        self.conv2 = ConvBlock(32, 64, dropout=0.2)
        self.conv3 = ConvBlock(64, 128, dropout=0.25)
        self.conv4 = ConvBlock(128, 256, dropout=0.3)

        # compute shape after convs
        with torch.no_grad():
            dummy = torch.zeros(1,1,time_frames,n_mels)
            x = self._forward_convs(dummy)
            _, c, t, f = x.shape
            transformer_input_dim = c * f
            transformer_time = t

        # BiLSTM
        self.lstm = nn.LSTM(input_size=transformer_input_dim, hidden_size=lstm_hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)

        # small Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden*2, nhead=transformer_heads, dim_feedforward=512, dropout=0.2, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # attention pooling
        self.attn_w = nn.Linear(lstm_hidden*2, 1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def _forward_convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward(self, x):
        # x: (B, T, F)
        x = x.unsqueeze(1)              # (B,1,T,F)
        x = self._forward_convs(x)      # (B, C, T', F')
        b, c, t, f = x.shape
        x = x.permute(0,2,1,3).contiguous().view(b, t, c * f)  # (B, T', C*F)
        # LSTM
        lstm_out, _ = self.lstm(x)      # (B, T', H*2)
        # Transformer expects (T, B, C)
        trans_in = lstm_out.permute(1,0,2)
        trans_out = self.transformer(trans_in).permute(1,0,2)  # (B,T',H*2)
        # attention pooling
        attn_logits = self.attn_w(trans_out)  # (B, T', 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = (attn_weights * trans_out).sum(dim=1)  # (B, H*2)
        logits = self.classifier(context)
        return logits

# ------------------ Helpers ------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def save_checkpoint(path, model, label_map, target_frames, n_mels, global_mean, global_std):
    data = {
        'model_state_dict': model.state_dict(),
        'label_map': label_map,
        'target_frames': target_frames,
        'n_mels': n_mels,
        'global_mean': global_mean,
        'global_std': global_std
    }
    torch.save(data, path)

# ------------------ Training / Eval loops ------------------
def train_one_epoch(model, loader, optimizer, scheduler, epoch, scaler=None, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for features, labels in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # MixUp
        if mixup_alpha > 0:
            mixed_x, (y_a, y_b, lam) = mixup_data(features, labels, alpha=mixup_alpha, device=DEVICE)
            inputs = mixed_x
        else:
            inputs = features

        if USE_AMP and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                if mixup_alpha > 0:
                    loss = mixup_criterion(lambda_logits, logits, y_a, y_b, lam) if False else None
                # label smoothing implemented via helper
                loss = label_smoothing_loss(logits, labels, smoothing=LABEL_SMOOTHING) if mixup_alpha == 0 else (lam * label_smoothing_loss(logits, y_a, smoothing=LABEL_SMOOTHING) + (1-lam) * label_smoothing_loss(logits, y_b, smoothing=LABEL_SMOOTHING))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            if mixup_alpha > 0:
                # mixup with label-smoothing via soft labels
                loss = lam * label_smoothing_loss(logits, y_a, smoothing=LABEL_SMOOTHING) + (1-lam) * label_smoothing_loss(logits, y_b, smoothing=LABEL_SMOOTHING)
            else:
                loss = label_smoothing_loss(logits, labels, smoothing=LABEL_SMOOTHING)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits.detach().cpu(), dim=1).numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    # scheduler step for OneCycleLR is handled per-batch in training (we do below in main)
    return epoch_loss, acc, prec, rec, f1

def evaluate(model, loader):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Eval", leave=False):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            loss = label_smoothing_loss(logits, labels, smoothing=LABEL_SMOOTHING)
            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits.detach().cpu(), dim=1).numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return epoch_loss, acc, prec, rec, f1, all_labels, all_preds

# Label smoothing helper
def label_smoothing_loss(logits, targets, smoothing=0.1):
    """
    logits: (B, C), targets: (B,) int
    """
    n_cls = logits.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (n_cls - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
    log_prob = torch.log_softmax(logits, dim=1)
    return torch.mean(torch.sum(- true_dist * log_prob, dim=1))

# ------------------ Main ------------------
def main():
        # =================== PATH SETUP ===================
    #BASE_DIR = "C:\\SoundModel\\data\\gtzan"
    #FEATURE_DIR = os.path.join(BASE_DIR, "features")
    #MANIFEST_DIR = os.path.join(BASE_DIR, "manifests")

    #TRAIN_MANIFEST = os.path.join(MANIFEST_DIR, "train_manifest.csv")
    #VAL_MANIFEST = os.path.join(MANIFEST_DIR, "val_manifest.csv")
    # ==================================================
    # datasets
    # Fix manifest file paths (since all features are in one flat folder)
    global TRAIN_MANIFEST, VAL_MANIFEST, FEATURE_DIR
    TRAIN_MANIFEST = fix_manifest_paths(TRAIN_MANIFEST, FEATURE_DIR)
    VAL_MANIFEST = fix_manifest_paths(VAL_MANIFEST, FEATURE_DIR)

    train_ds = GTZANFeatureDataset(TRAIN_MANIFEST, mode="train", target_frames=TARGET_FRAMES, normalize=True)
    val_ds = GTZANFeatureDataset(VAL_MANIFEST, mode="val", target_frames=TARGET_FRAMES, normalize=True)

    # reuse normalization stats
    val_ds.global_mean = train_ds.global_mean
    val_ds.global_std = train_ds.global_std

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    n_classes = len(train_ds.label_map)
    print(f"Device={DEVICE}, AMP={USE_AMP}, Train samples={len(train_ds)}, Val samples={len(val_ds)}, Classes={n_classes}")
    print(f"Feature shape (T x F) = ({train_ds.target_frames} x {train_ds.n_mels})")

    # Model
    model = UltimateNet(time_frames=train_ds.target_frames, n_mels=train_ds.n_mels, num_classes=n_classes).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    # optimizer + scheduler (OneCycleLR)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS, pct_start=0.1, anneal_strategy='cos', final_div_factor=100)

    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    # bookkeeping
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    metrics_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        # training
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        # manual batch loop so we can step OneCycleLR per batch
        for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc=f"Train E{epoch}", leave=False)):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            # mixup with some probability
            if MIXUP_ALPHA > 0 and random.random() < 0.5:
                inputs, (y_a, y_b, lam) = mixup_data(features, labels, alpha=MIXUP_ALPHA, device=DEVICE)
            else:
                inputs = features
                y_a, y_b, lam = None, None, None

            if USE_AMP and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    if lam is not None:
                        loss = lam * label_smoothing_loss(logits, y_a, smoothing=LABEL_SMOOTHING) + (1-lam) * label_smoothing_loss(logits, y_b, smoothing=LABEL_SMOOTHING)
                    else:
                        loss = label_smoothing_loss(logits, labels, smoothing=LABEL_SMOOTHING)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                if lam is not None:
                    loss = lam * label_smoothing_loss(logits, y_a, smoothing=LABEL_SMOOTHING) + (1-lam) * label_smoothing_loss(logits, y_b, smoothing=LABEL_SMOOTHING)
                else:
                    loss = label_smoothing_loss(logits, labels, smoothing=LABEL_SMOOTHING)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                optimizer.step()

            # step LR scheduler per batch
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits.detach().cpu(), dim=1).numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

        # validation
        val_loss, val_acc, val_prec, val_rec, val_f1, v_labels, v_preds = evaluate(model, val_loader)

        t1 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch}/{NUM_EPOCHS}]  Time: {(t1-t0):.1f}s  LR: {lr_now:.6g}")
        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, Prec:{train_prec:.3f}, Rec:{train_rec:.3f}, F1:{train_f1:.3f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, Prec:{val_prec:.3f}, Rec:{val_rec:.3f}, F1:{val_f1:.3f}")

        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_acc': train_acc, 'val_acc': val_acc,
            'train_prec': train_prec, 'val_prec': val_prec,
            'train_rec': train_rec, 'val_rec': val_rec,
            'train_f1': train_f1, 'val_f1': val_f1,
            'lr': lr_now
        })

        # save best
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_path = os.path.join(ARTIFACTS_DIR, "best_ultimate.pth")
            save_checkpoint(checkpoint_path, model, train_ds.label_map, train_ds.target_frames, train_ds.n_mels, train_ds.global_mean, train_ds.global_std)
            print(f"✅ Saved best model (val_acc={val_acc*100:.2f}%) to {checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping.")
                break

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_history)
    csv_path = os.path.join(ARTIFACTS_DIR, "metrics_history.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics CSV to {csv_path}")

    # final confusion matrix and classification report on val
    cm = confusion_matrix(v_labels, v_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    cr = classification_report(v_labels, v_preds, target_names=[train_ds.inv_label_map[i] for i in range(len(train_ds.inv_label_map))], zero_division=0)
    print("Classification report:\n", cr)
    with open(os.path.join(ARTIFACTS_DIR, "classification_report.txt"), "w") as fh:
        fh.write(cr)

    # plot metrics
    plt.figure(figsize=(8,5))
    plt.plot(metrics_df['epoch'], metrics_df['train_acc'], label='train_acc')
    plt.plot(metrics_df['epoch'], metrics_df['val_acc'], label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(ARTIFACTS_DIR, "accuracy.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='train_loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(ARTIFACTS_DIR, "loss.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(metrics_df['epoch'], metrics_df['train_f1'], label='train_f1')
    plt.plot(metrics_df['epoch'], metrics_df['val_f1'], label='val_f1')
    plt.title('F1 Score')
    plt.xlabel('Epoch'); plt.ylabel('F1')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(ARTIFACTS_DIR, "f1.png"))
    plt.close()

    print(f"Artifacts saved to {ARTIFACTS_DIR}")
    print(f"Training complete. Best val acc: {best_val_acc*100:.2f}% at epoch {best_epoch}")

if __name__ == "__main__":
    main()
