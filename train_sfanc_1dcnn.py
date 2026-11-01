
"""
Train a 6-class 1D CNN for SFANC filter selection.

Inputs:
    noise_dataset/My_Dataset_16k/train_labels.csv
    noise_dataset/My_Dataset_16k/val_labels.csv
    noise_dataset/My_Dataset_16k/test_labels.csv
    noise_dataset/My_Dataset_16k/*.wav

Features:
- 16 kHz, 1-second mono waveform input
- 6 classes: W1..W6
- CUDA GPU automatically used when available
- Class-weighted CrossEntropyLoss
- Best model selected by validation macro-F1
- Early stopping
- Test accuracy / macro-F1 / per-class recall
- Confusion matrix saved as PNG
- Training history saved as CSV
- Best model saved as .pth

Outputs:
    sfanc_cnn_best.pth
    sfanc_training_history.csv
    sfanc_confusion_matrix.png
    sfanc_test_predictions.csv
"""

from pathlib import Path
import copy
import csv
import math
import random

import numpy as np
import pandas as pd
from scipy.io import wavfile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Paths / settings
# -------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "noise_dataset" / "My_Dataset_16k"

TRAIN_CSV = DATA_DIR / "train_labels.csv"
VAL_CSV = DATA_DIR / "val_labels.csv"
TEST_CSV = DATA_DIR / "test_labels.csv"

MODEL_OUT = BASE / "sfanc_cnn_best.pth"
HISTORY_OUT = BASE / "sfanc_training_history.csv"
CM_OUT = BASE / "sfanc_confusion_matrix.png"
PRED_OUT = BASE / "sfanc_test_predictions.csv"

FS = 16000
INPUT_LENGTH = 16000

CLASS_NAMES = ["W1", "W2", "W3", "W4", "W5", "W6"]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
NUM_WORKERS = 0  # Windows safest default
SEED = 42


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class SFANCDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.df)

    def _read_wav(self, path):
        fs, x = wavfile.read(path)

        if fs != FS:
            raise ValueError(f"{path.name}: expected {FS} Hz, got {fs} Hz")

        if x.ndim == 2:
            x = x.mean(axis=1)

        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            scale = float(max(abs(info.min), info.max))
            x = x.astype(np.float32) / scale
        else:
            x = x.astype(np.float32)

        # Force fixed 1-second input length.
        if len(x) < INPUT_LENGTH:
            x = np.pad(x, (0, INPUT_LENGTH - len(x)))
        elif len(x) > INPUT_LENGTH:
            x = x[:INPUT_LENGTH]

        # Remove DC and normalize RMS scale mildly.
        x = x - np.mean(x)

        rms = np.sqrt(np.mean(x * x) + 1e-12)
        if rms > 0:
            x = x / rms

        return x.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.data_dir / row["filename"]

        x = self._read_wav(path)
        y = LABEL_TO_IDX[row["label"]]

        # shape: [channel=1, time]
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        return x, y, row["filename"]


# -------------------------------------------------------------------------
# 1D CNN
# -------------------------------------------------------------------------
class SFANC1DCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -------------------------------------------------------------------------
# Class weights
# sqrt(N / N_i): softer than full inverse-frequency weighting
# -------------------------------------------------------------------------
def compute_class_weights(train_csv):
    df = pd.read_csv(train_csv)

    counts = np.array(
        [(df["label"] == name).sum() for name in CLASS_NAMES],
        dtype=np.float64
    )

    total = counts.sum()
    weights = np.sqrt(total / counts)

    # Normalize around mean = 1 for easier interpretation.
    weights = weights / weights.mean()

    print("\nClass counts / weights")
    print("======================")
    for name, count, weight in zip(CLASS_NAMES, counts, weights):
        print(f"{name}: count={int(count):6d}, weight={weight:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


# -------------------------------------------------------------------------
# Train / evaluate
# -------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_y = []
    all_pred = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)

            pred = torch.argmax(logits, dim=1)

            all_y.extend(y.detach().cpu().numpy())
            all_pred.extend(pred.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    macro_f1 = f1_score(all_y, all_pred, average="macro")

    return avg_loss, acc, macro_f1


def evaluate_test(model, loader, device):
    model.eval()

    all_y = []
    all_pred = []
    all_files = []

    with torch.no_grad():
        for x, y, filenames in loader:
            x = x.to(device, non_blocking=True)

            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            all_y.extend(y.numpy())
            all_pred.extend(pred)
            all_files.extend(list(filenames))

    return np.array(all_y), np.array(all_pred), all_files


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(6))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SFANC 1D CNN Confusion Matrix")

    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_OUT, dpi=150)
    plt.close(fig)


def main():
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = SFANCDataset(TRAIN_CSV, DATA_DIR)
    val_ds = SFANCDataset(VAL_CSV, DATA_DIR)
    test_ds = SFANCDataset(TEST_CSV, DATA_DIR)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    weights = compute_class_weights(TRAIN_CSV).to(device)

    model = SFANC1DCNN(num_classes=6).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_f1 = -1.0
    best_state = None
    epochs_without_improve = 0
    history = []

    print()
    print("Training starts")
    print("===============")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        scheduler.step(val_f1)

        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": lr,
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} F1 {val_f1:.4f} | "
            f"lr {lr:.2e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0

            torch.save({
                "model_state_dict": best_state,
                "class_names": CLASS_NAMES,
                "fs": FS,
                "input_length": INPUT_LENGTH,
                "best_val_macro_f1": best_f1,
            }, MODEL_OUT)

        else:
            epochs_without_improve += 1

            if epochs_without_improve >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping: no validation macro-F1 improvement "
                    f"for {EARLY_STOPPING_PATIENCE} epochs."
                )
                break

    pd.DataFrame(history).to_csv(HISTORY_OUT, index=False, encoding="utf-8-sig")

    # Restore best model.
    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)

    y_true, y_pred, files = evaluate_test(model, test_loader, device)

    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro")

    print()
    print("================ Test result ================")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Macro F1 : {test_macro_f1:.4f}")
    print()
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    save_confusion_matrix(y_true, y_pred)

    pred_df = pd.DataFrame({
        "filename": files,
        "true_label": [CLASS_NAMES[i] for i in y_true],
        "pred_label": [CLASS_NAMES[i] for i in y_pred],
    })
    pred_df.to_csv(PRED_OUT, index=False, encoding="utf-8-sig")

    print(f"Saved model      : {MODEL_OUT}")
    print(f"Saved history    : {HISTORY_OUT}")
    print(f"Saved confusion  : {CM_OUT}")
    print(f"Saved predictions: {PRED_OUT}")
    print("=============================================")


if __name__ == "__main__":
    main()
