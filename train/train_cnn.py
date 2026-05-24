"""Train CNNClassifier on an HMDB-style dataset.

Expected layout (Option A — one class per LetterName_Position folder):
    data/chars/
        Ain_Isolated/  img_001.png  …
        Ain_Start/     …
        Ain_Middle/    …
        Ain_End/       …
        Baa_Isolated/  …
        …

Each folder name becomes a single class label (~112 classes = 28 × 4).
Augmentation: ±10° rotation, Gaussian noise, random dilation/erosion.
Saves best validation-accuracy checkpoint to models/cnn/model.pt.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from arabic_ocr.config import MODELS_DIR, DATA_DIR
from arabic_ocr.classifiers import CNNClassifier
from arabic_ocr.features.normalize import normalize
from arabic_ocr.utils.arabic_utils import hmdb_label_to_unicode


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(img: np.ndarray) -> np.ndarray:
    angle = random.uniform(-10, 10)
    h, w  = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    if random.random() < 0.5:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        kernel = np.ones((2, 2), np.uint8)
        op = cv2.dilate if random.random() < 0.5 else cv2.erode
        img = op(img, kernel, iterations=1)

    return img


# ── Dataset ───────────────────────────────────────────────────────────────────

class HMDBCharDataset(Dataset):
    """Loads images from LetterName_Position subfolders.

    Labels are the folder names verbatim (Option A: ~112 classes).
    The LabelEncoder maps them to integer indices for CrossEntropyLoss.
    """

    def __init__(self, imgs: np.ndarray, labels: np.ndarray, train: bool = True):
        self.imgs   = imgs
        self.labels = labels
        self.train  = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].copy()
        if self.train:
            img = augment(img)
        t    = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
        dots = torch.zeros(4, dtype=torch.float32)  # dots added at inference
        return t, dots, torch.tensor(self.labels[idx], dtype=torch.long)


# ── Data loading ──────────────────────────────────────────────────────────────

# ── Label normalisation ───────────────────────────────────────────────────────
# Some datasets name the same letter differently (e.g. "Gem" vs "Gen" for ج).
# Merging these at load time prevents the model from trying to learn
# visually-identical classes as separate outputs.
_LABEL_MERGES: dict[str, str] = {
    # Gem_* and Gen_* are both ج (Jeem) — Egyptian "G" vs MSA "J" pronunciation.
    # Merge all Gem_* into Gen_* since Gen has more training samples.
    "Gem_Isolated": "Gen_Isolated",
    "Gem_Start":    "Gen_Start",
    "Gem_Middle":   "Gen_Middle",
    "Gem_End":      "Gen_End",
}


def _normalize_label(label: str) -> str:
    return _LABEL_MERGES.get(label, label)


def load_dataset(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Walk data_dir and return (normalised_image_array, label_list).

    Prints a loading summary with Unicode glyphs where known.
    Labels are normalised via _normalize_label() to merge visually-identical
    duplicate classes (e.g. Gem_* → Gen_* for ج).
    """
    raw_imgs, ys = [], []

    folders = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not folders:
        raise FileNotFoundError(f"No subdirectories in {data_dir}")

    for folder in folders:
        raw_label = folder.name
        label     = _normalize_label(raw_label)
        loaded = 0
        img_paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
        for img_path in img_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            raw_imgs.append(normalize(img))
            ys.append(label)
            loaded += 1

        if loaded:
            merged = f" → {label}" if label != raw_label else ""
            glyph  = hmdb_label_to_unicode(label)
            display = f" ({glyph})" if glyph else ""
            print(f"  {raw_label}{merged}{display}: {loaded}")

    if not raw_imgs:
        raise RuntimeError(f"No PNG images found under {data_dir}")

    print(f"\nTotal: {len(raw_imgs)} images, {len(set(ys))} classes")
    return np.stack(raw_imgs), ys


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN on HMDB-style LetterName_Position folders"
    )
    parser.add_argument("--data-dir",   default=str(DATA_DIR / "chars"))
    parser.add_argument("--out",        default=str(MODELS_DIR / "cnn" / "model.pt"))
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--test-split", type=float, default=0.10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    X, ys = load_dataset(data_dir)

    enc   = LabelEncoder()
    enc.fit(ys)
    y_idx = enc.transform(ys)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_idx, test_size=args.test_split, random_state=42, stratify=y_idx,
    )

    train_loader = DataLoader(
        HMDBCharDataset(X_tr, y_tr, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        HMDBCharDataset(X_val, y_val, train=False),
        batch_size=args.batch_size,
    )

    n_classes = len(enc.classes_)
    print(f"Classes: {n_classes}  Train: {len(X_tr)}  Val: {len(X_val)}")

    clf       = CNNClassifier(n_classes=n_classes, lr=args.lr, epochs=args.epochs)
    clf.classes_ = enc.classes_
    clf.model    = clf._build_model()

    opt       = torch.optim.Adam(clf.model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc   = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        clf.model.train()
        train_loss = 0.0
        for xb, db, yb in train_loader:
            opt.zero_grad()
            loss = criterion(clf.model(xb, db), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(yb)
        scheduler.step()
        train_loss /= len(X_tr)

        # ── Validate ─────────────────────────────────────────────────────────
        clf.model.eval()
        correct = total = 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, db, yb in val_loader:
                preds = clf.model(xb, db).argmax(dim=1)
                correct      += (preds == yb).sum().item()
                total        += len(yb)
                all_preds.extend(preds.numpy())
                all_true.extend(yb.numpy())

        acc = correct / total
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc   = acc
            best_epoch = epoch
            clf.save(Path(args.out))
            print(f"  → checkpoint saved (acc={best_acc:.4f})")

    print(f"\nBest val accuracy: {best_acc:.4f} at epoch {best_epoch}")

    # ── Per-class report on the validation set ────────────────────────────────
    print("\n── Per-class report (validation set) ────────────────────────────────")
    present_indices = sorted(set(all_true))
    present_labels  = enc.classes_[present_indices]
    target_names = [
        f"{lbl} {hmdb_label_to_unicode(lbl)}".strip()
        for lbl in present_labels
    ]
    print(classification_report(
        all_true, all_preds,
        labels=present_indices,
        target_names=target_names,
        zero_division=0,
    ))
    print(f"Model saved → {args.out}")


if __name__ == "__main__":
    main()
