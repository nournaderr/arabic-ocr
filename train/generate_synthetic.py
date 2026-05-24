"""Generate synthetic printed Arabic character images using Arabic fonts.

Renders each HMDB class label using system Arabic fonts, applies scan-style
augmentation, and saves to the existing data/chars/<LabelName>/ folders.

Also creates folders for commonly missing positional forms (e.g. Haa_Middle,
Mem_Middle) which are absent from the handwritten dataset.

Usage:
    python train/generate_synthetic.py
    python train/generate_synthetic.py --n 60 --fonts-dir C:/Windows/Fonts
    python train/generate_synthetic.py --missing-only
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

from arabic_ocr.config import DATA_DIR
from arabic_ocr.features.normalize import normalize
from arabic_ocr.utils.arabic_utils import HMDB_TO_UNICODE, hmdb_label_to_unicode

# ── Arabic fonts to try (in priority order) ───────────────────────────────────
# DT Naskh is newspaper-style; Traditional Arabic is the next best match.
_FONT_CANDIDATES = [
    "DTNASKH0.TTF",   # DT Naskh Regular
    "DTNASKH1.TTF",   # DT Naskh weights
    "DTNASKH2.TTF",
    "DTNASKH3.TTF",
    "DTNASKH4.TTF",
    "trado.ttf",      # Traditional Arabic
    "tradbdo.ttf",    # Traditional Arabic Bold
    "arabtype.ttf",   # Arabic Typesetting
    "Candarab.ttf",   # Candara Arabic
    "ARABSQ.TTF",
]

# Render sizes — variety forces size-invariant features
_SIZES = [40, 52, 64, 80]

# Positional forms that are frequently missing from handwritten datasets
# but are common in printed text.  The pipeline creates these folders so
# that retraining covers them.
_MISSING_FORMS = {
    "Haa_Middle":  "ـحـ",   # ح medial — very common
    "Khaa_Middle": "ـخـ",   # خ medial
    "Mem_Middle":  "ـمـ",   # م medial — very common
    "Tah_Start":   "طـ",    # ط initial
    "Tah_End":     "ـط",    # ط final
    "Zah_Start":   "ظـ",    # ظ initial
    "Zah_End":     "ـظ",    # ظ final
    "Gen_Middle":  "ـجـ",   # ج medial (was Gem_Middle — merged at train time)
    "Gen_Start":   "جـ",    # ج initial (Gen variant)
}


# ── Rendering ─────────────────────────────────────────────────────────────────

def _tight_crop(arr: np.ndarray) -> np.ndarray | None:
    """Crop to the bounding box of dark pixels.  Returns None if blank."""
    dark = arr < 200
    if not dark.any():
        return None
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    pad = 2
    r0 = max(0, rows[0]  - pad)
    r1 = min(arr.shape[0] - 1, rows[-1] + pad)
    c0 = max(0, cols[0]  - pad)
    c1 = min(arr.shape[1] - 1, cols[-1] + pad)
    return arr[r0:r1 + 1, c0:c1 + 1]


def render(unicode_text: str, font: ImageFont.FreeTypeFont, size: int) -> np.ndarray | None:
    """Render Arabic text and return a tight-cropped grayscale array.

    arabic_reshaper assigns the correct glyph form based on surrounding
    tatweels, so passing "ـحـ" gives the medial ح shape.
    """
    reshaped  = arabic_reshaper.reshape(unicode_text)
    bidi_text = get_display(reshaped)

    canvas = size * 4
    img  = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    x = (canvas - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (canvas - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), bidi_text, fill=0, font=font)

    return _tight_crop(np.array(img, dtype=np.uint8))


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(img: np.ndarray) -> np.ndarray:
    """Simulate newspaper scan artifacts."""
    # Gaussian blur (scanner defocus)
    if random.random() < 0.5:
        k = random.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Gaussian noise (scanner grain)
    if random.random() < 0.6:
        sigma = random.uniform(3, 14)
        noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Brightness shift (uneven illumination)
    if random.random() < 0.4:
        shift = random.randint(-25, 25)
        img   = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

    # JPEG compression artefacts (scanned newspapers saved as JPEG)
    if random.random() < 0.35:
        q = random.randint(50, 80)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img    = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    # Slight rotation (±5°)
    if random.random() < 0.4:
        angle = random.uniform(-5, 5)
        h, w  = img.shape
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h), borderValue=255)

    return img


# ── Per-class generation ──────────────────────────────────────────────────────

def generate_class(
    label: str,
    unicode_text: str,
    fonts: list[tuple[Path, str]],
    out_dir: Path,
    n_per_font: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("syn_*.png")))
    count    = 0

    for font_path, font_tag in fonts:
        for size in _SIZES:
            try:
                font = ImageFont.truetype(str(font_path), size)
            except Exception:
                continue

            base = render(unicode_text, font, size)
            if base is None:
                continue

            per_size = max(1, n_per_font // len(_SIZES))
            for _ in range(per_size):
                aug  = augment(base.copy())
                norm = normalize(aug)
                path = out_dir / f"syn_{font_tag}_{existing + count:05d}.png"
                cv2.imwrite(str(path), norm)
                count += 1

    return count


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic printed Arabic character images"
    )
    parser.add_argument(
        "--fonts-dir", default="C:/Windows/Fonts",
        help="Directory containing Arabic .ttf/.TTF font files",
    )
    parser.add_argument(
        "--out", default=str(DATA_DIR / "chars"),
        help="data/chars root (must already contain HMDB-style subfolders)",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Synthetic images per font per class (default 50)",
    )
    parser.add_argument(
        "--missing-only", action="store_true",
        help="Only generate for the missing positional-form classes",
    )
    args = parser.parse_args()

    fonts_dir = Path(args.fonts_dir)
    out_root  = Path(args.out)

    # ── Discover fonts ────────────────────────────────────────────────────────
    fonts: list[tuple[Path, str]] = []
    for fname in _FONT_CANDIDATES:
        fp = fonts_dir / fname
        if fp.exists():
            fonts.append((fp, fp.stem))

    if not fonts:
        print(f"No Arabic fonts found in {fonts_dir}.")
        print("Install Amiri/Noto Naskh Arabic or adjust --fonts-dir.")
        return

    print(f"Fonts ({len(fonts)}): {[t for _, t in fonts]}\n")

    # ── Build label → unicode mapping ────────────────────────────────────────
    # Start with all existing data/chars/ subfolders
    labels: dict[str, str] = {}

    if not args.missing_only:
        for folder in sorted(out_root.iterdir()):
            if not folder.is_dir():
                continue
            uval = hmdb_label_to_unicode(folder.name)
            if uval:
                labels[folder.name] = uval
            else:
                print(f"  skip (no unicode mapping): {folder.name}")

    # Add missing positional forms
    for label, uval in _MISSING_FORMS.items():
        if label not in labels:
            labels[label] = uval
            print(f"  + adding missing class: {label}")

    print(f"\nGenerating for {len(labels)} classes, {args.n} images/font/class ...\n")

    total = 0
    for label, uval in sorted(labels.items()):
        n = generate_class(label, uval, fonts, out_root / label, n_per_font=args.n)
        total += n
        print(f"  {label:30s}  +{n:4d}")

    print(f"\nTotal generated: {total:,} images")
    print("Next step: retrain the CNN  ->  python train/train_cnn.py")


if __name__ == "__main__":
    main()
