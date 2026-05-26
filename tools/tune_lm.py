"""Grid-search BIGRAM_WEIGHT and DAWG_BOOST by measuring non-Arabic character rate in OCR output.
Usage: python tools/tune_lm.py data/test_images/arabic2.jpg --classifier cnn
Metric: count of non-Arabic characters (lower is better).
Note: quick heuristic; full evaluation requires labeled ground truth.
"""
import subprocess
import sys
from itertools import product
from pathlib import Path

from arabic_ocr import config as cfg

IMG = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/arabic2.jpg"
CLS = sys.argv[2] if len(sys.argv) > 2 else "cnn"
OUT_FILE = Path("output") / (Path(IMG).stem + ".txt")

bigram_weights = [0.0, 0.1, 0.2, 0.3, 0.5]
dawg_boosts = [0.0, 0.25, 0.5]

results = []

for bw, db in product(bigram_weights, dawg_boosts):
    cfg.BIGRAM_WEIGHT = float(bw)
    cfg.DAWG_BOOST = float(db)
    # run OCR
    proc = subprocess.run([
        sys.executable, "run.py", "--image", IMG, "--classifier", CLS
    ], capture_output=True, text=True)
    # read output file
    text = ""
    if OUT_FILE.exists():
        text = OUT_FILE.read_text(encoding="utf-8")
    # count non-Arabic characters (excluding whitespace/newline)
    non_ar = sum(1 for ch in text if ch.strip() and not (0x0600 <= ord(ch) <= 0x06FF))
    total_chars = sum(1 for ch in text if ch.strip()) or 1
    score = non_ar
    results.append((bw, db, score, non_ar, total_chars))
    print(f"bw={bw} db={db} -> non_ar={non_ar} / {total_chars}")

results.sort(key=lambda t: (t[2], abs(t[0]-0.3)))
print('\nTop results:')
for bw, db, score, non_ar, total in results[:6]:
    print(f"BIGRAM_WEIGHT={bw} DAWG_BOOST={db} non_ar={non_ar}/{total}")

best = results[0]
print(f"\nBest setting: BIGRAM_WEIGHT={best[0]} DAWG_BOOST={best[1]} non_ar={best[3]}/{best[4]}")
print('Persisting to config.py')
cfg.BIGRAM_WEIGHT = float(best[0])
cfg.DAWG_BOOST = float(best[1])
