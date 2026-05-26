"""Grid-search DOT_RERANK_BOOST and DOT_RERANK_PENALTY using paw_diagnostics as metric.
Usage: python tools/tune_rerank.py data/test_images/arabic2.jpg

Metric: number of flagged PAWs from tools/paw_diagnostics.py (lower is better).
"""
import subprocess
import sys
from itertools import product
from statistics import mean

from arabic_ocr import config as cfg

IMG = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/arabic2.jpg"

boosts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
penalties = [0.0, 0.025, 0.05, 0.1, 0.15]

results = []

for b, p in product(boosts, penalties):
    cfg.DOT_RERANK_BOOST = float(b)
    cfg.DOT_RERANK_PENALTY = float(p)
    # run diagnostics
    proc = subprocess.run([
        sys.executable, "tools/paw_diagnostics.py", IMG
    ], capture_output=True, text=True)
    out = proc.stdout
    # count flagged lines after 'Flagged PAWs:'
    flagged_section = False
    flagged_count = 0
    for line in out.splitlines():
        if line.strip().startswith('Flagged PAWs:'):
            flagged_section = True
            continue
        if flagged_section:
            if not line.strip():
                break
            flagged_count += 1
    results.append((b, p, flagged_count))
    print(f"b={b:.3f} p={p:.3f} -> flagged={flagged_count}")

# sort by flagged count then by boost magnitude
results.sort(key=lambda t: (t[2], abs(t[0]-0.2)))
print('\nTop results:')
for b, p, f in results[:10]:
    print(f"boost={b:.3f} penalty={p:.3f} flagged={f}")

# print best setting
best = results[0]
print(f"\nBest setting: boost={best[0]:.3f} penalty={best[1]:.3f} flagged={best[2]}")
print('Note: this tunes only the diagnostics metric; validate with full OCR.')
