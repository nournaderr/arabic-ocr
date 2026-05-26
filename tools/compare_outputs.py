"""Compare two OCR output files and compute Levenshtein distance.

Usage:
    python tools/compare_outputs.py file1 file2
"""
import sys
from pathlib import Path


def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[lb]


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_outputs.py file1 file2")
        return
    f1 = Path(sys.argv[1])
    f2 = Path(sys.argv[2])
    a = f1.read_text(encoding="utf-8")
    b = f2.read_text(encoding="utf-8")
    d = levenshtein(a, b)
    total = max(1, max(len(a), len(b)))
    print(f"Levenshtein distance: {d}")
    print(f"Relative change: {d}/{total} = {d/total:.3%}")


if __name__ == "__main__":
    main()
