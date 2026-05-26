from pathlib import Path
import sys
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_ocr.pipeline import ArabicOCRPipeline
from arabic_ocr.postprocess.reranker import RERANKER
from arabic_ocr.utils.arabic_utils import is_arabic_char


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


def count_non_arabic(text: str) -> int:
    return sum(1 for ch in text if not is_arabic_char(ch) and not ch.isspace())


def main():
    data_dir = Path("data/test_images")
    images = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))
    if not images:
        print("No test images found in data/test_images/")
        return

    pipe = ArabicOCRPipeline(classifier="cnn", debug=False)
    saved = RERANKER.model

    rows = []
    for img in images:
        print(f"Processing {img.name}...")
        # baseline (heuristic)
        RERANKER.model = None
        base = pipe.run(img)
        (Path('output') / f"{img.stem}_baseline.txt").write_text(base, encoding='utf-8')
        # reranked
        RERANKER.model = saved
        rer = pipe.run(img)
        (Path('output') / f"{img.stem}_reranked.txt").write_text(rer, encoding='utf-8')

        nb_base = count_non_arabic(base)
        nb_rer = count_non_arabic(rer)
        ld = levenshtein(base, rer)
        rows.append((img.name, nb_base, nb_rer, ld, len(rer)))

    # restore
    RERANKER.model = saved

    print("\nSummary:")
    for r in rows:
        print(f"{r[0]}: base_non_ar={r[1]} rer_non_ar={r[2]} lev={r[3]} len={r[4]}")
    print("\nAggregate:")
    print("images:", len(rows))
    print("avg lev:", mean(r[3] for r in rows))
    print("avg base_non_ar:", mean(r[1] for r in rows))
    print("avg rer_non_ar:", mean(r[2] for r in rows))


if __name__ == '__main__':
    main()
