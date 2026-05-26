"""Evaluate reranker effect by running OCR with model disabled/enabled.

Usage:
    python tools/eval_reranker.py [image_path]

Outputs saved to `output/arabic2_baseline.txt` and `output/arabic2_reranked.txt`.
Prints non-Arabic char counts and a short diff summary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_ocr.pipeline import ArabicOCRPipeline
from arabic_ocr.postprocess.reranker import RERANKER
from arabic_ocr.utils.arabic_utils import is_arabic_char
from arabic_ocr.config import OUTPUT_DIR


def count_non_arabic(text: str) -> int:
    return sum(1 for ch in text if not is_arabic_char(ch) and not ch.isspace())


def run_once(pipeline: ArabicOCRPipeline, image_path: str) -> str:
    return pipeline.run(image_path)


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/arabic2.jpg"
    pipe = ArabicOCRPipeline(classifier="cnn", debug=False)

    # Baseline: disable learned model (use heuristic)
    saved = RERANKER.model
    RERANKER.model = None
    baseline = run_once(pipe, img)
    out_base = OUTPUT_DIR / "arabic2_baseline.txt"
    out_base.write_text(baseline, encoding="utf-8")

    # Reranked: restore model
    RERANKER.model = saved
    reranked = run_once(pipe, img)
    out_rer = OUTPUT_DIR / "arabic2_reranked.txt"
    out_rer.write_text(reranked, encoding="utf-8")

    nb_base = count_non_arabic(baseline)
    nb_rer = count_non_arabic(reranked)
    total = max(1, len([c for c in reranked if not c.isspace()]))

    print("Baseline non-Arabic chars:", nb_base)
    print("Reranked non-Arabic chars:", nb_rer)
    print(f"Change: {nb_rer - nb_base} (lower is better)")
    print("Saved:", out_base, out_rer)

    # show short preview
    print("\nBaseline preview:\n", "\n".join(baseline.splitlines()[:6]))
    print("\nReranked preview:\n", "\n".join(reranked.splitlines()[:6]))


if __name__ == "__main__":
    main()
