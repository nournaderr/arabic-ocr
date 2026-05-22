"""CLI entry point for the Arabic OCR pipeline.

Usage examples:
    python run.py --image scan.jpg --classifier svm
    python run.py --image scan.jpg --classifier cnn --debug
    python run.py --batch images/ --classifier rf
"""
import argparse
import sys
from pathlib import Path


def _write_arabic_txt(path: Path, text: str) -> None:
    """Write Arabic text as UTF-8-BOM so Windows apps auto-detect encoding.

    UTF-8-BOM is the safest choice when the file will be opened by end-users
    on Windows: Notepad, Word, and most Arabic word-processors recognise the
    BOM and display RTL text correctly without manual encoding selection.
    A trailing newline is always appended so the file is POSIX-compliant.
    """
    with path.open("w", encoding="utf-8-sig", newline="\n") as fh:
        fh.write(text)
        if text and not text.endswith("\n"):
            fh.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Arabic OCR — classical pipeline (no ready-made OCR engine)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",  metavar="PATH", help="Path to a single input image")
    mode.add_argument("--batch",  metavar="DIR",  help="Directory of images to process")

    parser.add_argument(
        "--classifier", default="svm",
        choices=["svm", "rf", "cnn"],
        help="Classifier backend (default: svm)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate debug visualizations to output/debug/",
    )
    parser.add_argument(
        "--output", metavar="FILE",
        help="Write recognised text to this .txt file (default: output/<image_stem>.txt)",
    )
    args = parser.parse_args()

    from arabic_ocr.pipeline import ArabicOCRPipeline
    from arabic_ocr.config import OUTPUT_DIR
    pipe = ArabicOCRPipeline(classifier=args.classifier, debug=args.debug)

    if args.image:
        text = pipe.run(args.image)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / (Path(args.image).stem + ".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_arabic_txt(out_path, text)
        print(f"Saved: {out_path}")

    else:
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            print(f"Error: '{batch_dir}' is not a directory.", file=sys.stderr)
            sys.exit(1)

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        paths = sorted(
            p for p in batch_dir.iterdir()
            if p.suffix.lower() in image_exts
        )
        if not paths:
            print("No images found.", file=sys.stderr)
            sys.exit(1)

        results = pipe.run_batch(paths)
        out_dir = Path(args.output) if args.output else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        for path, text in zip(paths, results):
            out_path = out_dir / (path.stem + ".txt")
            _write_arabic_txt(out_path, text)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
