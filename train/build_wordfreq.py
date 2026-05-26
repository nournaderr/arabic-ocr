"""Build word-frequency list from corpus or Wikipedia dump.

Saves top-N frequent words to models/langmodel/word_freq.json

Usage:
    python train/build_wordfreq.py --corpus data/corpus/arabic_corpus.txt
"""
import argparse
import bz2
import json
import re
from collections import Counter
from pathlib import Path

from arabic_ocr.config import DATA_DIR, MODELS_DIR, WORD_FREQ_TOPN, WORD_FREQ_FILE

# Only Arabic script characters; excludes punctuation, digits, Latin.
_ARABIC_RE = re.compile(r"[؀-ۿ]+")


def extract_arabic_tokens_from_plaintext(corpus_path: str | Path):
    with open(str(corpus_path), encoding="utf-8", errors="replace") as f:
        for line in f:
            for token in _ARABIC_RE.findall(line):
                yield token


def extract_arabic_tokens_from_dump(dump_path: str | Path):
    skip_re = re.compile(r"^\s*[|{}\[\]<#*]")
    with bz2.open(str(dump_path), "rt", encoding="utf-8", errors="replace") as f:
        inside_text = False
        for line in f:
            if "<text" in line:
                inside_text = True
            if inside_text and not skip_re.match(line):
                for token in _ARABIC_RE.findall(line):
                    yield token
            if "</text>" in line:
                inside_text = False


def main():
    parser = argparse.ArgumentParser(description="Build Arabic word frequency list")
    parser.add_argument("--dump", default=str(DATA_DIR / "corpus" / "arwiki-latest-pages-articles.xml.bz2"))
    parser.add_argument("--corpus", default=str(DATA_DIR / "corpus" / "arabic_corpus.txt"))
    parser.add_argument("--out", default=str(WORD_FREQ_FILE))
    parser.add_argument("--topn", type=int, default=WORD_FREQ_TOPN)
    args = parser.parse_args()

    dump_path = Path(args.dump)
    corpus_path = Path(args.corpus)

    if dump_path.exists():
        token_iter = extract_arabic_tokens_from_dump(dump_path)
    elif corpus_path.exists():
        token_iter = extract_arabic_tokens_from_plaintext(corpus_path)
    else:
        print("No corpus source found. Provide --dump or --corpus")
        return

    counter = Counter()
    total = 0
    for tok in token_iter:
        counter[tok] += 1
        total += 1
        if total % 1000000 == 0:
            print(f"Processed {total:,} tokens; {len(counter):,} unique words")

    topn = args.topn
    most = counter.most_common(topn)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({w: c for w, c in most}, f, ensure_ascii=False)

    print(f"Saved top {len(most):,} words → {out_path}")


if __name__ == "__main__":
    main()
