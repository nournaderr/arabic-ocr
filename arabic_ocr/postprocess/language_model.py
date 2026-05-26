import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from arabic_ocr.config import BIGRAM_WEIGHT, DAWG_BOOST, MODELS_DIR
from arabic_ocr.config import WORD_FREQ_FILE, WORD_FREQ_BOOST


class ArabicLanguageModel:
    """Arabic character bigram language model with lexicon (DAWG) boost.

    Scoring reference: Jelinek 1998, Statistical Methods for Speech Recognition.
    Bigram source: Arabic Wikipedia dump or Quran text corpus.

    combined_score = (1 - BIGRAM_WEIGHT) * clf_conf + BIGRAM_WEIGHT * bigram_norm
    """

    def __init__(
        self,
        bigrams_path: Optional[Path] = None,
        dawg_path: Optional[Path] = None,
    ):
        self.bigrams: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._dawg_root = None
        self._loaded_bigrams = False

        bp = bigrams_path or MODELS_DIR / "langmodel" / "bigrams.json"
        dp = dawg_path    or MODELS_DIR / "langmodel" / "dawg.pkl"

        if Path(bp).exists():
            self.load_bigrams(Path(bp))
        if Path(dp).exists():
            self.load_dawg(Path(dp))
        # Optional word frequency file (word -> count)
        wf = WORD_FREQ_FILE
        if Path(wf).exists():
            self.load_word_freq(Path(wf))

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _bigram_score(self, sequence: list[str]) -> float:
        """Sum of log bigram probabilities for consecutive pairs."""
        if not self._loaded_bigrams or len(sequence) < 2:
            return 0.0
        score = 0.0
        for a, b in zip(sequence[:-1], sequence[1:]):
            p = self.bigrams.get(a, {}).get(b, 1e-6)
            score += float(np.log(p))
        return score

    def _in_lexicon(self, word: str) -> bool:
        """Trie lookup — True if word appears in the Arabic word list."""
        if self._dawg_root is None:
            return False
        from arabic_ocr.postprocess.dawg import dawg_search
        return word in dawg_search(self._dawg_root, word)

    def rescore_candidates(
        self,
        candidates: list[tuple[str, float]],
        context: list[str],
    ) -> list[tuple[str, float]]:
        """Re-rank top-K candidate characters given the preceding context.

        combined_score = (1 - BIGRAM_WEIGHT) * clf_conf
                       + BIGRAM_WEIGHT * bigram_norm
        """
        if not self._loaded_bigrams or not context:
            return candidates

        prev = context[-1]
        scored: list[tuple[str, float]] = []
        for char, clf_conf in candidates:
            bp = self.bigrams.get(prev, {}).get(char, 1e-6)
            # Normalise bigram to [0,1] range via sigmoid-like mapping
            bigram_norm = float(bp)
            combined = (1.0 - BIGRAM_WEIGHT) * clf_conf + BIGRAM_WEIGHT * bigram_norm
            scored.append((char, combined))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def rescore_word(self, word: str, clf_conf: float) -> float:
        """Return boosted confidence if word is in the Arabic lexicon."""
        boost = 0.0
        if self._in_lexicon(word):
            boost += DAWG_BOOST
        # Add word-frequency based boost (normalised using log scale)
        if getattr(self, "_word_freq", None):
            maxf = float(self._max_word_freq or 1)
            f = float(self._word_freq.get(word, 0))
            # Log-frequency is vastly better for words due to Zipf's law
            freq_norm = float(np.log(f + 1) / np.log(maxf + 1)) if maxf > 0 else 0.0
            boost += WORD_FREQ_BOOST * freq_norm
        return clf_conf + boost

    def load_word_freq(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._word_freq = {}
            self._max_word_freq = 0
            return
        # data expected as {word: count}
        self._word_freq = {w: int(c) for w, c in data.items()}
        self._max_word_freq = max(self._word_freq.values()) if self._word_freq else 0

    # ── Training ──────────────────────────────────────────────────────────────

    def train_from_text(self, text: str) -> None:
        """Estimate bigram probabilities from a raw Arabic text corpus."""
        chars = [c for c in text if "؀" <= c <= "ۿ"]
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for a, b in zip(chars[:-1], chars[1:]):
            counts[a][b] += 1
        for a, nexts in counts.items():
            total = sum(nexts.values())
            for b, cnt in nexts.items():
                self.bigrams[a][b] = cnt / total
        self._loaded_bigrams = True

    # ── Persistence ───────────────────────────────────────────────────────────

    def load_bigrams(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.bigrams = defaultdict(lambda: defaultdict(float))
        for a, nexts in data.items():
            for b, p in nexts.items():
                self.bigrams[a][b] = float(p)
        self._loaded_bigrams = True

    def save_bigrams(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({a: dict(nexts) for a, nexts in self.bigrams.items()},
                      f, ensure_ascii=False)

    def load_dawg(self, path: Path) -> None:
        from arabic_ocr.postprocess.dawg import load_dawg
        self._dawg_root = load_dawg(path)

    def save_dawg(self, path: Path) -> None:
        if self._dawg_root is not None:
            from arabic_ocr.postprocess.dawg import save_dawg
            save_dawg(self._dawg_root, path)
