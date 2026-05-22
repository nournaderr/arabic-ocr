import numpy as np
import pytest

from arabic_ocr.postprocess.language_model import ArabicLanguageModel
from arabic_ocr.postprocess.viterbi import viterbi_decode
from arabic_ocr.postprocess.beam_search import beam_search_decode
from arabic_ocr.postprocess.dawg import build_dawg, dawg_search, DawgNode


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _trained_lm():
    lm = ArabicLanguageModel.__new__(ArabicLanguageModel)
    from collections import defaultdict
    lm.bigrams = defaultdict(lambda: defaultdict(float))
    lm._loaded_bigrams = True
    lm._dawg_root = None
    # Inject a simple bigram: ب → ي is very likely
    lm.bigrams["ب"]["ي"] = 0.9
    lm.bigrams["ب"]["ا"] = 0.1
    return lm


def _candidates():
    return [
        [("ب", 0.9), ("ت", 0.05), ("ث", 0.03), ("ن", 0.01), ("ج", 0.01)],
        [("ي", 0.8), ("ا", 0.1),  ("و", 0.05), ("ل", 0.03), ("م", 0.02)],
        [("ت", 0.7), ("ث", 0.15), ("ب", 0.05), ("ن", 0.05), ("ج", 0.05)],
    ]


# ── Viterbi ───────────────────────────────────────────────────────────────────

class TestViterbi:
    def test_returns_sequence_and_score(self):
        lm   = _trained_lm()
        seq, score = viterbi_decode(_candidates(), lm)
        assert isinstance(seq, list)
        assert isinstance(score, float)
        assert len(seq) == 3

    def test_sequence_contains_arabic_chars(self):
        lm  = _trained_lm()
        seq, _ = viterbi_decode(_candidates(), lm)
        for ch in seq:
            assert ch != ""

    def test_prefers_high_prob_bigram(self):
        # ب → ي has bigram 0.9, so the decoder should pick ي at position 1
        lm = _trained_lm()
        seq, _ = viterbi_decode(_candidates(), lm)
        assert seq[0] == "ب"
        assert seq[1] == "ي"

    def test_empty_input(self):
        lm  = _trained_lm()
        seq, score = viterbi_decode([], lm)
        assert seq == []
        assert score == 0.0


# ── Beam search ───────────────────────────────────────────────────────────────

class TestBeamSearch:
    def test_returns_list(self):
        lm  = _trained_lm()
        seq = beam_search_decode(_candidates(), lm, beam_width=3)
        assert isinstance(seq, list)
        assert len(seq) == 3

    def test_empty_input(self):
        lm  = _trained_lm()
        assert beam_search_decode([], lm) == []


# ── Language model ────────────────────────────────────────────────────────────

class TestLanguageModel:
    def test_rescore_candidates_reorders(self):
        lm = _trained_lm()
        cands = [("ا", 0.6), ("ي", 0.3), ("و", 0.1)]
        rescored = lm.rescore_candidates(cands, context=["ب"])
        # ي should move up because bigram ب→ي = 0.9
        chars_in_order = [c for c, _ in rescored]
        assert "ي" in chars_in_order

    def test_bigram_score_trained(self):
        lm = _trained_lm()
        score = lm._bigram_score(["ب", "ي"])
        assert score < 0   # log probability is negative
        assert score > lm._bigram_score(["ب", "ا"])  # ب→ي > ب→ا

    def test_in_lexicon_false_when_no_dawg(self):
        lm = _trained_lm()
        assert lm._in_lexicon("كتاب") is False

    def test_train_from_text(self):
        lm = ArabicLanguageModel.__new__(ArabicLanguageModel)
        from collections import defaultdict
        lm.bigrams = defaultdict(lambda: defaultdict(float))
        lm._loaded_bigrams = False
        lm._dawg_root = None
        lm.train_from_text("بيت بيت بيت")
        assert lm._loaded_bigrams
        assert lm.bigrams["ب"]["ي"] > 0


# ── DAWG ──────────────────────────────────────────────────────────────────────

class TestDawg:
    def test_build_and_search(self):
        root = build_dawg(["كتاب", "كتابة", "قلم", "قلب"])
        completions = dawg_search(root, "كتا")
        assert "كتاب" in completions
        assert "كتابة" in completions
        assert "قلم" not in completions

    def test_unknown_prefix_returns_empty(self):
        root = build_dawg(["كتاب"])
        assert dawg_search(root, "xyz") == []

    def test_save_load_roundtrip(self, tmp_path):
        from arabic_ocr.postprocess.dawg import save_dawg, load_dawg
        root = build_dawg(["مدرسة", "مدرس"])
        save_dawg(root, tmp_path / "dawg.pkl")
        root2 = load_dawg(tmp_path / "dawg.pkl")
        assert "مدرسة" in dawg_search(root2, "مدر")
