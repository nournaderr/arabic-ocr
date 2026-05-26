"""Integration tests: run the full pipeline on synthetic images.

These tests verify that every stage connects without crashing and that the
output is a valid (possibly empty) Arabic string.  They do NOT require a
trained model — the classifier is mocked to return fixed candidates.
"""
import numpy as np
import pytest

from arabic_ocr.utils.arabic_utils import is_arabic_char, ARABIC_LETTERS


# ── Synthetic test image ──────────────────────────────────────────────────────

def _page_image() -> np.ndarray:
    """White BGR image with two black horizontal bands simulating text lines."""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    img[60:100,  50:550] = 0   # line 1
    img[180:220, 50:550] = 0   # line 2
    return img


# ── Mock classifier ───────────────────────────────────────────────────────────

class _MockClassifier:
    """Always returns ('ب', 0.9) as top prediction."""

    classes_ = np.array(ARABIC_LETTERS)

    def predict(self, char_img, dot_list=None):
        from arabic_ocr.config import TOP_K
        return [(ARABIC_LETTERS[i % len(ARABIC_LETTERS)], 0.9 / (i + 1))
                for i in range(TOP_K)]

    def predict_batch(self, char_imgs, dot_lists=None):
        return [self.predict(img) for img in char_imgs]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def setup_method(self):
        from arabic_ocr.pipeline import ArabicOCRPipeline
        from arabic_ocr.postprocess.language_model import ArabicLanguageModel

        self.pipe = ArabicOCRPipeline.__new__(ArabicOCRPipeline)
        self.pipe.classifier = _MockClassifier()
        self.pipe.lm         = ArabicLanguageModel.__new__(ArabicLanguageModel)
        from collections import defaultdict
        self.pipe.lm.bigrams         = defaultdict(lambda: defaultdict(float))
        self.pipe.lm._loaded_bigrams = False
        self.pipe.lm._dawg_root      = None
        self.pipe.debug = False

    def test_run_array_returns_string(self):
        img = _page_image()
        result = self.pipe.run_array(img)
        assert isinstance(result, str)

    def test_output_chars_are_arabic_or_space(self):
        img = _page_image()
        result = self.pipe.run_array(img)
        for ch in result:
            assert is_arabic_char(ch) or ch == " ", f"Unexpected char: {ch!r}"

    def test_empty_image_returns_empty_string(self):
        blank = np.full((200, 300, 3), 255, dtype=np.uint8)
        result = self.pipe.run_array(blank)
        assert result == ""

    def test_run_batch_length(self, tmp_path):
        import cv2
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            cv2.imwrite(str(p), _page_image())
            paths.append(p)
        results = self.pipe.run_batch(paths)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)

    def test_candidate_filter_fallback_keeps_original_candidates(self):
        from arabic_ocr.pipeline import _filter_candidates_with_fallback

        candidates = [("Baa_Start", 0.9), ("Alf_Start", 0.1)]
        result = _filter_candidates_with_fallback(
            candidates,
            position="initial",
            dots_above=0,
            dots_below=1,
        )

        assert result == candidates

    def test_pipeline_handles_classifier_exception(self):
        # Build a real pipeline object but inject a classifier that raises
        from arabic_ocr.pipeline import ArabicOCRPipeline
        pipe = ArabicOCRPipeline.__new__(ArabicOCRPipeline)
        class BrokenClassifier:
            def predict_batch(self, imgs, dots=None):
                raise RuntimeError("classifier crashed")
        pipe.classifier = BrokenClassifier()
        pipe.lm = None
        pipe.debug = False

        # Create a simple synthetic image with one line so segmentation yields crops
        import numpy as np
        img = np.full((60, 200, 3), 255, dtype=np.uint8)
        img[10:50, 20:180] = 0

        # Should not raise — postprocess will receive undetected candidates and return ""
        result = pipe.run_array(img)
        assert isinstance(result, str)


class TestArabicUtils:
    def test_is_arabic_char(self):
        for ch in ARABIC_LETTERS:
            assert is_arabic_char(ch)
        assert not is_arabic_char("a")
        assert not is_arabic_char("1")

    def test_normalize_arabic_strips_harakat(self):
        from arabic_ocr.utils.arabic_utils import normalize_arabic
        # Arabic word with fatha (U+064E)
        with_haraka = "كَتَبَ"
        normalized = normalize_arabic(with_haraka)
        assert "َ" not in normalized
        assert "ك" in normalized

    def test_join_text_rtl_contains_words(self):
        from arabic_ocr.utils.arabic_utils import join_text_rtl
        result = join_text_rtl(["كتاب", "قلم"])
        assert "كتاب" in result
        assert "قلم" in result
