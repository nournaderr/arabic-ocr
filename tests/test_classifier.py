import numpy as np
import pytest

from arabic_ocr.config import TOP_K
from arabic_ocr.segment.dots import Dot
from arabic_ocr.utils.arabic_utils import ARABIC_LETTERS, is_arabic_char


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_dataset(n: int = 60):
    """Return (X, y) with n images per class for the first 4 Arabic letters."""
    classes = ARABIC_LETTERS[:4]
    imgs, labels = [], []
    for letter in classes:
        for _ in range(n):
            img = np.random.randint(0, 256, (20, 15), dtype=np.uint8)
            imgs.append(img)
            labels.append(letter)
    return imgs, np.array(labels)


def _train_svm(imgs, labels):
    from arabic_ocr.classifiers import SVMClassifier
    from arabic_ocr.features import extract_batch
    clf = SVMClassifier()
    X = extract_batch(imgs)
    clf.train(X, labels)
    return clf


def _train_rf(imgs, labels):
    from arabic_ocr.classifiers import RFClassifier
    from arabic_ocr.features import extract_batch
    clf = RFClassifier(n_estimators=10)
    X = extract_batch(imgs)
    clf.train(X, labels)
    return clf


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSVMClassifier:
    def setup_method(self):
        imgs, labels = _synthetic_dataset()
        self.clf = _train_svm(imgs, labels)
        self.test_img = imgs[0]

    def test_predict_returns_top_k(self):
        result = self.clf.predict(self.test_img)
        assert len(result) == TOP_K

    def test_predict_chars_are_arabic(self):
        result = self.clf.predict(self.test_img)
        for char, conf in result:
            assert is_arabic_char(char), f"Not Arabic: {char!r}"

    def test_predict_confidences_sum_leq_one(self):
        result = self.clf.predict(self.test_img)
        total = sum(conf for _, conf in result)
        assert total <= 1.01  # allow floating-point rounding

    def test_predict_confidences_descending(self):
        result = self.clf.predict(self.test_img)
        confs = [c for _, c in result]
        assert confs == sorted(confs, reverse=True)

    def test_predict_batch_length(self):
        imgs = [self.test_img] * 3
        results = self.clf.predict_batch(imgs)
        assert len(results) == 3
        for r in results:
            assert len(r) == TOP_K

    def test_save_load_roundtrip(self, tmp_path):
        from arabic_ocr.classifiers import SVMClassifier
        path = tmp_path / "svm.pkl"
        self.clf.save(path)
        clf2 = SVMClassifier()
        clf2.load(path)
        r1 = self.clf.predict(self.test_img)
        r2 = clf2.predict(self.test_img)
        assert r1[0][0] == r2[0][0]


class TestRFClassifier:
    def setup_method(self):
        imgs, labels = _synthetic_dataset()
        self.clf = _train_rf(imgs, labels)
        self.test_img = imgs[0]

    def test_predict_returns_top_k(self):
        assert len(self.clf.predict(self.test_img)) == TOP_K

    def test_predict_chars_are_arabic(self):
        for char, _ in self.clf.predict(self.test_img):
            assert is_arabic_char(char)

    def test_predict_with_dots(self):
        dots = [Dot(cx=10, cy=3, cluster_size=1, position="above")]
        result = self.clf.predict(self.test_img, dot_list=dots)
        assert len(result) == TOP_K
