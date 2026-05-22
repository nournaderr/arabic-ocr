import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

from arabic_ocr.config import TOP_K, MODELS_DIR
from arabic_ocr.segment.dots import Dot
from arabic_ocr.features import extract
from .base import BaseClassifier


class SVMClassifier(BaseClassifier):
    """RBF-SVM with Platt-scaling probability estimates.

    Paper: El-Sherif & Abdelazim IJACSA 2012 — SVM for Arabic characters.
    Uses handcrafted feature vector (grid + HOG + contour + Zernike + dots).
    """

    def __init__(self, C: float = 10.0, gamma: str = "scale"):
        self.svm     = SVC(C=C, kernel="rbf", gamma=gamma, probability=True)
        self.scaler  = StandardScaler()
        self.encoder = LabelEncoder()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.encoder.fit(y)
        y_enc = self.encoder.transform(y)
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y_enc)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        char_img: np.ndarray,
        dot_list: Optional[list[Dot]] = None,
    ) -> list[tuple[str, float]]:
        feat = extract(char_img, dot_list).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        proba = self.svm.predict_proba(feat_scaled)[0]
        return self._top_k(proba)

    def predict_batch(
        self,
        char_imgs: list[np.ndarray],
        dot_lists: Optional[list[Optional[list[Dot]]]] = None,
    ) -> list[list[tuple[str, float]]]:
        if dot_lists is None:
            dot_lists = [None] * len(char_imgs)
        feats = np.stack([extract(img, d) for img, d in zip(char_imgs, dot_lists)])
        feats_scaled = self.scaler.transform(feats)
        probas = self.svm.predict_proba(feats_scaled)
        return [self._top_k(row) for row in probas]

    def _top_k(self, proba: np.ndarray) -> list[tuple[str, float]]:
        k = min(TOP_K, len(proba))
        indices = np.argsort(proba)[::-1][:k]
        return [(str(self.encoder.classes_[i]), float(proba[i])) for i in indices]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"svm": self.svm, "scaler": self.scaler,
                         "encoder": self.encoder}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.svm     = data["svm"]
        self.scaler  = data["scaler"]
        self.encoder = data["encoder"]
