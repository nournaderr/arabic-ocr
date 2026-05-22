import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from arabic_ocr.config import TOP_K
from arabic_ocr.segment.dots import Dot
from arabic_ocr.features import extract
from .base import BaseClassifier


class RFClassifier(BaseClassifier):
    """Random Forest baseline — no scaling or hyperparameter tuning needed.

    Uses the same handcrafted feature vector as SVMClassifier.
    Useful as a fast baseline to validate the feature pipeline.
    """

    def __init__(self, n_estimators: int = 300, n_jobs: int = -1):
        self.rf      = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None,
            n_jobs=n_jobs, random_state=42,
        )
        self.encoder = LabelEncoder()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.encoder.fit(y)
        self.rf.fit(X, self.encoder.transform(y))

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        char_img: np.ndarray,
        dot_list: Optional[list[Dot]] = None,
    ) -> list[tuple[str, float]]:
        feat = extract(char_img, dot_list).reshape(1, -1)
        proba = self.rf.predict_proba(feat)[0]
        return self._top_k(proba)

    def predict_batch(
        self,
        char_imgs: list[np.ndarray],
        dot_lists: Optional[list[Optional[list[Dot]]]] = None,
    ) -> list[list[tuple[str, float]]]:
        if dot_lists is None:
            dot_lists = [None] * len(char_imgs)
        feats = np.stack([extract(img, d) for img, d in zip(char_imgs, dot_lists)])
        probas = self.rf.predict_proba(feats)
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
            pickle.dump({"rf": self.rf, "encoder": self.encoder}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.rf      = data["rf"]
        self.encoder = data["encoder"]
