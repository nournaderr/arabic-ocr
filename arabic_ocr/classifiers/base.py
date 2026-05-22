from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from arabic_ocr.config import TOP_K
from arabic_ocr.segment.dots import Dot


class BaseClassifier(ABC):
    """Abstract base for all Arabic character classifiers.

    predict / predict_batch are the only interface the pipeline calls.
    Each method handles feature extraction internally so callers never
    touch feature vectors directly.
    """

    @abstractmethod
    def predict(
        self,
        char_img: np.ndarray,
        dot_list: Optional[list[Dot]] = None,
    ) -> list[tuple[str, float]]:
        """Classify one character image.

        Returns a list of (arabic_char, confidence) of length TOP_K,
        sorted by confidence descending.
        """

    @abstractmethod
    def predict_batch(
        self,
        char_imgs: list[np.ndarray],
        dot_lists: Optional[list[Optional[list[Dot]]]] = None,
    ) -> list[list[tuple[str, float]]]:
        """Classify a batch of character images.

        Returns one top-K list per image.
        """

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier. X: (N, feat_dim), y: Arabic char strings."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model from disk."""
