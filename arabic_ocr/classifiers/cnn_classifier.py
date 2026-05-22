from pathlib import Path
from typing import Optional

import numpy as np

from arabic_ocr.config import TOP_K, NORM_SIZE
from arabic_ocr.segment.dots import Dot
from arabic_ocr.features.normalize import normalize
from arabic_ocr.features.dot_features import dot_features
from .base import BaseClassifier


class CNNClassifier(BaseClassifier):
    """Lightweight CNN operating on raw 32×32 pixels (no handcrafted features).

    Architecture:
        Conv(1→32) BN ReLU MaxPool  → 16×16
        Conv(32→64) BN ReLU MaxPool  → 8×8
        Conv(64→128) BN ReLU AdaptiveAvgPool(4×4)
        Flatten(2048) → Linear(512) ReLU Dropout(0.4)
        Concat dot_features(4) → Linear(516, n_classes)

    Dot features are concatenated just before the final classification layer
    so spatial conv features and dot counts are fused at decision time.
    """

    def __init__(self, n_classes: int = 28, lr: float = 1e-3, epochs: int = 40):
        self.n_classes = n_classes
        self.lr        = lr
        self.epochs    = epochs
        self.model     = None
        self.classes_: Optional[np.ndarray] = None

    # ── PyTorch model ─────────────────────────────────────────────────────────

    def _build_model(self):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
                    nn.ReLU(), nn.MaxPool2d(2),

                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                    nn.ReLU(), nn.MaxPool2d(2),

                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                    nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                )
                self.fc1     = nn.Linear(128 * 4 * 4, 512)
                self.relu    = nn.ReLU()
                self.dropout = nn.Dropout(0.4)
                self.fc2     = nn.Linear(512 + 4, n_classes)  # +4 for dot_features

            def forward(self, x, dots):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = torch.cat([x, dots], dim=1)
                return self.fc2(x)

        import torch
        return _Net(self.n_classes)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, 32, 32) uint8 pixel arrays; y: Arabic char strings."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import LabelEncoder

        self._encoder = LabelEncoder()
        self._encoder.fit(y)
        self.classes_ = self._encoder.classes_
        self.n_classes = len(self.classes_)
        y_idx = self._encoder.transform(y)

        imgs  = torch.tensor(X / 255.0, dtype=torch.float32).unsqueeze(1)  # (N,1,H,W)
        # X rows are raw pixels, no dot features in train array — use zeros
        d_feat = torch.zeros(len(X), 4, dtype=torch.float32)
        labels = torch.tensor(y_idx, dtype=torch.long)

        self.model = self._build_model()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(imgs, d_feat, labels), batch_size=64, shuffle=True
        )

        self.model.train()
        for _ in range(self.epochs):
            for xb, db, yb in loader:
                opt.zero_grad()
                criterion(self.model(xb, db), yb).backward()
                opt.step()
            scheduler.step()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        char_img: np.ndarray,
        dot_list: Optional[list[Dot]] = None,
    ) -> list[tuple[str, float]]:
        return self.predict_batch([char_img], [dot_list])[0]

    def predict_batch(
        self,
        char_imgs: list[np.ndarray],
        dot_lists: Optional[list] = None,
    ) -> list[list[tuple[str, float]]]:
        import torch
        import torch.nn.functional as F

        if dot_lists is None:
            dot_lists = [None] * len(char_imgs)

        norm_imgs = np.stack([normalize(img) for img in char_imgs])
        imgs_t = torch.tensor(norm_imgs / 255.0, dtype=torch.float32).unsqueeze(1)
        dots_t = torch.tensor(
            np.stack([dot_features(d) for d in dot_lists]),
            dtype=torch.float32,
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(imgs_t, dots_t)
            proba  = F.softmax(logits, dim=1).numpy()

        return [self._top_k(row) for row in proba]

    def _top_k(self, proba: np.ndarray) -> list[tuple[str, float]]:
        k = min(TOP_K, len(proba))
        indices = np.argsort(proba)[::-1][:k]
        return [(str(self.classes_[i]), float(proba[i])) for i in indices]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "classes":     self.classes_,
            "n_classes":   self.n_classes,
        }, path)

    def load(self, path: Path) -> None:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.n_classes = data["n_classes"]
        self.classes_  = data["classes"]
        self.model     = self._build_model()
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
