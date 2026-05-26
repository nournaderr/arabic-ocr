import pickle
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from arabic_ocr.config import MODELS_DIR
from arabic_ocr.utils import arabic_utils as _au


class LearnedReranker:
    """Learned re-ranker for character candidates.

    At inference, if a trained model exists at MODELS_DIR / 'reranker.pkl'
    the model is used to score candidate (label, base_conf) pairs. Otherwise
    the object falls back to a simple dot-aware heuristic used previously.
    """

    def __init__(self):
        self.model_path = Path(MODELS_DIR) / "reranker.pkl"
        self.model = None
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None

    def rerank(
        self,
        candidates: List[Tuple[str, float]],
        observed_dot_feat: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Return a re-ranked, normalized list of (label, score).

        Args:
            candidates: classifier output list (label, conf)
            observed_dots: (dots_above, dots_below)
        """
        if not candidates:
            return candidates

        # Interpret observed dot features
        if observed_dot_feat is None:
            obs_above = 0.0
            obs_below = 0.0
            obs_has = 0.0
            obs_spread = 0.0
        else:
            # expected layout from features.dot_features: [above, below, has_any, spread]
            obs_above = float(observed_dot_feat[0])
            obs_below = float(observed_dot_feat[1])
            obs_has = float(observed_dot_feat[2])
            obs_spread = float(observed_dot_feat[3])

        if self.model is None:
            # Fallback: simple dot-aware heuristic (preserve behaviour)
            boost = 0.20
            penalty = 0.10
            reranked = []
            for lab, conf in candidates:
                base = lab.rsplit("_", 1)[0]
                expected = _au._LETTER_DOT_COUNTS.get(base)
                new_conf = conf
                if expected is not None:
                    if expected == (int(obs_above), int(obs_below)):
                        new_conf = conf + boost
                    else:
                        new_conf = max(0.0, conf - penalty)
                reranked.append((lab, new_conf))
            reranked.sort(key=lambda t: t[1], reverse=True)
            total = sum(c for _, c in reranked) or 1.0
            return [(lab, c / total) for lab, c in reranked]

        # Use model: build richer feature matrix per candidate.
        # Feature vector: [conf, expected_above, expected_below, obs_above, obs_below, obs_has, obs_spread]
        X = []
        for lab, conf in candidates:
            base = lab.rsplit("_", 1)[0]
            exp = _au._LETTER_DOT_COUNTS.get(base, (0, 0))
            X.append([
                float(conf),
                float(exp[0]),
                float(exp[1]),
                obs_above,
                obs_below,
                obs_has,
                obs_spread,
            ])
        X = np.asarray(X, dtype=np.float32)
        try:
            probs = self.model.predict_proba(X)[:, 1]
        except Exception:
            probs = self.model.predict(X)
        # Combine model score with original confidence (weighted average)
        if probs is None:
            reranked = candidates
        else:
            alpha = 0.6  # weight for original classifier confidence
            reranked = []
            for (lab, conf), p in zip(candidates, probs):
                combined = float(alpha * conf + (1.0 - alpha) * float(p))
                reranked.append((lab, combined))

        # Normalise
        total = sum(c for _, c in reranked) or 1.0
        return [(lab, c / total) for lab, c in reranked]


RERANKER = LearnedReranker()


def save_model(model, path: Optional[Path] = None) -> None:
    p = Path(path) if path is not None else Path(MODELS_DIR) / "reranker.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(model, f)
