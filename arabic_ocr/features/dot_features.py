import numpy as np

from arabic_ocr.segment.dots import Dot


def dot_features(dot_list: list[Dot] | None) -> np.ndarray:
    """4-d feature vector encoding dot information for an Arabic character.

    Vector layout:
        [0] dot_count_above     — dots above baseline (ت=2, ث=3, ن=1)
        [1] dot_count_below     — dots below baseline (ب=1, ي=2)
        [2] has_any_dot         — 0 or 1
        [3] dot_horizontal_spread — std of dot x-centroids (0 if ≤1 dot)

    Key discriminator for: ب(1↓) ت(2↑) ث(3↑) ج(1↓) ح(0) خ(1↑) ن(1↑) ي(2↓)
    """
    if not dot_list:
        return np.zeros(4, dtype=np.float32)

    above = sum(d.cluster_size for d in dot_list if d.position == "above")
    below = sum(d.cluster_size for d in dot_list if d.position == "below")
    has_dot = 1.0 if dot_list else 0.0

    xs = [d.cx for d in dot_list]
    spread = float(np.std(xs)) if len(xs) > 1 else 0.0

    return np.array([above, below, has_dot, spread], dtype=np.float32)
