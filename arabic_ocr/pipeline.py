from pathlib import Path
import numpy as np

from arabic_ocr.preprocess import preprocess
from arabic_ocr.segment import segment, CharCrop
from arabic_ocr.classifiers import get_classifier, BaseClassifier
from arabic_ocr.postprocess import ArabicLanguageModel, postprocess
from arabic_ocr.postprocess.reranker import RERANKER
from arabic_ocr.utils.image_io import load_image, resize_if_large
from arabic_ocr.utils.arabic_utils import (
    filter_candidates_by_position,
    filter_candidates_by_dots,
)
from arabic_ocr.utils import arabic_utils as _au
from arabic_ocr import config as cfg
from arabic_ocr.utils.visualize import (
    draw_lines, draw_paws, draw_chars, draw_dots, save_debug_visualization,
)
from arabic_ocr.config import OUTPUT_DIR
import logging

logger = logging.getLogger(__name__)


class ArabicOCRPipeline:
    """End-to-end Arabic OCR pipeline.

    Stages:
        1. preprocess  — enhance, binarize, deskew, filter, pad
        2. segment     — lines → PAWs → dots → chars → CharCrop list
        3. classify    — predict() on every CharCrop; fills .candidates
        4. postprocess — Viterbi + bigram LM → final Arabic string

    Usage::

        pipe = ArabicOCRPipeline(classifier='svm', debug=True)
        text = pipe.run('scan.jpg')
    """

    def __init__(
        self,
        classifier: str = "svm",
        debug: bool = False,
    ):
        self.classifier: BaseClassifier = get_classifier(classifier)
        self.lm = ArabicLanguageModel()
        self.debug = debug

        if debug:
            import arabic_ocr.config as cfg
            cfg.DEBUG = True

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image_path: str | Path, frame_number: int = 0) -> str:
        """Load image from disk, run full pipeline, return Arabic text."""
        img = load_image(image_path)
        return self._process(img, frame_number=frame_number)

    def run_array(self, img: np.ndarray, frame_number: int = 0) -> str:
        """Accept a pre-loaded numpy array (BGR)."""
        return self._process(img, frame_number=frame_number)

    def run_batch(self, image_paths: list[str | Path]) -> list[str]:
        """Recognise multiple images; returns one string per image."""
        return [self.run(p, frame_number=i) for i, p in enumerate(image_paths)]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process(self, img: np.ndarray, frame_number: int = 0) -> str:
        img = resize_if_large(img)

        # Stage 1 — preprocess
        binary = preprocess(img, frame_number=frame_number)

        # Stage 2 — segment
        char_crops: list[CharCrop] = segment(binary)

        if not char_crops:
            return ""

        if self.debug:
            self._save_debug(binary, char_crops, frame_number)

        # Stage 3 — classify (batch for efficiency)
        imgs      = [c.img  for c in char_crops]
        dot_lists = [c.dots for c in char_crops]
        try:
            all_candidates = self.classifier.predict_batch(imgs, dot_lists)
        except Exception:
            logger.exception("Classifier.predict_batch failed — falling back to undetected candidates")
            # Provide a safe default: one undetected candidate per position.
            all_candidates = [[("", 1.0)] for _ in imgs]

        # Truncate or extend with defaults so we always have one list per crop
        if len(all_candidates) != len(char_crops):
            logger.warning("Classifier returned %d candidate lists for %d crops; normalising.",
                           len(all_candidates), len(char_crops))
            all_candidates = list(all_candidates[:len(char_crops)])
            while len(all_candidates) < len(char_crops):
                all_candidates.append([("", 1.0)])

        # Assign candidates to each CharCrop and apply the reranker (learned or heuristic)
        from arabic_ocr.features.dot_features import dot_features
        for crop, cands in zip(char_crops, all_candidates):
            df = dot_features(crop.dots)
            crop.candidates = RERANKER.rerank(cands, df.tolist())

        # Stage 4 — postprocess
        return postprocess(char_crops, self.lm)

    def _save_debug(
        self,
        binary: np.ndarray,
        crops: list[CharCrop],
        frame: int,
    ) -> None:
        from itertools import groupby
        debug_dir = OUTPUT_DIR / "debug" / f"{frame:04d}"

        from arabic_ocr.segment.lines import segment_lines
        lines = segment_lines(binary)
        line_bounds = [(y1, y2) for y1, y2, _ in lines]

        # Individual character boxes
        char_boxes = [
            (c.abs_x, c.abs_y,
             c.abs_x + c.img.shape[1], c.abs_y + c.img.shape[0])
            for c in crops
        ]

        # PAW boxes: union of all char boxes sharing (line_idx, paw_idx)
        paw_boxes: list[tuple[int, int, int, int]] = []
        for _, group in groupby(crops, key=lambda c: (c.line_idx, c.paw_idx)):
            grp = list(group)
            x1 = min(c.abs_x for c in grp)
            y1 = min(c.abs_y for c in grp)
            x2 = max(c.abs_x + c.img.shape[1] for c in grp)
            y2 = max(c.abs_y + c.img.shape[0] for c in grp)
            paw_boxes.append((x1, y1, x2, y2))

        all_dots = [d for c in crops for d in c.dots]

        save_debug_visualization(draw_lines(binary, line_bounds),
                                 "lines", debug_dir)
        save_debug_visualization(draw_paws(binary, paw_boxes),
                                 "paws", debug_dir)
        save_debug_visualization(draw_chars(binary, char_boxes),
                                 "chars", debug_dir)
        save_debug_visualization(draw_dots(binary, all_dots),
                                 "dots", debug_dir)


def _filter_candidates_with_fallback(
    candidates: list[tuple[str, float]],
    position: str,
    dots_above: int,
    dots_below: int,
) -> list[tuple[str, float]]:
    """Apply positional and dot filters, but keep the original candidates if all are removed."""
    # Apply position filter first (strict): keeps only matching positional forms
    filtered = filter_candidates_by_position(candidates, position)

    # Re-rank by dot agreement rather than strictly filtering so LM can still
    # correct ambiguous cases. Boost candidates whose expected dot counts
    # match the observed dots; penalise those that explicitly disagree.
    observed = (dots_above, dots_below)
    boost = getattr(cfg, "DOT_RERANK_BOOST", 0.20)
    penalty = getattr(cfg, "DOT_RERANK_PENALTY", 0.10)

    def expected_for_label(label: str):
        base = label.rsplit("_", 1)[0]
        return _au._LETTER_DOT_COUNTS.get(base)

    reranked = []
    for label, conf in filtered:
        expected = expected_for_label(label)
        new_conf = conf
        if expected is not None:
            if expected == observed:
                new_conf = conf + boost
            else:
                # if we observed no dots but candidate expects dots, penalise
                new_conf = max(0.0, conf - penalty)
        reranked.append((label, new_conf))

    # Keep original ordering for equal scores; sort by adjusted confidence
    reranked.sort(key=lambda t: t[1], reverse=True)

    # If dot filtering would have returned empty (no sensible matches), fall back
    # to the original candidate list; otherwise return reranked top list but keep
    # the same length as input filtered list.
    if not reranked:
        return candidates
    # Normalize confidences to sum to 1 for downstream expectations
    total = sum(c for _, c in reranked) or 1.0
    normalized = [(lab, c / total) for lab, c in reranked]
    return normalized
