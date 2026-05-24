from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .lines import segment_lines
from .paws import segment_paws
from .dots import separate_dots, Dot
from .chars import segment_chars, _estimate_ah


@dataclass
class CharCrop:
    img:        np.ndarray
    abs_x:      int
    abs_y:      int
    line_idx:   int
    paw_idx:    int
    char_idx:   int
    dots:       list[Dot] = field(default_factory=list)
    position:   str = "isolated"          # initial / medial / final / isolated
    candidates: list[tuple[str, float]] = field(default_factory=list)  # set by classifier


def segment(binary: np.ndarray) -> list[CharCrop]:
    """Full segmentation pipeline: lines → PAWs → dots → chars.

    Returns flat list of CharCrop sorted by (line_idx, paw_idx, char_idx DESC).
    """
    all_chars: list[CharCrop] = []

    lines = segment_lines(binary)

    for line_idx, (ly1, ly2, line_img) in enumerate(lines):
        ah = _estimate_ah(line_img)
        paws = segment_paws(line_img, ah=ah)

        for paw_idx, (px1, px2, paw_img) in enumerate(paws):
            body_img, dot_list = separate_dots(paw_img, ah)
            char_boxes = segment_chars(body_img, px1, ly1, ah)

            # Assign position tags (initial/medial/final/isolated)
            n_chars = len(char_boxes)
            for char_idx, (cx1, cy1, cx2, cy2) in enumerate(char_boxes):
                local_x  = cx1 - px1
                local_y  = cy1 - ly1
                char_w   = cx2 - cx1
                char_h   = cy2 - cy1
                if char_h <= 0 or char_w <= 0:
                    continue

                # Use paw_img (original with dots) not body_img (dots erased).
                # HMDB training data contains full characters including dots,
                # so the classifier must receive the same visual input.
                # body_img is still used above for segment_chars (valley-finding
                # is cleaner without floating dots) and dot_list carries the
                # dot metadata separately for the dot_features vector.
                char_crop_img = paw_img[
                    max(0, local_y): local_y + char_h,
                    max(0, local_x): local_x + char_w,
                ]

                pos = _position_tag(char_idx, n_chars)

                # Reassociate dots by x-overlap; small margin catches dots
                # whose centroid lands just outside the character bounds
                # (common with ي whose tail extends beyond its dots).
                dot_margin = max(4, int(ah * 0.5))
                char_dots = [
                    d for d in dot_list
                    if (cx1 - dot_margin) <= d.cx <= (cx2 + dot_margin)
                ]

                all_chars.append(CharCrop(
                    img=char_crop_img,
                    abs_x=cx1,
                    abs_y=cy1,
                    line_idx=line_idx,
                    paw_idx=paw_idx,
                    char_idx=char_idx,
                    dots=char_dots,
                    position=pos,
                ))

    # Sort: line asc, paw asc, char ASC within paw.
    # char_idx=0 is rightmost (initial form, first in Arabic reading order).
    # Ascending order puts initial chars first → correct Unicode logical order.
    all_chars.sort(key=lambda c: (c.line_idx, c.paw_idx, c.char_idx))
    return all_chars


def _position_tag(char_idx: int, n_chars: int) -> str:
    if n_chars == 1:
        return "isolated"
    # chars are sorted descending by x (right-to-left):
    #   char_idx=0       = rightmost = first in Arabic reading order = initial
    #   char_idx=n_chars-1 = leftmost  = last  in Arabic reading order = final
    if char_idx == 0:
        return "initial"
    if char_idx == n_chars - 1:
        return "final"
    return "medial"


__all__ = ["segment", "CharCrop"]
