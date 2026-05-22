import unicodedata
from typing import TYPE_CHECKING

from .language_model import ArabicLanguageModel
from .viterbi import viterbi_decode
from .beam_search import beam_search_decode
from .dawg import DawgNode, build_dawg, dawg_search, save_dawg, load_dawg
from arabic_ocr.utils.arabic_utils import hmdb_label_to_unicode

if TYPE_CHECKING:
    from arabic_ocr.segment import CharCrop

_TATWEEL = "ـ"


def _to_unicode(label: str) -> str:
    """Convert a classifier label to a clean Unicode character.

    Handles both HMDB-style labels ("Meem_Isolated" → "م") and
    labels that are already plain Unicode characters (passed through as-is).
    Tatweels used as positional markers in HMDB values are stripped.
    """
    converted = hmdb_label_to_unicode(label)
    if converted:
        return converted.replace(_TATWEEL, "")
    return label


def postprocess(
    char_crops: list["CharCrop"],
    lm: ArabicLanguageModel,
) -> str:
    """Convert classified CharCrops into a final Arabic string.

    Per-word pipeline:
      1. Group CharCrops by (line_idx, paw_idx) — one group = one PAW.
      2. Viterbi decode the candidate list for each PAW.
      3. Convert each decoded label to Unicode (handles HMDB label format).
      4. rescore_word on the decoded word.
      5. Join all words with spaces; words are already in RTL reading order.
      6. NFC-normalise the result.
    """
    if not char_crops:
        return ""

    from itertools import groupby

    # First group by line_idx to preserve physical line breaks
    lines_text: list[str] = []
    for _, line_group in groupby(char_crops, key=lambda c: c.line_idx):
        line_crops = list(line_group)
        words: list[str] = []
        for _, paw_group in groupby(line_crops, key=lambda c: c.paw_idx):
            crops = list(paw_group)
            candidates_per_pos = [
                getattr(c, "candidates", [("", 1.0)]) for c in crops
            ]
            decoded_chars, _ = viterbi_decode(candidates_per_pos, lm)
            word = "".join(_to_unicode(ch) for ch in decoded_chars)
            lm.rescore_word(word, clf_conf=1.0)
            if word.strip():
                words.append(word)
        if words:
            # Words within a line are already RTL-ordered; join with space
            lines_text.append(" ".join(words))

    text = "\n".join(lines_text)
    return unicodedata.normalize("NFC", text)


__all__ = [
    "ArabicLanguageModel",
    "viterbi_decode",
    "beam_search_decode",
    "postprocess",
    "DawgNode", "build_dawg", "dawg_search", "save_dawg", "load_dawg",
]
