import numpy as np

from arabic_ocr.config import BIGRAM_WEIGHT


def beam_search_decode(
    candidates_per_position: list[list[tuple[str, float]]],
    lm,
    beam_width: int = 5,
) -> list[str]:
    """Beam search decoding over per-position top-K candidate lists.

    More efficient than Viterbi for long sequences: keeps only the top
    beam_width hypotheses at each step rather than the full trellis.

    Args:
        candidates_per_position: list of [(char, prob), ...] per position.
        lm: ArabicLanguageModel with ._bigram_score() method.
        beam_width: number of hypotheses to keep at each step.

    Returns:
        Best character sequence as list of strings.
    """
    if not candidates_per_position:
        return []

    # Each beam: (accumulated_log_score, [chars_so_far])
    beams: list[tuple[float, list[str]]] = [(0.0, [])]

    for candidates in candidates_per_position:
        new_beams: list[tuple[float, list[str]]] = []

        for acc_score, path in beams:
            for char, prob in candidates:
                log_emit = np.log(max(prob, 1e-9))

                if path:
                    prev = path[-1]
                    bigram_p = lm.bigrams.get(prev, {}).get(char, 1e-6)
                    lm_score = np.log(bigram_p) * BIGRAM_WEIGHT
                else:
                    lm_score = 0.0

                new_score = acc_score + log_emit + lm_score
                new_beams.append((new_score, path + [char]))

        # Prune to top beam_width
        new_beams.sort(key=lambda t: t[0], reverse=True)
        beams = new_beams[:beam_width]

    return beams[0][1] if beams else []
