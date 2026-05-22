import numpy as np

from arabic_ocr.config import BIGRAM_WEIGHT, MIN_CONF


def viterbi_decode(
    candidates_per_position: list[list[tuple[str, float]]],
    lm,
) -> tuple[list[str], float]:
    """Viterbi decoding over a sequence of per-position top-K candidates.

    State:      (position, character)
    Emission:   log(classifier_probability)
    Transition: BIGRAM_WEIGHT * log(bigram_prob(prev → curr))
    Combined:   log(emission) + transition

    Paper: Viterbi IEEE Trans. 1967 — optimal sequence decoding.

    Returns:
        (best_character_sequence, total_log_score)
    """
    if not candidates_per_position:
        return [], 0.0

    T = len(candidates_per_position)

    # dp[t][char] = (best_log_score, prev_char | None)
    dp: list[dict[str, tuple[float, str | None]]] = [{}]

    for char, prob in candidates_per_position[0]:
        dp[0][char] = (np.log(max(prob, 1e-9)), None)

    if not dp[0]:
        char, prob = candidates_per_position[0][0]
        dp[0][char] = (np.log(max(prob, 1e-9)), None)

    for t in range(1, T):
        dp.append({})
        for char, prob in candidates_per_position[t]:
            log_emit = np.log(max(prob, 1e-9))
            best_score = -np.inf
            best_prev: str | None = None

            for prev_char, (prev_score, _) in dp[t - 1].items():
                bigram_p = lm.bigrams.get(prev_char, {}).get(char, 1e-6)
                score = prev_score + log_emit + BIGRAM_WEIGHT * np.log(bigram_p)
                if score > best_score:
                    best_score = score
                    best_prev  = prev_char

            dp[t][char] = (best_score, best_prev)

        if not dp[t]:
            char, prob = candidates_per_position[t][0]
            dp[t][char] = (np.log(max(prob, 1e-9)), None)

    # Back-track
    last       = dp[T - 1]
    best_last  = max(last, key=lambda c: last[c][0])
    final_score = last[best_last][0]

    path = [best_last]
    for t in range(T - 1, 0, -1):
        _, prev = dp[t][path[-1]]
        path.append(prev if prev is not None else "")
    path.reverse()

    return path, float(final_score)
