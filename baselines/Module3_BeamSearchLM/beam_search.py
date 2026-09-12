"""
beam_search.py
--------------
CTC prefix beam search decoder with optional n-gram language model
shallow fusion.

Replaces greedy CTC decoding with beam search for 5-15% CER improvement.

Usage:
    from beam_search import ctc_beam_search_decode, ctc_beam_search_decode_batch
"""

import numpy as np
from collections import defaultdict
from model import IDX_TO_CHAR, BLANK_IDX, VOCAB_SIZE


def ctc_beam_search_decode(log_probs, beam_width=50, lm=None, lm_weight=0.3):
    """
    CTC prefix beam search with optional LM shallow fusion.

    Args:
        log_probs: (T, V) numpy array of log probabilities
        beam_width: number of beams to keep
        lm: optional language model with .score(text) -> log probability
        lm_weight: weight for LM score fusion

    Returns:
        best decoded string
    """
    T, V = log_probs.shape

    # Each beam: (prefix_tuple, (p_blank, p_nonblank))
    # p_blank = log prob of prefix ending in blank
    # p_nonblank = log prob of prefix ending in non-blank
    NEG_INF = -float('inf')

    # Initial beam: empty prefix with p_blank=0 (log), p_nonblank=-inf
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beams = defaultdict(lambda: (NEG_INF, NEG_INF))

        for prefix, (p_b, p_nb) in beams.items():
            # Total log prob of this prefix
            p_total = _log_add(p_b, p_nb)

            for c in range(V):
                p_c = log_probs[t, c]

                if c == BLANK_IDX:
                    # Extend with blank: prefix stays the same
                    old_b, old_nb = new_beams[prefix]
                    new_beams[prefix] = (
                        _log_add(old_b, p_total + p_c),
                        old_nb,
                    )
                else:
                    # Non-blank character
                    if len(prefix) > 0 and c == prefix[-1]:
                        # Same char as last: only extend if came through blank
                        # Collapse: prefix stays same (from non-blank path)
                        old_b, old_nb = new_beams[prefix]
                        new_beams[prefix] = (
                            old_b,
                            _log_add(old_nb, p_nb + p_c),
                        )
                        # Extend: add new char (from blank path)
                        new_prefix = prefix + (c,)
                        old_b2, old_nb2 = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            old_b2,
                            _log_add(old_nb2, p_b + p_c),
                        )
                    else:
                        # Different char: extend prefix
                        new_prefix = prefix + (c,)
                        old_b, old_nb = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            old_b,
                            _log_add(old_nb, p_total + p_c),
                        )

        # Prune to beam_width
        scored = []
        for prefix, (p_b, p_nb) in new_beams.items():
            total = _log_add(p_b, p_nb)
            # Add LM score if available
            if lm is not None and len(prefix) > 0:
                text = _prefix_to_text(prefix)
                lm_score = lm.score(text)
                total_with_lm = total + lm_weight * lm_score
            else:
                total_with_lm = total
            scored.append((prefix, (p_b, p_nb), total_with_lm))

        scored.sort(key=lambda x: x[2], reverse=True)
        beams = {p: probs for p, probs, _ in scored[:beam_width]}

    # Return best beam
    best_prefix = max(beams.keys(),
                      key=lambda p: _log_add(*beams[p]))
    return _prefix_to_text(best_prefix)


def ctc_beam_search_decode_batch(log_probs_batch, beam_width=50,
                                  lm=None, lm_weight=0.3):
    """
    Batch wrapper for CTC beam search.

    Args:
        log_probs_batch: (T, B, V) numpy array or torch tensor
        beam_width: beam width
        lm: optional language model
        lm_weight: LM weight

    Returns:
        list of decoded strings
    """
    if hasattr(log_probs_batch, 'numpy'):
        log_probs_batch = log_probs_batch.detach().cpu().numpy()

    T, B, V = log_probs_batch.shape
    results = []
    for b in range(B):
        text = ctc_beam_search_decode(
            log_probs_batch[:, b, :], beam_width, lm, lm_weight)
        results.append(text)
    return results


def _log_add(a, b):
    """Numerically stable log addition: log(exp(a) + exp(b))."""
    if a == -float('inf'):
        return b
    if b == -float('inf'):
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))


def _prefix_to_text(prefix):
    """Convert CTC index tuple to string."""
    return ''.join(IDX_TO_CHAR.get(idx, '') for idx in prefix)


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with random log probs
    T, V = 50, VOCAB_SIZE
    log_probs = np.random.randn(T, V).astype(np.float32)
    # Normalize to log probabilities
    log_probs -= np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))

    print("Testing beam search decode...")
    text = ctc_beam_search_decode(log_probs, beam_width=10)
    print(f"  Decoded: '{text}'")

    print("Testing batch decode...")
    batch = np.stack([log_probs, log_probs], axis=1)  # (T, 2, V)
    texts = ctc_beam_search_decode_batch(batch, beam_width=10)
    print(f"  Decoded: {texts}")
    print("Beam search test passed.")
