#!/usr/bin/env python3
"""
text.py — character vocab (shared with Modules 3-7) + word-level WER.

VOCAB: blank=0, space=1, a..z=2..27, apostrophe=28  → CTC vocab size 29.
Attention decoder (M6) additionally uses SOS=29, EOS=30 → dec vocab 31.
"""

VOCAB = {' ': 1, "'": 28}
for _i, _c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    VOCAB[_c] = _i + 2
VOCAB_SIZE = 29
BLANK_IDX  = 0
SOS_IDX    = 29
EOS_IDX    = 30
DEC_VOCAB_SIZE = 31

IDX_TO_CHAR = {0: '', 1: ' ', 28: "'"}
for _i, _c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    IDX_TO_CHAR[_i + 2] = _c


def normalize(text):
    """Lowercase; keep only chars in the vocab."""
    return ''.join(c for c in text.lower() if c in VOCAB)


def encode(text):
    return [VOCAB[c] for c in normalize(text)]


def ctc_greedy_decode(logits):
    """logits: (T, B, V) → list[str] (collapse repeats, drop blanks)."""
    preds = logits.argmax(dim=-1).T   # (B, T)
    out = []
    for seq in preds.tolist():
        chars, prev = [], BLANK_IDX
        for idx in seq:
            if idx != prev and idx != BLANK_IDX:
                chars.append(IDX_TO_CHAR.get(idx, ''))
            prev = idx
        out.append(''.join(chars))
    return out


def _edit_distance(a, b):
    """Levenshtein on token lists."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def corpus_wer(refs, hyps):
    """Word-level WER over a corpus. Returns dict(wer, n_words, n_utts)."""
    tot_err, tot_words = 0, 0
    for r, h in zip(refs, hyps):
        rw, hw = normalize(r).split(), normalize(h).split()
        tot_err += _edit_distance(rw, hw)
        tot_words += len(rw)
    wer = tot_err / max(1, tot_words)
    return {'wer': wer, 'n_words': tot_words, 'n_utts': len(refs)}
