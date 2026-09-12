#!/usr/bin/env python3
"""
asr.py — Whisper transcription + a dependency-free Word Error Rate.

The reference (clean LibriSpeech at 16 kHz) transcribed by Whisper is treated as
pseudo ground-truth; WER of the powerline reconstruction against it measures how
much intelligible speech survived the powerline channel.  (Exact LibriSpeech
transcripts can be substituted later once per-utterance time alignment exists.)
"""

import re
import numpy as np


# ── Whisper ────────────────────────────────────────────────────────────────────

def load_whisper(name: str, device: str = 'cuda'):
    import whisper
    print(f'[asr] loading Whisper "{name}" on {device}')
    return whisper.load_model(name, device=device)


def transcribe(model, audio_16k: np.ndarray, language: str = 'en') -> dict:
    """audio_16k: 1-D float32 at 16 kHz in [-1, 1]. Returns whisper result dict."""
    audio = audio_16k.astype(np.float32)
    return model.transcribe(audio, language=language, fp16=True, verbose=False)


# ── text normalization + WER ────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^a-z0-9'\s]")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(' ', s)
    return ' '.join(s.split())


def wer(ref: str, hyp: str) -> dict:
    """Word error rate via Levenshtein on word tokens. Returns dict with details."""
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    n, m = len(r), len(h)
    # DP edit distance with S/D/I backtrace counts
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1,        # deletion
                          d[i, j - 1] + 1,        # insertion
                          d[i - 1, j - 1] + cost) # sub / match
    # backtrace for S/D/I
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if r[i-1] == h[j-1] else 1):
            if r[i - 1] != h[j - 1]:
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    rate = (S + D + I) / max(n, 1)
    return {'wer': rate, 'sub': S, 'del': D, 'ins': I,
            'ref_words': n, 'hyp_words': m}


def corpus_wer(refs, hyps) -> dict:
    """Corpus-level WER: total edits / total reference words over many pairs."""
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        w = wer(r, h)
        S += w['sub']; D += w['del']; I += w['ins']; N += w['ref_words']
    return {'wer': (S + D + I) / max(N, 1), 'sub': S, 'del': D, 'ins': I,
            'ref_words': N, 'n_utts': len(refs)}
