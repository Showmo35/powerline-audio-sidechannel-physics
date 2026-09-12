#!/usr/bin/env python3
"""
whisper_features.py — adapt an AM-sideband mel into Whisper encoder input.

Whisper's encoder consumes a log-mel of shape [80, 3000] (80 bins, 30 s @ 100
fps, log-compressed and normalized).  Our stored features are 80-bin *linear*
mels at ~80.2 fps (22050 Hz / hop 275) and variable length.  This converts ours
to Whisper's tensor convention so the mel can be fed STRAIGHT into the encoder —
no Griffin-Lim audio round-trip, which is the whole point of storing mels.

Note on the mel *filterbank*: ours is htk@22050 (f_max 11025), Whisper's is
slaney@16k (f_max 8000) — a different 80-bin layout.  Because we fine-tune the
encoder end-to-end on a consistent front-end, the model adapts to our bin layout;
matching Whisper's exact filterbank (rebuild at 16 kHz) is the obvious lever if
results stall.
"""

import numpy as np

WHISPER_N_FRAMES = 3000
WHISPER_FPS = 100.0


def mel_to_input_features(mel_native: np.ndarray, native_fps: float,
                          n_frames: int = WHISPER_N_FRAMES) -> np.ndarray:
    """[80, T] linear mel → [80, 3000] log, time-resampled, normalized, padded."""
    mel = mel_native.astype(np.float32)
    L = np.log10(np.maximum(mel, 1e-8))
    L = np.maximum(L, L.max() - 8.0)               # clamp 8 decades (Whisper-style)
    L = (L - L.mean()) / (L.std() + 1e-6)          # per-utt standardize → ~[-1,1]

    # resample time axis native_fps → 100 fps
    n_mels, T = L.shape
    new_T = max(1, int(round(T * WHISPER_FPS / native_fps)))
    xi = np.linspace(0, T - 1, new_T)
    idx = np.arange(T)
    Lr = np.stack([np.interp(xi, idx, L[b]) for b in range(n_mels)]).astype(np.float32)

    floor = float(Lr.min())                        # silence pad value
    out = np.full((n_mels, n_frames), floor, dtype=np.float32)
    k = min(new_T, n_frames)
    out[:, :k] = Lr[:, :k]
    return out
