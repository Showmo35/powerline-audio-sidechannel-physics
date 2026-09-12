#!/usr/bin/env python3
"""
labels.py — exact LibriSpeech utterance → (chunk, time) alignment.

`build_audio_chunks.py` made the reference WAVs by concatenating
`sorted(rglob("*.flac"))` (LibriSpeech is natively 16 kHz, so NO resampling) and
slicing at exactly 28_800_000 samples = 30 min per chunk, with no padding between
utterances.  Therefore every utterance's position is deterministic: the running
sum of FLAC frame counts gives its global sample range, and chunk N (1-indexed)
spans global samples [(N-1)·CH, N·CH).

This module rebuilds that mapping (cached to outputs/dataset/flac_index.json) and
exposes the utterances that fall inside any chunk, with their exact in-chunk start
/end times and ground-truth transcript text.
"""

import json
import os
from pathlib import Path

import soundfile as sf

from config import OUT_DIR, DATA_ROOT

CHUNK_SAMPLES = 28_800_000          # 30 min @ 16 kHz, must match build_audio_chunks.py
FLAC_ROOT = os.path.join(DATA_ROOT, 'LibriSpeech', 'train-clean-100')
INDEX_CACHE = os.path.join(OUT_DIR, 'dataset', 'flac_index.json')


def load_transcripts(flac_root: str = FLAC_ROOT) -> dict:
    """{utt_id: text} parsed from all *.trans.txt under the split."""
    out = {}
    for tp in Path(flac_root).rglob('*.trans.txt'):
        for line in open(tp):
            line = line.strip()
            if not line:
                continue
            uid, _, text = line.partition(' ')
            out[uid] = text
    return out


def build_index(flac_root: str = FLAC_ROOT, cache: str = INDEX_CACHE) -> list:
    """List of {utt_id, frames, g0, g1} in concatenation order. Cached to JSON."""
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    print(f'[labels] building FLAC index from {flac_root} …')
    flacs = sorted(Path(flac_root).rglob('*.flac'))
    index, g = [], 0
    for i, p in enumerate(flacs):
        info = sf.info(str(p))
        if info.samplerate != 16000:
            raise RuntimeError(f'{p} is {info.samplerate} Hz, expected 16000 '
                               '(build assumed no resample)')
        n = info.frames
        index.append({'utt_id': p.stem, 'frames': n, 'g0': g, 'g1': g + n})
        g += n
        if (i + 1) % 5000 == 0:
            print(f'  …{i+1}/{len(flacs)} flacs')
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, 'w') as f:
        json.dump(index, f)
    print(f'[labels] indexed {len(index)} utterances, {g} samples '
          f'({g/16000/3600:.2f} h) → {cache}')
    return index


def utterances_in_chunk(chunk_n: int, index: list, transcripts: dict,
                        full_only: bool = True) -> list:
    """Utterances overlapping chunk N (1-indexed) with in-chunk timing + text.

    Returns dicts: {utt_id, text, start_s, end_s, dur_s}. With full_only the
    utterance must lie entirely within the chunk (no boundary cuts).
    """
    base = (chunk_n - 1) * CHUNK_SAMPLES
    end = chunk_n * CHUNK_SAMPLES
    out = []
    for u in index:
        g0, g1 = u['g0'], u['g1']
        if g1 <= base or g0 >= end:
            continue
        if full_only and not (g0 >= base and g1 <= end):
            continue
        text = transcripts.get(u['utt_id'], '')
        if not text:
            continue
        out.append({
            'utt_id': u['utt_id'], 'text': text,
            'start_s': (g0 - base) / 16000.0,
            'end_s':   (g1 - base) / 16000.0,
            'dur_s':   u['frames'] / 16000.0,
        })
    return out


if __name__ == '__main__':
    idx = build_index()
    tr = load_transcripts()
    for n in (1, 2, 46):
        us = utterances_in_chunk(n, idx, tr)
        tot = sum(u['dur_s'] for u in us)
        print(f'chunk_{n:03d}: {len(us)} full utterances, {tot/60:.1f} min covered, '
              f'first="{us[0]["text"][:50]}…"' if us else f'chunk_{n:03d}: none')
