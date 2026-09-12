#!/usr/bin/env python3
"""WER eval of M14 step-73k generated audio, following M10 asr.py methodology:
Whisper transcript of REAL audio = pseudo ground-truth reference."""
import os, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, "<REPO_ROOT>/soundbar_setup/M10_new_setup_soundbar")
from asr import load_whisper, normalize_text, wer, corpus_wer

BASE = "<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen/outputs/cluster_a100"
REAL_DIR = os.path.join(BASE, "audio_grid")       # has *_REAL.wav (real audio)
GEN_DIR  = os.path.join(BASE, "audio_grid_73k")   # step-73k generated + target (Griffin-Lim)

def load_wav(path):
    x, sr = sf.read(path, dtype='float32')
    assert sr == 16000, (path, sr)
    return np.ascontiguousarray(x)

tags = sorted(f[:-len('_generated.wav')] for f in os.listdir(GEN_DIR) if f.endswith('_generated.wav'))

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tr(model, path):
    r = model.transcribe(load_wav(path), language='en', fp16=(device == 'cuda'), verbose=False)
    return r['text'].strip()

for name in ('small', 'large-v3'):
    model = load_whisper(name, device=device)
    refs, gl_hyps, gen_hyps = [], [], []
    for tag in tags:
        ref = tr(model, os.path.join(REAL_DIR, f'{tag}_REAL.wav'))
        gl  = tr(model, os.path.join(GEN_DIR,  f'{tag}_target.wav'))
        gen = tr(model, os.path.join(GEN_DIR,  f'{tag}_generated.wav'))
        refs.append(ref); gl_hyps.append(gl); gen_hyps.append(gen)
        print(f'\n== {tag} [{name}] ==')
        print(f'  REAL     : {ref}')
        print(f'  GL-target: {gl}   [wer vs real: {wer(ref, gl)["wer"]:.2f}]')
        print(f'  generated: {gen}   [wer vs real: {wer(ref, gen)["wer"]:.2f}]')
    print(f'\n──── corpus WER, whisper-{name} (vs transcript of REAL audio) ────')
    print(f'  Griffin-Lim target (vocoder ceiling): {corpus_wer(refs, gl_hyps)}')
    print(f'  PLF generated (step 73k)            : {corpus_wer(refs, gen_hyps)}')
    print(f'  generated vs GL-target transcript   : {corpus_wer(gl_hyps, gen_hyps)}')
    del model
    torch.cuda.empty_cache()
