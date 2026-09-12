#!/usr/bin/env python3
"""Envelope (loudness contour) per word: audio vs powerline, 5 reps overlaid.
Rows = words, cols = [audio envelope, powerline envelope]. Each curve time-
normalized to 50 points and peak-normalized so shape is comparable."""
import sys, json, collections
import numpy as np, torch, wave
import torchaudio
from scipy.signal import resample_poly
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = '<REPO_ROOT>'
BIN = ROOT + '/Powerline_Data_Captures/soundbar_bin_captures'
WAVD = ROOT + '/Powerline_Data_Captures/audio_chunks'
CAP_SR = 200_000
TARGET_WORDS = ['missus', 'said', 'little', 'time']
NOCC, NCHUNKS, NP = 5, 6, 50
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_wav(path, sr_out=16000):
    with wave.open(path, 'rb') as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    return (resample_poly(x, sr_out, sr).astype(np.float32) if sr != sr_out else x), sr_out


def lag_s(chunk):
    import os
    p = f'{BIN}/{chunk}.lag'
    return (float(open(p).read().strip()) / 1000.0) if os.path.exists(p) else 0.0


def envelope(x, sr, npts=NP):
    fr = max(1, int(0.004 * sr))
    e = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(fr) / fr, 'same'))
    e = resample_poly(e, npts, max(npts, len(e)))[:npts] if len(e) > npts else np.interp(
        np.linspace(0, 1, npts), np.linspace(0, 1, len(e)), e)
    e = np.interp(np.linspace(0, 1, npts), np.linspace(0, 1, len(e)), e)
    return e / (e.max() + 1e-9)


man = json.load(open(ROOT + '/M15_soundbar_melgen_word/full_manifest.json'))
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
w2v = bundle.get_model().to(dev).eval(); labels = bundle.get_labels()
lab2id = {c: i for i, c in enumerate(labels)}


def align(wave16, text):
    tnorm = ''.join(c for c in text.upper() if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ' ")
    toks = [lab2id[c] for c in tnorm.replace(' ', '|') if c in lab2id]
    if len(toks) < 2:
        return []
    with torch.inference_mode():
        emission, _ = w2v(torch.from_numpy(wave16)[None].to(dev)); logp = torch.log_softmax(emission, -1)
    if len(toks) >= emission.shape[1]:
        return []
    try:
        aligned, scores = torchaudio.functional.forced_align(logp, torch.tensor([toks], device=dev), blank=0)
    except RuntimeError:
        return []
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
    ratio = wave16.shape[0] / emission.shape[1] / 16000.0
    words, cur, cs, prev_end = [], [], None, 0
    for sp in spans:
        ch = labels[sp.token]
        if ch == '|':
            if cur:
                words.append((''.join(cur).lower(), cs * ratio, prev_end * ratio)); cur = []
            continue
        if not cur:
            cs = sp.start
        cur.append(ch); prev_end = sp.end
    if cur:
        words.append((''.join(cur).lower(), cs * ratio, prev_end * ratio))
    return words


occ = collections.defaultdict(list)
for ch in sorted({r['chunk'] for r in man})[:NCHUNKS]:
    wav, _ = read_wav(f'{WAVD}/{ch}.wav')
    for r in [r for r in man if r['chunk'] == ch]:
        seg = wav[int(r['start_s'] * 16000):int(r['end_s'] * 16000)]
        if len(seg) < 3200:
            continue
        for w, ws, we in align(seg, r['text']):
            if w in TARGET_WORDS and 0.18 <= we - ws <= 0.9 and len(occ[w]) < NOCC:
                occ[w].append((ch, r['start_s'] + ws, r['start_s'] + we))
    if all(len(occ[w]) >= NOCC for w in TARGET_WORDS):
        break

plc = {}
def env_pair(ch, s, e):
    wav, _ = read_wav(f'{WAVD}/{ch}.wav')
    ae = envelope(wav[int(s * 16000):int(e * 16000)], 16000)
    if ch not in plc:
        plc[ch] = np.fromfile(f'{BIN}/{ch}.bin', dtype=np.float32)
    lg = lag_s(ch); pe = envelope(plc[ch][int((s + lg) * CAP_SR):int((e + lg) * CAP_SR)], CAP_SR)
    return ae, pe

fig, ax = plt.subplots(len(TARGET_WORDS), 2, figsize=(9, 2.3 * len(TARGET_WORDS)), dpi=130, sharex=True)
t = np.linspace(0, 1, NP)
for ri, w in enumerate(TARGET_WORDS):
    aes, pes = [], []
    for o in occ[w]:
        ae, pe = env_pair(*o); aes.append(ae); pes.append(pe)
    for col, (curves, name) in enumerate([(aes, 'audio'), (pes, 'powerline')]):
        a = ax[ri, col]
        for c in curves:
            a.plot(t, c, color='#3b6ea5', alpha=0.5, lw=1)
        a.plot(t, np.mean(curves, 0), color='#c0392b', lw=2.2)
        a.set_ylim(-0.05, 1.08); a.set_yticks([])
        if ri == 0:
            a.set_title(f'{name} envelope', fontsize=12)
        if col == 0:
            a.set_ylabel(f'"{w}"', fontsize=13)
fig.suptitle('Word envelopes — 5 reps overlaid (blue), mean (red). Rows=words.\n'
             'tight overlay = consistent;  audio vs powerline = does the wire preserve it', fontsize=11)
fig.tight_layout()
out = ROOT + '/Capture_Analysis/word_envelopes.png'
fig.savefig(out, bbox_inches='tight'); print('[saved]', out, {w: len(occ[w]) for w in TARGET_WORDS})
