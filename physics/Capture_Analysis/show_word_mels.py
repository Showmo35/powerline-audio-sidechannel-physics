#!/usr/bin/env python3
"""M14 generated mels for several occurrences of the same word vs different words.
Rows = words, columns = occurrences. Fixed sampler seed so differences reflect the
powerline input, not random sampling."""
import sys, json, importlib, collections
import numpy as np, torch, wave
import torchaudio
from scipy.signal import resample_poly
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = '<REPO_ROOT>'
BIN = ROOT + '/Powerline_Data_Captures/soundbar_bin_captures'
WAVD = ROOT + '/Powerline_Data_Captures/audio_chunks'
CAP_SR = 200_000
TARGET_WORDS = ['missus', 'said', 'little', 'time']
NOCC = 5
NCHUNKS = 6
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

# ---- M14 ----
sys.path.insert(0, ROOT + '/M14_new_setup_soundbar_melgen')
cfg = importlib.import_module('config').CFG
m14 = importlib.import_module('models').build(cfg).to(dev)
ck = torch.load(ROOT + '/M14_new_setup_soundbar_melgen/outputs/cluster_a100/last.pt',
                map_location=dev, weights_only=False)
m14.load_state_dict(ck.get('ema', ck['model'])); m14.eval()
mean, std = ck['mel_mean'], ck['mel_std']
plc = {}

@torch.no_grad()
def gen_mel(ch, s, e):
    if ch not in plc:
        plc[ch] = np.fromfile(f'{BIN}/{ch}.bin', dtype=np.float32)
    a = int((s + lag_s(ch)) * CAP_SR); raw = plc[ch][a:a + int(cfg.win_s * CAP_SR)]
    raw = np.pad(raw, (0, max(0, int(cfg.win_s * CAP_SR) - len(raw))))
    xr = resample_poly(raw, cfg.in_sr, CAP_SR).astype(np.float32)
    torch.manual_seed(1234)                                  # same noise for every generation
    g = m14.sample(torch.from_numpy(xr)[None].to(dev))[0].float().cpu() * std + mean
    fr = max(6, int(round((e - s) * cfg.fps)))
    g = g[:, :fr]
    return torch.nn.functional.interpolate(g[None, None], size=(cfg.n_mels, 40),
                                            mode='bilinear', align_corners=False)[0, 0].numpy()

mels = {w: [gen_mel(*o) for o in occ[w]] for w in TARGET_WORDS}
allv = np.concatenate([m.ravel() for w in TARGET_WORDS for m in mels[w]])
vmn, vmx = np.percentile(allv, 2), np.percentile(allv, 98)

fig, ax = plt.subplots(len(TARGET_WORDS), NOCC, figsize=(2.2 * NOCC, 2.2 * len(TARGET_WORDS)), dpi=130)
for ri, w in enumerate(TARGET_WORDS):
    for ci in range(NOCC):
        a = ax[ri, ci]
        if ci < len(mels[w]):
            a.imshow(mels[w][ci], origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
        a.set_xticks([]); a.set_yticks([])
        if ci == 0:
            a.set_ylabel(f'"{w}"', fontsize=13)
        if ri == 0:
            a.set_title(f'occ {ci+1}', fontsize=10)
fig.suptitle('M14 mel from powerline — rows = same word, across rows = different words\n'
             '(fixed sampler seed: differences reflect the powerline input only)', fontsize=12)
fig.tight_layout()
out = ROOT + '/Capture_Analysis/word_mels.png'
fig.savefig(out, bbox_inches='tight'); print('[saved]', out, {w: len(occ[w]) for w in TARGET_WORDS})
