#!/usr/bin/env python3
"""
M15 analysis on TEST set, word 'before':
  - generate the utterance mel with M15 (flow), crop to the 'before' region
  - vocode (Griffin-Lim) that region -> Whisper -> predicted word
  - for WRONG predictions, save: original-audio mel | M15-generated mel | predicted-word real mel
"""
import os, json, numpy as np, torch, wave, torchaudio
from scipy.signal import resample_poly
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from config import CFG
import models as M
import data_io as io

ROOT = '<REPO_ROOT>'
OUT = 'outputs/before_analysis'; os.makedirs(OUT, exist_ok=True)
CTX = 0.12
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- M15 (mature last.pt) ----
ck = torch.load('outputs/run1/last.pt', map_location=dev, weights_only=False)
model = M.build(CFG).to(dev); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
print(f'[M15] step {ck.get("step")}')

# ---- data: test 'before' occurrences mapped to their utterances ----
occ = json.load(open(f'{ROOT}/M20_powerline_word_classifier/word_index.json'))
test = CFG.test_chunks()
man = json.load(open('full_manifest.json'))
by_chunk = {}
for r in man:
    by_chunk.setdefault(r['chunk'], []).append(r)


def find_utt(ch, ws, we):
    for r in by_chunk.get(ch, []):
        if r['start_s'] <= ws and r['end_s'] >= we and CFG.min_dur_s <= r['dur_s'] <= CFG.max_dur_s:
            return r
    return None


before_test = [(c, s, e) for (c, s, e) in occ['before'] if c in test and 0.28 <= e - s <= 0.6]
picks = []
for c, s, e in before_test:
    u = find_utt(c, s, e)
    if u:
        picks.append((c, s, e, u))
    if len(picks) >= 8:
        break
print(f'[data] {len(picks)} test "before" occurrences with utterances')

# ---- mel + GL + whisper ----
melfn = torchaudio.transforms.MelSpectrogram(CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop, win_length=CFG.win_length,
                                             n_mels=CFG.n_mels, f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)
_fb = torchaudio.functional.melscale_fbanks(CFG.n_fft // 2 + 1, CFG.fmin, CFG.fmax, CFG.n_mels, CFG.ref_sr,
                                            norm=None, mel_scale='htk')
_pinv = torch.linalg.pinv(_fb.T)
_gl = torchaudio.transforms.GriffinLim(CFG.n_fft, 64, win_length=CFG.win_length, hop_length=CFG.hop, power=2.0)
import whisper; wm = whisper.load_model('small', device=dev)


def gl(lm):
    w = _gl((_pinv @ (lm.exp() - CFG.log_eps).clamp(min=0)).clamp(min=0)); return (w / (w.abs().max() + 1e-8)).numpy()
def savewav(p, x):
    with wave.open(p, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(CFG.ref_sr); w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
def whisp(x):
    r = wm.transcribe(np.ascontiguousarray(x.astype(np.float32)), language='en', fp16=(dev == 'cuda'), verbose=False)
    ws = ''.join(ch for ch in r['text'].lower() if ch.isalpha() or ch == ' ').split()
    return ws[0] if ws else '(silence)'
def real_mel(ch, t0, dur):
    w = io.read_wav_window(CFG.wav_path(ch), t0, dur, CFG.ref_sr).astype(np.float32)
    return torch.log(melfn(torch.from_numpy(w)) + CFG.log_eps)
def imsave(path, mels, titles):
    n = len(mels); f, ax = plt.subplots(1, n, figsize=(2.4 * n, 1.7), dpi=125)
    if n == 1: ax = [ax]
    vmn = min(m.min() for m in mels); vmx = max(m.max() for m in mels)
    for a, m, t in zip(ax, mels, titles):
        a.imshow(m, origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
        a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    f.tight_layout(pad=0.3); f.savefig(path, bbox_inches='tight', transparent=True); plt.close(f)


results = []
for k, (ch, s, e, u) in enumerate(picks):
    lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0
    dur = min(u['dur_s'], CFG.max_dur_s)
    raw = io.read_bin_window(CFG.bin_path(ch), u['start_s'] + lag, dur, CFG.cap_sr).astype(np.float32)
    raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
    x = np.zeros(CFG.a_len, np.float32); x[:min(len(raw), CFG.a_len)] = raw[:CFG.a_len]
    with torch.no_grad():
        gen = model.sample(torch.from_numpy(x)[None].to(dev))[0].float().cpu() * ms + mm   # [80,1600]
    # before-word frames within utterance (100 fps)
    f0 = max(0, int((s - CTX - u['start_s']) * CFG.fps)); f1 = int((e + CTX - u['start_s']) * CFG.fps)
    gen_b = gen[:, f0:f1]
    real_b = real_mel(ch, s - CTX, (e - s) + 2 * CTX)
    L = min(gen_b.shape[1], real_b.shape[1]); gen_b, real_b = gen_b[:, :L], real_b[:, :L]
    savewav(f'{OUT}/ex{k}_orig.wav', io.read_wav_window(CFG.wav_path(ch), s - CTX, (e - s) + 2 * CTX, CFG.ref_sr))
    gw = gl(gen_b); savewav(f'{OUT}/ex{k}_powerGL.wav', gw)
    pred = whisp(gw)
    results.append({'k': k, 'chunk': ch, 'pred': pred, 'correct': (pred == 'before'),
                    'gen_b': gen_b.numpy(), 'real_b': real_b.numpy(), 's': s, 'e': e})
    print(f'[ex{k}] {ch}: pred="{pred}" {"OK" if pred=="before" else "WRONG"}', flush=True)

# wrong ones: add predicted-word real mel
wrongs = [r for r in results if not r['correct']][:5]
man_manifest = []
for r in wrongs:
    k, pw = r['k'], r['pred']
    pred_mel = None
    if pw in occ and occ[pw]:
        cand = [o for o in occ[pw] if 0.25 <= o[2] - o[1] <= 0.6]
        if cand:
            pc, ps, pe = cand[len(cand) // 2]
            pred_mel = real_mel(pc, ps - CTX, (pe - ps) + 2 * CTX).numpy()
    mels = [r['real_b'], r['gen_b']] + ([pred_mel] if pred_mel is not None else [])
    titles = ['original "before"', 'M15 generated', f'real "{pw}"'][:len(mels)]
    imsave(f'{OUT}/ex{k}_mels.png', mels, titles)
    man_manifest.append({'k': k, 'chunk': r['chunk'], 'pred': pw, 'has_pred_mel': pred_mel is not None})
    print(f'  [wrong ex{k}] before -> "{pw}"  pred_mel={pred_mel is not None}', flush=True)
json.dump({'all': [{'k': r['k'], 'chunk': r['chunk'], 'pred': r['pred'], 'correct': r['correct']} for r in results],
           'wrong': man_manifest}, open(f'{OUT}/manifest.json', 'w'), indent=1)
print(f'[done] {len(results)} generated, {len(wrongs)} wrong analyzed -> {OUT}')
