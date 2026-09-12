#!/usr/bin/env python3
"""
Final analysis: for wrongly-predicted 'before' (M15-generated mels), correlate the
generated mel against (a) the TRUE 'before' mel and (b) the PREDICTED word's mel,
with a random-word baseline. Report full / envelope / envelope-removed(phonetic) r.
"""
import os, json, numpy as np, torch, wave, torchaudio
from scipy.signal import resample_poly
import torch.nn.functional as F

from config import CFG
import models as M
import data_io as io

ROOT = '<REPO_ROOT>'
CTX, N, NF = 0.12, 24, 40      # NF = common time length
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

ck = torch.load('outputs/run1/last.pt', map_location=dev, weights_only=False)
model = M.build(CFG).to(dev); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
occ = json.load(open(f'{ROOT}/M20_powerline_word_classifier/word_index.json'))
test = CFG.test_chunks()
man = json.load(open('full_manifest.json'))
by_chunk = {}
[by_chunk.setdefault(r['chunk'], []).append(r) for r in man]
melfn = torchaudio.transforms.MelSpectrogram(CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop, win_length=CFG.win_length,
                                             n_mels=CFG.n_mels, f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)
_fb = torchaudio.functional.melscale_fbanks(CFG.n_fft // 2 + 1, CFG.fmin, CFG.fmax, CFG.n_mels, CFG.ref_sr, norm=None, mel_scale='htk')
_pinv = torch.linalg.pinv(_fb.T); _gl = torchaudio.transforms.GriffinLim(CFG.n_fft, 48, win_length=CFG.win_length, hop_length=CFG.hop, power=2.0)
import whisper; wm = whisper.load_model('small', device=dev)


def real_mel(ch, t0, dur):
    w = io.read_wav_window(CFG.wav_path(ch), t0, dur, CFG.ref_sr).astype(np.float32)
    return torch.log(melfn(torch.from_numpy(w)) + CFG.log_eps)
def norm_t(m):                                    # [80,L] -> [80,NF]
    return F.interpolate(torch.as_tensor(m)[None, None], size=(CFG.n_mels, NF), mode='bilinear', align_corners=False)[0, 0].numpy()
def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.std() * b.std(); return float((a * b).mean() / d) if d > 1e-9 else 0.0
def metrics(g, o):
    g, o = norm_t(g), norm_t(o)
    full = pear(g.ravel(), o.ravel())
    env = pear(g.mean(0), o.mean(0))
    phon = pear((g - g.mean(0, keepdims=True)).ravel(), (o - o.mean(0, keepdims=True)).ravel())
    return full, env, phon
def word_mel(w, rng):
    cand = [o for o in occ.get(w, []) if 0.22 <= o[2] - o[1] <= 0.7]
    if not cand: return None
    c, s, e = cand[rng.randint(len(cand))]; return real_mel(c, s - CTX, (e - s) + 2 * CTX).numpy()


before_test = [(c, s, e) for (c, s, e) in occ['before'] if c in test and 0.28 <= e - s <= 0.6]
rng = np.random.RandomState(0)
rows = []
for c, s, e in before_test:
    u = next((r for r in by_chunk.get(c, []) if r['start_s'] <= s and r['end_s'] >= e
              and CFG.min_dur_s <= r['dur_s'] <= CFG.max_dur_s), None)
    if not u: continue
    lag = io.read_lag_ms(CFG.lag_path(c)) / 1000.0; dur = min(u['dur_s'], CFG.max_dur_s)
    raw = io.read_bin_window(CFG.bin_path(c), u['start_s'] + lag, dur, CFG.cap_sr).astype(np.float32)
    raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
    x = np.zeros(CFG.a_len, np.float32); x[:min(len(raw), CFG.a_len)] = raw[:CFG.a_len]
    with torch.no_grad():
        gen = (model.sample(torch.from_numpy(x)[None].to(dev))[0].float().cpu() * ms + mm)
    f0 = max(0, int((s - CTX - u['start_s']) * CFG.fps)); f1 = int((e + CTX - u['start_s']) * CFG.fps)
    gen_b = gen[:, f0:f1].numpy()
    real_b = real_mel(c, s - CTX, (e - s) + 2 * CTX).numpy()
    L = min(gen_b.shape[1], real_b.shape[1]); gen_b, real_b = gen_b[:, :L], real_b[:, :L]
    gw = _gl((_pinv @ (torch.from_numpy(gen_b).exp() - CFG.log_eps).clamp(min=0)).clamp(min=0)); gw = (gw / (gw.abs().max() + 1e-8)).numpy()
    r = wm.transcribe(np.ascontiguousarray(gw), language='en', fp16=(dev == 'cuda'), verbose=False)
    pw = (''.join(ch for ch in r['text'].lower() if ch.isalpha() or ch == ' ').split() or ['(silence)'])[0]
    if pw == 'before': continue                    # only WRONG cases
    pm = word_mel(pw, rng)
    rm = word_mel(next(w for w in occ if w not in ('before', pw) and len(occ[w]) > 30), rng)  # random baseline
    e_true = metrics(gen_b, real_b)
    e_pred = metrics(gen_b, pm) if pm is not None else (None, None, None)
    e_rand = metrics(gen_b, rm) if rm is not None else (None, None, None)
    rows.append({'chunk': c, 'pred': pw, 'true': e_true, 'pred_m': e_pred, 'rand': e_rand})
    print(f'[{c}] before->{pw}: vs TRUE full={e_true[0]:.2f} env={e_true[1]:.2f} phon={e_true[2]:.2f}'
          + (f' | vs PRED full={e_pred[0]:.2f} env={e_pred[1]:.2f} phon={e_pred[2]:.2f}' if pm is not None else ' | vs PRED n/a'), flush=True)
    if len(rows) >= N: break


def agg(key, idx):
    v = [r[key][idx] for r in rows if r[key][idx] is not None]
    return (np.mean(v), np.std(v), len(v))


print(f'\n===== M15 generated "before" mel correlations (n={len(rows)} wrong cases) =====')
print(f'{"":20}{"full r":>16}{"env r":>16}{"phon r (env-removed)":>24}')
for label, key in [('vs TRUE "before"', 'true'), ('vs PREDICTED word', 'pred_m'), ('vs RANDOM word', 'rand')]:
    f, e, p = agg(key, 0), agg(key, 1), agg(key, 2)
    print(f'{label:20}{f"{f[0]:+.3f}±{f[1]:.2f}":>16}{f"{e[0]:+.3f}±{e[1]:.2f}":>16}{f"{p[0]:+.3f}±{p[1]:.2f}":>24}  (n={f[2]})')
json.dump({'n': len(rows), 'rows': [{k: (v if k in ('chunk', 'pred') else list(v)) for k, v in r.items()} for r in rows]},
          open('outputs/before_analysis/corr.json', 'w'), indent=1)
print('\n[read] env r ~ how much the generated mel matches loudness; phon r ~ spectral/phonetic match.')
