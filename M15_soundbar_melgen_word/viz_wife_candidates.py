#!/usr/bin/env python3
"""viz_wife_candidates.py — visualize why M15's generated 'wife' mel matches its
top-10 centroid candidates. Shows: M15-GEN 'wife' | REAL 'wife' avg | each candidate's
average real mel (time-normalized 80x64), with the centroid cosine score. Saves
outputs/wife_candidates.png."""
import os, json, numpy as np, torch
import torch.nn.functional as F
import torchaudio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from config import CFG
import models as M
import data_io as io

PROOT = '<REPO_ROOT>'
WIDX = os.path.join(PROOT, 'M20_powerline_word_classifier', 'word_index.json')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CTX = 0.10; T = 64
TRUE = 'wife'
CANDS = ['beg', 'platte', 'back', 'dock', 'dive', 'drug', 'bought', 'bet', 'pack', 'debt']

ck = torch.load('outputs/run1/last.pt', map_location=dev, weights_only=False)
model = M.build(CFG).to(dev); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
melfn = torchaudio.transforms.MelSpectrogram(CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop,
                                             win_length=CFG.win_length, n_mels=CFG.n_mels,
                                             f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)


def norm64(m):                                            # (80,f) -> (80,64)
    return F.interpolate(torch.as_tensor(m)[None, None].float(), size=(80, T),
                         mode='bilinear', align_corners=False)[0, 0]

def embed(m64):                                           # (80,64) -> centered/L2
    z = m64.reshape(-1); z = z - z.mean(); return z / (z.norm() + 1e-8)

def real_word_mel64(ch, s, e):
    w = io.read_wav_window(CFG.wav_path(ch), s - CTX, (e - s) + 2 * CTX, CFG.ref_sr).astype(np.float32)
    return norm64(torch.log(melfn(torch.from_numpy(w)) + CFG.log_eps))

occ = json.load(open(WIDX))
def is_test(ch): return int(ch.split('_')[1]) % 12 == 0

def word_avg_and_centroid(w, cap=40):
    rows = [(c, s, e) for (c, s, e) in occ.get(w, []) if not is_test(c)][:cap]
    mels = [real_word_mel64(c, s, e) for (c, s, e) in rows]
    if not mels:
        return None, None
    stack = torch.stack(mels)                             # (n,80,64)
    avg = stack.mean(0)                                   # display prototype
    cen = torch.stack([embed(m) for m in mels]).mean(0)   # embedding centroid
    return avg.numpy(), (cen / (cen.norm() + 1e-8))

# ── M15-generated 'wife' from the sentence (chunk_084, seed-1 picks[2]) ──
man = json.load(open('full_manifest.json'))
test = [r for r in man if is_test(r['chunk']) and 2.0 <= r['dur_s'] <= 3.8 and len(r['text'].split()) >= 4]
rng = np.random.RandomState(1); rng.shuffle(test); utt = test[2]
ch, us = utt['chunk'], utt['start_s']
wife_occ = [(s, e) for (c, s, e) in occ['wife'] if c == ch and s >= us - 0.05 and e <= utt['end_s'] + 0.05]
ws, we = wife_occ[0]
lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0
dur = min(utt['dur_s'], CFG.max_dur_s)
raw = io.read_bin_window(CFG.bin_path(ch), us + lag, dur, CFG.cap_sr).astype(np.float32)
raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
x = np.zeros(CFG.a_len, np.float32); x[:min(len(raw), CFG.a_len)] = raw[:CFG.a_len]
with torch.no_grad():
    gen = (model.sample(torch.from_numpy(x)[None].to(dev))[0].float().cpu() * ms + mm)
f0 = max(0, int((ws - CTX - us) * CFG.fps)); f1 = min(gen.shape[1], int((we + CTX - us) * CFG.fps))
gen_wife64 = norm64(gen[:, f0:f1])
gen_emb = embed(gen_wife64)

# ── panels ──
panels = [('M15-GEN "wife"', gen_wife64.numpy(), None)]
avg_w, cen_w = word_avg_and_centroid(TRUE)
panels.append(('REAL "wife" avg', avg_w, float((gen_emb @ cen_w).item())))
for w in CANDS:
    avg, cen = word_avg_and_centroid(w)
    if avg is None:
        continue
    panels.append((w, avg, float((gen_emb @ cen).item())))

vmn = min(p[1].min() for p in panels); vmx = max(p[1].max() for p in panels)
ncol = 6; nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.0 * nrow))
axes = np.array(axes).reshape(-1)
for ax, (title, mel, sc) in zip(axes, panels):
    ax.imshow(mel, origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    ax.set_xticks([]); ax.set_yticks([])
    col = '#1b7837' if title.startswith('M15') else ('#2563d6' if 'REAL' in title else 'black')
    t = title if sc is None else f'{title}   r={sc:.2f}'
    ax.set_title(t, fontsize=9, color=col)
for ax in axes[len(panels):]:
    ax.axis('off')
fig.suptitle('M15-generated "wife" vs its top-10 centroid candidates (avg real mel, time-normalized)\n'
             'all short monosyllables — same envelope/duration, so all score ~0.70; true "wife" ranked 207',
             fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('outputs/wife_candidates.png', dpi=145, bbox_inches='tight')
print('wrote outputs/wife_candidates.png', flush=True)
print('scores:', [(t, round(s, 3)) for (t, _, s) in panels if s is not None], flush=True)
