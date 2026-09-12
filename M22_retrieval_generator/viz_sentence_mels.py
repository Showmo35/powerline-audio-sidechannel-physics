#!/usr/bin/env python3
"""viz_sentence_mels.py — M22 generated full-sentence mel vs the REAL mel, side by side,
for 5 held-out (test) sentences. Same window fed to M22 (powerline) and to the real
16 kHz reference. Saves outputs/sentence_mels.png."""
import os, json, numpy as np, torch
import torchaudio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from config import CFG
import models as M
import data_io as io

ROOT = '<REPO_ROOT>'
MAN = os.path.join(ROOT, 'M15_soundbar_melgen_word', 'full_manifest.json')
OUT = 'outputs'; os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

ck = torch.load('outputs/best.pt', map_location=dev, weights_only=False)
model = M.build(CFG).to(dev); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
melfn = torchaudio.transforms.MelSpectrogram(CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop,
                                             win_length=CFG.win_length, n_mels=CFG.n_mels,
                                             f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)

man = json.load(open(MAN))
test = [r for r in man if CFG.is_test(r['chunk']) and 2.0 <= r['dur_s'] <= 3.8 and len(r['text'].split()) >= 4]
rng = np.random.RandomState(1); rng.shuffle(test)
picks = test[:5]

fig, axes = plt.subplots(5, 2, figsize=(15, 12))
for k, r in enumerate(picks):
    ch, s, dur = r['chunk'], r['start_s'], min(r['dur_s'], CFG.win_s)
    lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0
    raw = io.read_bin_window(CFG.bin_path(ch), s + lag, CFG.win_s, CFG.cap_sr, CFG.in_sr).astype(np.float32)
    raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
    x = np.zeros(CFG.in_len, np.float32); x[:min(len(raw), CFG.in_len)] = raw[:CFG.in_len]
    with torch.no_grad():
        gen = (model.sample(torch.from_numpy(x)[None].to(dev))[0].float().cpu() * ms + mm).numpy()  # (80,400)
    real16 = io.read_wav_window(CFG.wav_path(ch), s, dur, CFG.ref_sr).astype(np.float32)
    real = torch.log(melfn(torch.from_numpy(real16)) + CFG.log_eps).numpy()                          # (80,~dur*100)
    vfr = min(gen.shape[1], real.shape[1], int(round(dur * CFG.fps)))
    gen, real = gen[:, :vfr], real[:, :vfr]
    vmn = min(gen.min(), real.min()); vmx = max(gen.max(), real.max())
    axes[k, 0].imshow(real, origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    axes[k, 1].imshow(gen, origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    axes[k, 0].set_ylabel(f'[{ch}]\n80 mel', fontsize=8)
    txt = ' '.join(r['text'].split()[:12])
    axes[k, 0].set_title(f'ORIGINAL (real) — "{txt}"', fontsize=9)
    axes[k, 1].set_title('M22 GENERATED (from powerline)', fontsize=9, color='#b2182b')
    for a in axes[k]:
        a.set_xticks([]); a.set_yticks([])
fig.suptitle('M22 full-sentence generated mel vs original real mel — 5 held-out test sentences',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(f'{OUT}/sentence_mels.png', dpi=140, bbox_inches='tight')
print('wrote', os.path.abspath(f'{OUT}/sentence_mels.png'))
