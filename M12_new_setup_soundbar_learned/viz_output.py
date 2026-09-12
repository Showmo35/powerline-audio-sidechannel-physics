#!/usr/bin/env python3
"""viz_output.py — visualize/listen to the learned front-end output vs clean & fixed."""
import json, os, random, wave
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchaudio.functional as AF
import torchaudio.transforms as TT
from config import CFG, OUT_DIR, FEAT_DIR, SHARED_MANIFEST
from model_unet import MelUNet
import data_io as io

random.seed(3)
N = 4
rows = [json.loads(l) for l in open(SHARED_MANIFEST) if l.strip()]
te = [r for r in rows if r['chunk'] in {f'chunk_{n:03d}' for n in range(41, 47)}]
sample = random.sample(te, N)

net = MelUNet(in_ch=2*CFG.n_harmonics, base=CFG.base_ch).eval()
net.load_state_dict(torch.load(os.path.join(OUT_DIR, 'enh_unet', 'best_eval.pt'),
                               map_location='cpu'))

fb = AF.melscale_fbanks(CFG.mel_n_fft//2+1, CFG.mel_fmin, CFG.mel_fmax,
                        CFG.mel_n_mels, CFG.aud_sr, norm=None, mel_scale='htk')
fb_pinv = torch.linalg.pinv(fb)
gl = TT.GriffinLim(CFG.mel_n_fft, win_length=CFG.mel_win, hop_length=CFG.mel_hop,
                   power=1.0, n_iter=64)

def logmel_to_wav16(logmel):
    spec = (fb_pinv.T @ torch.from_numpy(np.exp(logmel)).float()).clamp(min=0)
    return io.resample(gl(spec).numpy(), CFG.aud_sr, CFG.asr_sr)

def save_wav(path, w):
    w = w/(np.abs(w).max()+1e-9)*0.9
    with wave.open(path,'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(CFG.asr_sr)
        f.writeframes((w*32767).astype(np.int16).tobytes())

fig, axes = plt.subplots(3, N, figsize=(4*N, 8))
xc, yc = {}, {}
for j, r in enumerate(sample):
    ch, uid = r['chunk'], r['utt_id']
    if ch not in xc:
        xc[ch] = np.load(os.path.join(FEAT_DIR, f'{ch}.x.npz'))
        yc[ch] = np.load(os.path.join(FEAT_DIR, f'{ch}.y.npz'))
    x = xc[ch][uid].astype(np.float32); y = yc[ch][uid].astype(np.float32)
    xz = (x - x.mean())/(x.std()+1e-6)
    with torch.no_grad():
        pred = net(torch.from_numpy(xz)[None]).numpy()[0]
    fixed = np.log(np.maximum(np.exp(x).mean(0), 1e-5))
    vmin, vmax = np.percentile(y, 5), np.percentile(y, 99)
    for i, (name, M) in enumerate((('CLEAN target', y),
                                   ('FIXED 8-harm sum', fixed),
                                   ('LEARNED U-Net', pred))):
        ax = axes[i, j]
        ax.imshow(M, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0: ax.set_ylabel(name, fontsize=10)
        if i == 0: ax.set_title(' '.join(r['text'].split()[:6])+'…', fontsize=8)
    print(f'[{j}] {uid}: "{r["text"][:70]}"')
    if j == 0:                       # dump audio for the first example
        save_wav(os.path.join(OUT_DIR, 'example_clean.wav'), logmel_to_wav16(y))
        save_wav(os.path.join(OUT_DIR, 'example_learned.wav'), logmel_to_wav16(pred))
        save_wav(os.path.join(OUT_DIR, 'example_fixed.wav'), logmel_to_wav16(fixed))

fig.suptitle('Powerline learned front-end: clean vs fixed vs learned mel '
             '(held-out chunks 41-46)', fontsize=12)
fig.tight_layout()
out = os.path.join(OUT_DIR, 'unet_output.png')
fig.savefig(out, dpi=130, bbox_inches='tight')
print('saved', out)
print('audio: outputs/example_{clean,fixed,learned}.wav')
