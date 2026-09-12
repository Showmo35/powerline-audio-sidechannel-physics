#!/usr/bin/env python3
"""Generate M14's predicted mel for each of the 5 powerline hellos; plot vs real."""
import sys, numpy as np, torch, wave, importlib
from scipy.signal import resample_poly, correlate
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

CAP = '<REPO_ROOT>/physics/Capture_Analysis/capture.bin'
WAV = '<REPO_ROOT>/physics/Capture_Analysis/hello5.wav'
CAP_SR = 200_000
ONSETS = [0.06, 1.48, 2.90, 4.32, 5.75]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

pl = np.fromfile(CAP, dtype=np.float32)
with wave.open(WAV, 'rb') as w:
    asr = w.getframerate(); au = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
au = au / (np.abs(au).max() + 1e-9)

def env_1k(x, sr):
    fr = int(0.005 * sr); e = np.sqrt(np.convolve(x ** 2, np.ones(fr) / fr, 'same'))
    return resample_poly(e, 1000, sr)
pe, ae = env_1k(pl, CAP_SR), env_1k(au, asr)
n = min(len(pe), len(ae)); c = correlate(pe[:n] - pe[:n].mean(), ae[:n] - ae[:n].mean(), method='fft')
lag_s = (c.argmax() - (n - 1)) / 1000.0

sys.path.insert(0, '<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen')
cfg = importlib.import_module('config').CFG
import dataset as Dm
m14 = importlib.import_module('models').build(cfg).to(dev)
ck = torch.load('<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen/'
                'outputs/cluster_a100/last.pt', map_location=dev, weights_only=False)
m14.load_state_dict(ck.get('ema', ck['model'])); m14.eval()
mean, std = ck['mel_mean'], ck['mel_std']

melfn = __import__('torchaudio').transforms.MelSpectrogram(
    sample_rate=cfg.ref_sr, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win_length,
    n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax, power=2.0)

FR = 80          # frames to show (0.8 s @ 100 fps)
gen_mels, real_mels = [], []
with torch.no_grad():
    for o in ONSETS:
        a = int(round((o + lag_s) * CAP_SR)); raw = pl[a:a + int(cfg.win_s * CAP_SR)]
        raw = np.pad(raw, (0, max(0, int(cfg.win_s * CAP_SR) - len(raw))))
        xr = resample_poly(raw, cfg.in_sr, CAP_SR).astype(np.float32)
        g = m14.sample(torch.from_numpy(xr)[None].to(dev))[0].float().cpu() * std + mean
        gen_mels.append(g[:, :FR].numpy())
        aw = au[int(o * asr):int((o + cfg.win_s) * asr)]
        aw = resample_poly(aw, cfg.ref_sr, asr).astype(np.float32)
        aw = np.pad(aw, (0, max(0, int(cfg.win_s * cfg.ref_sr) - len(aw))))
        rm = torch.log(melfn(torch.from_numpy(aw)) + cfg.log_eps).numpy()
        real_mels.append(rm[:, :FR])

fig, ax = plt.subplots(2, 5, figsize=(15, 5), dpi=130)
vmn = min(m.min() for m in real_mels); vmx = max(m.max() for m in real_mels)
for i in range(5):
    ax[0, i].imshow(real_mels[i], origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    ax[1, i].imshow(gen_mels[i], origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    ax[0, i].set_title(f'hello {i+1}', fontsize=10); ax[0, i].set_xticks([]); ax[1, i].set_xticks([])
    ax[0, i].set_yticks([]); ax[1, i].set_yticks([])
ax[0, 0].set_ylabel('REAL audio mel', fontsize=10); ax[1, 0].set_ylabel('M14 predicted\n(from powerline)', fontsize=10)
fig.suptitle('5 "hello" repetitions — real mel (top) vs M14 mel from powerline (bottom)', fontsize=12)
fig.tight_layout()
out = '<REPO_ROOT>/physics/Capture_Analysis/hello_mels.png'
fig.savefig(out, bbox_inches='tight'); print('[saved]', out)
