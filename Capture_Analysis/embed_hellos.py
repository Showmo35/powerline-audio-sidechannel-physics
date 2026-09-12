#!/usr/bin/env python3
"""Encode the 5 powerline 'hello's with M14 & M15 encoders; do they share a code?
Baseline = non-hello 'gap' segments, so we can tell if hello-hello cosine is
actually high vs just the general similarity level in that encoder's space."""
import sys, numpy as np, torch, wave, importlib
from scipy.signal import resample_poly, correlate

CAP = '<REPO_ROOT>/Capture_Analysis/capture.bin'
WAV = '<REPO_ROOT>/Capture_Analysis/hello5.wav'
CAP_SR = 200_000
HELLO_ONSETS = [0.06, 1.48, 2.90, 4.32, 5.75]      # from hello5.wav
GAP_ONSETS   = [0.85, 2.25, 3.68, 5.10]            # midpoints between hellos (non-speech)
SEG_S = 0.6
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

pl = np.fromfile(CAP, dtype=np.float32)

def env_1k(x, sr):
    fr = int(0.005 * sr); e = np.sqrt(np.convolve(x ** 2, np.ones(fr) / fr, 'same'))
    return resample_poly(e, 1000, sr)

with wave.open(WAV, 'rb') as w:
    asr = w.getframerate(); au = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
pe, ae = env_1k(pl, CAP_SR), env_1k(au, asr)
n = min(len(pe), len(ae)); c = correlate(pe[:n] - pe[:n].mean(), ae[:n] - ae[:n].mean(), method='fft')
lag_s = (c.argmax() - (n - 1)) / 1000.0
print(f'[align] powerline lags audio by {lag_s*1000:.0f} ms')

def pl_seg(onset):
    a = int(round((onset + lag_s) * CAP_SR)); L = int(round(SEG_S * CAP_SR)); x = pl[max(0, a):a + L]
    if len(x) < L: x = np.pad(x, (0, L - len(x)))
    return x / (x.std() + 1e-8)

hellos = [pl_seg(o) for o in HELLO_ONSETS]
gaps   = [pl_seg(o) for o in GAP_ONSETS]

def cos(A, B):
    A = A / (A.norm(dim=1, keepdim=True) + 1e-9); B = B / (B.norm(dim=1, keepdim=True) + 1e-9)
    return A @ B.T

def report(name, encode):
    H = torch.stack([encode(x) for x in hellos]); G = torch.stack([encode(x) for x in gaps])
    Shh = cos(H, H); Shg = cos(H, G); Sgg = cos(G, G)
    iu = torch.triu_indices(5, 5, 1); jg = torch.triu_indices(4, 4, 1)
    hh = Shh[iu[0], iu[1]].mean(); hg = Shg.mean(); gg = Sgg[jg[0], jg[1]].mean()
    print(f'\n[{name}]  hello-hello={hh:.3f}   hello-gap={hg:.3f}   gap-gap={gg:.3f}')
    print('  5x5 hello cosine:'); print('  ' + np.array2string(np.round(Shh.cpu().numpy(), 3)).replace('\n', '\n  '))

def load(path_dir, ckpt):
    for m in ('config', 'models'):
        sys.modules.pop(m, None)
    sys.path.insert(0, path_dir)
    cfg = importlib.import_module('config').CFG
    model = importlib.import_module('models').build(cfg).to(dev)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
    sys.path.remove(path_dir)
    return cfg, model

# ---- M14 (input at in_sr=32k) ----
cfg14, m14 = load('<REPO_ROOT>/M14_new_setup_soundbar_melgen',
                  '<REPO_ROOT>/M14_new_setup_soundbar_melgen/outputs/cluster_a100/last.pt')
@torch.no_grad()
def enc14(x):
    xr = resample_poly(x, cfg14.in_sr, CAP_SR).astype(np.float32)
    return m14.encode(torch.from_numpy(xr)[None].to(dev))[0].mean(0).float().cpu()
report('M14 encoder', enc14)

# ---- M15 (input at full cap_sr=200k) ----
cfg15, m15 = load('<REPO_ROOT>/M15_soundbar_melgen_word',
                  '<REPO_ROOT>/M15_soundbar_melgen_word/outputs/run1/best.pt')
@torch.no_grad()
def enc15(x):
    return m15.encode(torch.from_numpy(x.astype(np.float32))[None].to(dev))[0].mean(0).float().cpu()
report('M15 encoder', enc15)

print('\n[read] hello-hello >> hello-gap  => the 5 hellos share a code distinct from non-hello.'
      '\n       hello-hello ~= hello-gap    => high cosine is just the space, not word-specific.')
