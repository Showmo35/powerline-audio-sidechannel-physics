#!/usr/bin/env python3
"""10 'before' examples (6 correct, 4 wrong classifier detections). Per example:
powerline-vs-audio envelope plot, real+generated mel plot, and wavs
(original | real-mel->GL | powerline-mel->GL)."""
import sys, os, json, importlib
import numpy as np, torch, wave, torchaudio
from scipy.signal import resample_poly
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from config import CFG, OUT_DIR, BIN
import dataset as D, models as M

WORD, NC, NW, CTX = 'before', 6, 4, 0.10
OUT = os.path.join(OUT_DIR, 'before'); os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- classifier predictions (env-only, best) ----
ck = torch.load(os.path.join(OUT_DIR, 'env', 'best.pt'), map_location=dev, weights_only=False)
words, mean, std = ck['words'], ck['mean'], ck['std']; K = len(words)
clf = M.build(K).to(dev)
with torch.no_grad():
    _ = clf(torch.zeros(2, CFG.n_wbins, CFG.n_frames, device=dev))
clf.load_state_dict(ck['model']); clf.eval()
te_items, _ = D.build_items(split='test')
te_ds = D.CachedWords('test', flatten_freq=True)
dl = torch.utils.data.DataLoader(te_ds, batch_size=64, collate_fn=D.collate, num_workers=4)
preds = []
with torch.no_grad():
    for b in dl:
        preds.append(clf(((b['wide'] - mean) / std).to(dev)).argmax(1).cpu().numpy())
preds = np.concatenate(preds); ytrue = te_ds.y
bid = words.index(WORD)
rng = np.random.RandomState(1)
corr = [i for i in range(len(ytrue)) if ytrue[i] == bid and preds[i] == bid]
wrong = [i for i in range(len(ytrue)) if ytrue[i] == bid and preds[i] != bid]
rng.shuffle(corr); rng.shuffle(wrong)
picks = [(i, True) for i in corr[:NC]] + [(i, False) for i in wrong[:NW]]
print(f'[before] correct avail={len(corr)} wrong avail={len(wrong)}; using {NC}+{NW}')

# ---- M14 generated mel + GL vocoder ----
for _m in ('config', 'models'):
    sys.modules.pop(_m, None)
sys.path.insert(0, '<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen')
c14 = importlib.import_module('config').CFG
m14 = importlib.import_module('models').build(c14).to(dev)
_c = torch.load('<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen/'
                'outputs/cluster_a100/last.pt', map_location=dev, weights_only=False)
m14.load_state_dict(_c.get('ema', _c['model'])); m14.eval(); mm, msd = _c['mel_mean'], _c['mel_std']
melfn = torchaudio.transforms.MelSpectrogram(16000, 1024, hop_length=160, win_length=640,
                                             n_mels=80, f_min=0, f_max=8000, power=2.0)
_fb = torchaudio.functional.melscale_fbanks(513, 0, 8000, 80, 16000, norm=None, mel_scale='htk')
_pinv = torch.linalg.pinv(_fb.T)
_gl = torchaudio.transforms.GriffinLim(1024, n_iter=64, win_length=640, hop_length=160, power=2.0)


def gl(logmel):
    spec = (_pinv @ (logmel.exp() - 1e-5).clamp(min=0)).clamp(min=0)
    w = _gl(spec); return (w / (w.abs().max() + 1e-8)).numpy()


def savewav(p, x, sr=16000):
    with wave.open(p, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def env(x, sr, npts=48, win_ms=30):
    fr = max(1, int(win_ms / 1000 * sr)); e = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(fr) / fr, 'same'))
    e = np.interp(np.linspace(0, 1, npts), np.linspace(0, 1, len(e)), e); return e / (e.max() + 1e-9)


def read_wav(ch, t0, dur):
    with wave.open(CFG.wav_path(ch), 'rb') as w:
        sr = w.getframerate(); w.setpos(min(max(0, int(t0 * sr)), w.getnframes()))
        x = np.frombuffer(w.readframes(int(dur * sr)), np.int16).astype(np.float32) / 32768.0
    return resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x


plc = {}
manifest = []
for n, (i, correct) in enumerate(picks):
    ch, s, e, y = te_items[i]
    t0, dur = s - CTX, (e - s) + 2 * CTX
    aw = read_wav(ch, t0, dur)
    if ch not in plc:
        plc[ch] = np.memmap(CFG.bin_path(ch), dtype=np.float32, mode='r')
    lag = D.read_lag_s(ch); a = int((t0 + lag) * CFG.cap_sr)
    pw = np.array(plc[ch][a:a + int(dur * CFG.cap_sr)], np.float32)
    raw4 = np.array(plc[ch][a:a + int(c14.win_s * CFG.cap_sr)], np.float32)
    raw4 = np.pad(raw4, (0, max(0, int(c14.win_s * CFG.cap_sr) - len(raw4))))
    xr = resample_poly(raw4, c14.in_sr, CFG.cap_sr).astype(np.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        gen = m14.sample(torch.from_numpy(xr)[None].to(dev))[0].float().cpu() * msd + mm
    fr = max(6, int(round(dur * c14.fps))); gen = gen[:, :fr]
    real_mel = torch.log(melfn(torch.from_numpy(aw)) + 1e-5)
    tag = f'ex{n}'
    savewav(f'{OUT}/{tag}_orig.wav', aw)
    savewav(f'{OUT}/{tag}_realGL.wav', gl(real_mel))
    savewav(f'{OUT}/{tag}_powerGL.wav', gl(gen))
    # envelope plot
    f, ax = plt.subplots(figsize=(3.4, 1.15), dpi=120); t = np.linspace(0, 1, 48)
    ax.plot(t, env(aw, 16000), color='#2563d6', lw=2); ax.plot(t, env(pw, CFG.cap_sr), color='#e08a10', lw=2)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_color('#cbd3de') for sp in ax.spines.values()]
    f.tight_layout(pad=0.2); f.savefig(f'{OUT}/{tag}_env.png', transparent=True); plt.close(f)
    # mel plot (real | gen)
    f, ax = plt.subplots(1, 2, figsize=(5.0, 1.5), dpi=120)
    vmn, vmx = real_mel.min().item(), real_mel.max().item()
    ax[0].imshow(real_mel.numpy(), origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    ax[1].imshow(gen.numpy(), origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
    f.subplots_adjust(wspace=0.06); f.savefig(f'{OUT}/{tag}_mel.png', bbox_inches='tight', transparent=True); plt.close(f)
    manifest.append({'tag': tag, 'correct': bool(correct), 'true': WORD, 'pred': words[preds[i]],
                     'chunk': ch, 'dur': round(dur, 2)})
    print(f'[ex{n}] {"OK " if correct else "ERR"} pred={words[preds[i]]} ({ch})', flush=True)
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
print('[done]', len(manifest), '->', OUT)
