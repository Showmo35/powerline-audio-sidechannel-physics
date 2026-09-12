#!/usr/bin/env python3
"""For each WRONG 'before' detection, generate assets for a real occurrence of the
PREDICTED word (its envelope, real+generated mels, audio) — to compare against the
misclassified 'before'. Saves {tag}_pred_* alongside the existing before assets."""
import sys, os, json, importlib
import numpy as np, torch, wave, torchaudio
from scipy.signal import resample_poly
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from config import CFG, OUT_DIR
import dataset as D

CTX = 0.10
OUT = os.path.join(OUT_DIR, 'before')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
man = json.load(open(f'{OUT}/manifest.json'))
wrong = [m for m in man if not m['correct']]
occ_all = json.load(open(os.path.join(os.path.dirname(__file__), 'word_index.json')))

# M14 + GL
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
_pinv = torch.linalg.pinv(_fb.T); _gl = torchaudio.transforms.GriffinLim(1024, 64, win_length=640, hop_length=160, power=2.0)


def gl(lm):
    w = _gl((_pinv @ (lm.exp() - 1e-5).clamp(min=0)).clamp(min=0)); return (w / (w.abs().max() + 1e-8)).numpy()
def savewav(p, x):
    with wave.open(p, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
def env(x, sr, npts=48, ms=30):
    fr = max(1, int(ms / 1000 * sr)); e = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(fr) / fr, 'same'))
    e = np.interp(np.linspace(0, 1, npts), np.linspace(0, 1, len(e)), e); return e / (e.max() + 1e-9)
def read_wav(ch, t0, dur):
    with wave.open(CFG.wav_path(ch), 'rb') as w:
        sr = w.getframerate(); w.setpos(min(max(0, int(t0 * sr)), w.getnframes()))
        x = np.frombuffer(w.readframes(int(dur * sr)), np.int16).astype(np.float32) / 32768.0
    return resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x


plc = {}
rng = np.random.RandomState(7)
pred_manifest = []
for m in wrong:
    word = m['pred']
    cands = [r for r in occ_all[word] if 0.28 <= r[2] - r[1] <= 0.55 and D.CFG.is_test(r[0])]
    if not cands:
        cands = [r for r in occ_all[word] if 0.28 <= r[2] - r[1] <= 0.55]
    ch, s, e = cands[rng.randint(len(cands))]
    t0, dur = s - CTX, (e - s) + 2 * CTX
    aw = read_wav(ch, t0, dur)
    if ch not in plc:
        plc[ch] = np.memmap(CFG.bin_path(ch), dtype=np.float32, mode='r')
    lag = D.read_lag_s(ch); a = int((t0 + lag) * CFG.cap_sr)
    pw = np.array(plc[ch][a:a + int(dur * CFG.cap_sr)], np.float32)
    raw4 = np.pad(np.array(plc[ch][a:a + int(c14.win_s * CFG.cap_sr)], np.float32),
                  (0, max(0, int(c14.win_s * CFG.cap_sr) - int(dur * CFG.cap_sr))))[:int(c14.win_s * CFG.cap_sr)]
    xr = resample_poly(raw4, c14.in_sr, CFG.cap_sr).astype(np.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        gen = m14.sample(torch.from_numpy(xr)[None].to(dev))[0].float().cpu() * msd + mm
    fr = max(6, int(round(dur * c14.fps))); gen = gen[:, :fr]
    real_mel = torch.log(melfn(torch.from_numpy(aw)) + 1e-5)
    t = m['tag']
    savewav(f'{OUT}/{t}_pred_orig.wav', aw); savewav(f'{OUT}/{t}_pred_realGL.wav', gl(real_mel)); savewav(f'{OUT}/{t}_pred_powerGL.wav', gl(gen))
    f, ax = plt.subplots(figsize=(3.4, 1.15), dpi=120); tt = np.linspace(0, 1, 48)
    ax.plot(tt, env(aw, 16000), color='#2563d6', lw=2); ax.plot(tt, env(pw, CFG.cap_sr), color='#e08a10', lw=2)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_color('#cbd3de') for sp in ax.spines.values()]
    f.tight_layout(pad=0.2); f.savefig(f'{OUT}/{t}_pred_env.png', transparent=True); plt.close(f)
    f, ax = plt.subplots(1, 2, figsize=(5.0, 1.5), dpi=120); vmn, vmx = real_mel.min().item(), real_mel.max().item()
    ax[0].imshow(real_mel.numpy(), origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    ax[1].imshow(gen.numpy(), origin='lower', aspect='auto', cmap='magma', vmin=vmn, vmax=vmx)
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
    f.subplots_adjust(wspace=0.06); f.savefig(f'{OUT}/{t}_pred_mel.png', bbox_inches='tight', transparent=True); plt.close(f)
    pred_manifest.append({'tag': t, 'pred_word': word, 'chunk': ch, 'dur': round(dur, 2)})
    print(f'[{t}] pred word "{word}" from {ch}', flush=True)
json.dump(pred_manifest, open(f'{OUT}/manifest_pred.json', 'w'), indent=1)
print('[done]', len(pred_manifest))
