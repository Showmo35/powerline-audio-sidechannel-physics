#!/usr/bin/env python3
"""
6 occurrences of the SAME word, powerline vs original audio.
Row = one occurrence. Cols: envelope (audio vs powerline overlaid) | real audio mel
| powerline-generated (M14/M16) mel. Shows whether the word's envelope is a
consistent powerline signature, and how the generated mel compares to the real one.
"""
import sys, os, json, importlib
import numpy as np, torch, wave
from scipy.signal import resample_poly
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

WORD = os.environ.get('WORD', 'before')
NEX = 6
CTX = 0.08
ROOT = '<REPO_ROOT>'
BIN = ROOT + '/Powerline_Data_Captures/soundbar_bin_captures'
WAVD = ROOT + '/Powerline_Data_Captures/audio_chunks'
CAP_SR = 200_000
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

occ = json.load(open(ROOT + '/M20_powerline_word_classifier/word_index.json'))[WORD]
picks = [r for r in occ if 0.28 <= r[2] - r[1] <= 0.6][:NEX]


def lag_s(ch):
    p = f'{BIN}/{ch}.lag'
    return (float(open(p).read().strip()) / 1000.0) if os.path.exists(p) else 0.0


def read_wav(ch, t0, dur, sr_out=16000):
    with wave.open(f'{WAVD}/{ch}.wav', 'rb') as w:
        sr = w.getframerate(); w.setpos(min(max(0, int(t0 * sr)), w.getnframes()))
        x = np.frombuffer(w.readframes(int(dur * sr)), np.int16).astype(np.float32) / 32768.0
    return resample_poly(x, sr_out, sr).astype(np.float32) if sr != sr_out else x


def envelope(x, sr, npts=44, win_ms=20):
    fr = max(1, int(win_ms / 1000 * sr))
    e = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(fr) / fr, 'same'))
    e = np.interp(np.linspace(0, 1, npts), np.linspace(0, 1, len(e)), e)
    return e / (e.max() + 1e-9)


import torchaudio
melfn = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160,
                                             win_length=640, n_mels=80, f_min=0, f_max=8000, power=2.0)

# M14 for generated mel
for _m in ('config', 'models'):
    sys.modules.pop(_m, None)
sys.path.insert(0, ROOT + '/M14_new_setup_soundbar_melgen')
cfg14 = importlib.import_module('config').CFG
m14 = importlib.import_module('models').build(cfg14).to(dev)
c14 = torch.load(ROOT + '/M14_new_setup_soundbar_melgen/outputs/cluster_a100/last.pt',
                 map_location=dev, weights_only=False)
m14.load_state_dict(c14.get('ema', c14['model'])); m14.eval()
mm, ms = c14['mel_mean'], c14['mel_std']
plc = {}


def gen_mel(ch, t0, dur):
    if ch not in plc:
        plc[ch] = np.memmap(f'{BIN}/{ch}.bin', dtype=np.float32, mode='r')
    a = int((t0 + lag_s(ch)) * CAP_SR); raw = np.array(plc[ch][a:a + int(cfg14.win_s * CAP_SR)], np.float32)
    raw = np.pad(raw, (0, max(0, int(cfg14.win_s * CAP_SR) - len(raw))))
    xr = resample_poly(raw, cfg14.in_sr, CAP_SR).astype(np.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        g = m14.sample(torch.from_numpy(xr)[None].to(dev))[0].float().cpu() * ms + mm
    return g[:, :max(6, int(round(dur * cfg14.fps)))].numpy()


fig, ax = plt.subplots(NEX, 3, figsize=(13, 2.1 * NEX), dpi=130,
                       gridspec_kw={'width_ratios': [1.1, 1, 1]})
for r, (ch, s, e) in enumerate(picks):
    t0, dur = s - CTX, (e - s) + 2 * CTX
    aw = read_wav(ch, t0, dur)
    pl = plc.get(ch);
    if pl is None:
        pl = plc.setdefault(ch, np.memmap(f'{BIN}/{ch}.bin', dtype=np.float32, mode='r'))
    pa = int((t0 + lag_s(ch)) * CAP_SR); pw = np.array(pl[pa:pa + int(dur * CAP_SR)], np.float32)
    ea, ep = envelope(aw, 16000), envelope(pw, CAP_SR)
    t = np.linspace(0, 1, len(ea))
    ax[r, 0].plot(t, ea, color='#2c7fb8', lw=2, label='audio')
    ax[r, 0].plot(t, ep, color='#e6550d', lw=2, label='powerline')
    ax[r, 0].set_yticks([]); ax[r, 0].set_xticks([])
    ax[r, 0].set_ylabel(f'ex {r+1}', fontsize=11)
    amel = torch.log(melfn(torch.from_numpy(aw)) + 1e-5).numpy()
    gmel = gen_mel(ch, t0, dur)
    ax[r, 1].imshow(amel, origin='lower', aspect='auto', cmap='magma'); ax[r, 1].set_xticks([]); ax[r, 1].set_yticks([])
    ax[r, 2].imshow(gmel, origin='lower', aspect='auto', cmap='magma'); ax[r, 2].set_xticks([]); ax[r, 2].set_yticks([])
    if r == 0:
        ax[r, 0].set_title('ENVELOPE  (audio=blue, powerline=orange)', fontsize=11)
        ax[r, 0].legend(loc='upper left', fontsize=8)
        ax[r, 1].set_title('REAL audio mel', fontsize=11)
        ax[r, 2].set_title('powerline mel (M14/M16-generated)', fontsize=11)
fig.suptitle(f'Same word "{WORD}" x{NEX} occurrences — powerline vs original audio', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT + f'/M20_powerline_word_classifier/outputs/sameword_{WORD}.png'
fig.savefig(out, bbox_inches='tight'); print('[saved]', out, 'from', [(p[0]) for p in picks])
