#!/usr/bin/env python3
"""For 4 words, save audio: original | real-mel->GL | powerline-generated-mel->GL."""
import sys, os, json, importlib
import numpy as np, torch, wave, torchaudio
from scipy.signal import resample_poly

WORDS = ['missus', 'said', 'little', 'time']
NOCC, CTX = 3, 0.12
ROOT = '<REPO_ROOT>'
BIN = ROOT + '/Powerline_Data_Captures/soundbar_bin_captures'
WAVD = ROOT + '/Powerline_Data_Captures/audio_chunks'
OUT = ROOT + '/M20_powerline_word_classifier/outputs/word_audio'
os.makedirs(OUT, exist_ok=True)
CAP_SR = 200_000
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
occ_all = json.load(open(ROOT + '/M20_powerline_word_classifier/word_index.json'))


def lag_s(ch):
    p = f'{BIN}/{ch}.lag'
    return (float(open(p).read().strip()) / 1000.0) if os.path.exists(p) else 0.0


def read_wav(ch, t0, dur, sr_out=16000):
    with wave.open(f'{WAVD}/{ch}.wav', 'rb') as w:
        sr = w.getframerate(); w.setpos(min(max(0, int(t0 * sr)), w.getnframes()))
        x = np.frombuffer(w.readframes(int(dur * sr)), np.int16).astype(np.float32) / 32768.0
    return resample_poly(x, sr_out, sr).astype(np.float32) if sr != sr_out else x


# M14 (evict M20 modules for name clash)
for _m in ('config', 'models'):
    sys.modules.pop(_m, None)
sys.path.insert(0, ROOT + '/M14_new_setup_soundbar_melgen')
c14 = importlib.import_module('config').CFG
m14 = importlib.import_module('models').build(c14).to(dev)
_ck = torch.load(ROOT + '/M14_new_setup_soundbar_melgen/outputs/cluster_a100/last.pt',
                 map_location=dev, weights_only=False)
m14.load_state_dict(_ck.get('ema', _ck['model'])); m14.eval()
mm, ms = _ck['mel_mean'], _ck['mel_std']
melfn = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160,
                                             win_length=640, n_mels=80, f_min=0, f_max=8000, power=2.0)
_fb = torchaudio.functional.melscale_fbanks(513, 0, 8000, 80, 16000, norm=None, mel_scale='htk')
_pinv = torch.linalg.pinv(_fb.T)
_gl = torchaudio.transforms.GriffinLim(n_fft=1024, n_iter=64, win_length=640, hop_length=160, power=2.0)


def gl(logmel):
    powmel = (logmel.exp() - 1e-5).clamp(min=0)
    spec = (_pinv @ powmel).clamp(min=0)
    w = _gl(spec); return (w / (w.abs().max() + 1e-8)).numpy()


def save(name, x, sr=16000):
    with wave.open(f'{OUT}/{name}.wav', 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


plc = {}
manifest = []
for word in WORDS:
    picks = [r for r in occ_all[word] if 0.28 <= r[2] - r[1] <= 0.6][:NOCC]
    for k, (ch, s, e) in enumerate(picks):
        t0, dur = s - CTX, (e - s) + 2 * CTX
        aw = read_wav(ch, t0, dur)
        real_mel = torch.log(melfn(torch.from_numpy(aw)) + 1e-5)
        if ch not in plc:
            plc[ch] = np.memmap(f'{BIN}/{ch}.bin', dtype=np.float32, mode='r')
        a = int((t0 + lag_s(ch)) * CAP_SR); raw = np.array(plc[ch][a:a + int(c14.win_s * CAP_SR)], np.float32)
        raw = np.pad(raw, (0, max(0, int(c14.win_s * CAP_SR) - len(raw))))
        xr = resample_poly(raw, c14.in_sr, CAP_SR).astype(np.float32)
        torch.manual_seed(0)
        with torch.no_grad():
            gen = m14.sample(torch.from_numpy(xr)[None].to(dev))[0].float().cpu() * ms + mm
        fr = max(6, int(round(dur * c14.fps)))
        gen = gen[:, :fr]
        tag = f'{word}_{k}'
        save(f'{tag}_orig', aw)
        save(f'{tag}_realGL', gl(real_mel))
        save(f'{tag}_powerGL', gl(gen))
        manifest.append({'word': word, 'k': k, 'chunk': ch, 'dur': round(dur, 2)})
        print(f'[audio] {tag} ({ch})', flush=True)
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
print('[done]', len(manifest), 'examples ->', OUT)
