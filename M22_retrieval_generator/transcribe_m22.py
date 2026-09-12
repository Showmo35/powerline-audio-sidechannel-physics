#!/usr/bin/env python3
"""
transcribe_m22.py — can M22 do speech recognition?

Pipeline per TEST sentence:  powerline .bin window -> M22 generated mel
  -> Griffin-Lim vocode -> 16 kHz audio -> Whisper -> transcription.
Control: transcribe the REAL reference audio too (Whisper should nail it), so any
failure is the M22 mel, not the ASR. Prints GT vs M22-hyp vs real-hyp for 5 test
utterances + word error rate.

Test-only (chunk % 12 == 0) — M22 never trained on these. Run on GPU (debug-nextgen).
"""
import os, json, numpy as np, torch, wave
import torchaudio

from config import CFG
import models as M
import data_io as io

ROOT = '<REPO_ROOT>'
MAN = os.path.join(ROOT, 'M15_soundbar_melgen_word', 'full_manifest.json')
OUT = 'outputs/asr'; os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── M22 generator ──
ck = torch.load('outputs/best.pt', map_location=dev, weights_only=False)
model = M.build(CFG).to(dev); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
print(f'[m22] loaded step={ck.get("step")} mel_mean={mm:.2f} std={ms:.2f}', flush=True)

# ── Griffin-Lim (invert M22's mel: n_fft=1024 hop=160 win=640 80-mel fmax=8000) ──
_fb = torchaudio.functional.melscale_fbanks(CFG.n_fft // 2 + 1, CFG.fmin, CFG.fmax,
                                            CFG.n_mels, CFG.ref_sr, norm=None, mel_scale='htk')
_pinv = torch.linalg.pinv(_fb.T)
_gl = torchaudio.transforms.GriffinLim(CFG.n_fft, 64, win_length=CFG.win_length,
                                       hop_length=CFG.hop, power=2.0)


def gl(logmel):
    lin = (_pinv @ (logmel.exp() - CFG.log_eps).clamp(min=0)).clamp(min=0)
    w = _gl(lin)
    return (w / (w.abs().max() + 1e-8)).numpy()


def savewav(p, x, sr=CFG.ref_sr):
    with wave.open(p, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


import whisper
wm = whisper.load_model('small', device=dev)


def whisp(x):
    r = wm.transcribe(np.ascontiguousarray(x.astype(np.float32)), language='en', fp16=(dev == 'cuda'),
                      verbose=False)
    return ' '.join(''.join(c for c in r['text'].lower() if c.isalpha() or c == ' ').split())


def norm(t):
    return ' '.join(''.join(c for c in t.lower() if c.isalpha() or c == ' ').split())


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[len(r), len(h)] / max(1, len(r))


# ── pick 5 test sentences that fit the 4 s window ──
man = json.load(open(MAN))
test = [r for r in man if CFG.is_test(r['chunk']) and 2.0 <= r['dur_s'] <= 3.8
        and len(r['text'].split()) >= 4]
rng = np.random.RandomState(1); rng.shuffle(test)
picks = test[:5]

rows = []
for k, r in enumerate(picks):
    ch, s, dur = r['chunk'], r['start_s'], min(r['dur_s'], CFG.win_s)
    lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0
    raw = io.read_bin_window(CFG.bin_path(ch), s + lag, CFG.win_s, CFG.cap_sr, CFG.in_sr).astype(np.float32)
    raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
    x = np.zeros(CFG.in_len, np.float32); x[:min(len(raw), CFG.in_len)] = raw[:CFG.in_len]
    with torch.no_grad():
        gen = model.sample(torch.from_numpy(x)[None].to(dev))[0].float().cpu() * ms + mm   # (80,400)
    vfr = min(CFG.n_frames, int(round(dur * CFG.fps)))
    m22_wav = gl(gen[:, :vfr])
    savewav(f'{OUT}/utt{k}_m22GL.wav', m22_wav)
    real16 = io.read_wav_window(CFG.wav_path(ch), s, dur, CFG.ref_sr).astype(np.float32)
    savewav(f'{OUT}/utt{k}_real.wav', real16)

    gt = norm(r['text']); h_m22 = whisp(m22_wav); h_real = whisp(real16)
    rows.append({'chunk': ch, 'gt': gt, 'm22': h_m22, 'real': h_real,
                 'wer_m22': wer(gt, h_m22), 'wer_real': wer(gt, h_real)})
    print(f'\n=== utt{k} [{ch}] ===', flush=True)
    print(f'  GT        : {gt}', flush=True)
    print(f'  M22->ASR  : {h_m22}   (WER {rows[-1]["wer_m22"]*100:.0f}%)', flush=True)
    print(f'  REAL->ASR : {h_real}   (WER {rows[-1]["wer_real"]*100:.0f}%)', flush=True)

json.dump(rows, open(f'{OUT}/asr_results.json', 'w'), indent=1)
print(f'\n==== SUMMARY (5 test sentences) ====', flush=True)
print(f'  mean WER  M22->ASR  = {np.mean([r["wer_m22"] for r in rows])*100:.0f}%', flush=True)
print(f'  mean WER  REAL->ASR = {np.mean([r["wer_real"] for r in rows])*100:.0f}%  (ASR sanity)', flush=True)
print(f'  wavs + json -> {OUT}/', flush=True)
