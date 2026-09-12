#!/usr/bin/env python3
"""viz_blind.py — visualize the blind-vs-leaky speech_presence analysis."""
import os, wave
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

import neural_vocoder as nv
from am_vocoder import extract_mel_am
from am_reconstruct import detect_mains

SD = os.path.dirname(os.path.abspath(__file__))
CAPROOT = '<REPO_ROOT>/Powerline_Data_Captures'
CAP = os.path.join(CAPROOT, 'soundbar_bin_captures', 'chunk_002.bin')
WAV = os.path.join(CAPROOT, 'audio_chunks', 'chunk_002.wav')
DUR = 60.0

DARK='#0d0d14'; PANEL='#1e1e2e'; GRID='#313244'; TEXT='#cdd6f4'; MUT='#a6adc8'
BLUE='#89b4fa'; GRN='#a6e3a1'; RED='#f38ba8'; YLW='#f9e2af'; PURP='#cba6f7'


def load_wav(path, dur, sr_out):
    with wave.open(path, 'rb') as w:
        sr, nch = w.getframerate(), w.getnchannels()
        raw = w.readframes(int(dur*sr))
    x = np.frombuffer(raw, np.int16).astype(np.float32)/32768.0
    if nch > 1: x = x.reshape(-1, nch).mean(1)
    return nv.downsample(x, sr, sr_out) if sr != sr_out else x


def logmel(x): return np.log(np.maximum(nv.extract_mel(x), 1e-5))
def rms(x, sr=nv.AUD_SR, fs=0.05):
    f=int(fs*sr); n=len(x)//f
    return np.sqrt(np.mean(x[:n*f].reshape(n,f)**2, axis=1))


# ── load ──────────────────────────────────────────────────────────────────────
ref = load_wav(WAV, DUR, nv.AUD_SR)
blind = load_wav(os.path.join(SD,'blind_recon.wav'), DUR, nv.AUD_SR)
leaky = load_wav(os.path.join(SD,'leaky_recon.wav'), DUR, nv.AUD_SR)
cap = np.fromfile(CAP, np.float32, count=int(DUR*nv.CAP_SR))
cap_ds = nv.downsample(cap, nv.CAP_SR, nv.AUD_SR)
n = min(len(ref), len(cap_ds)); ref, cap_ds = ref[:n], cap_ds[:n]
fm = detect_mains(cap_ds, nv.AUD_SR, 60.0)
pl_mel = np.log(np.maximum(extract_mel_am(cap_ds, fm, 8), 1e-5))

m_ref, m_bl, m_lk = logmel(ref), logmel(blind), logmel(leaky)

# metrics (reference used only to score)
def env_r(rec):
    a, r = rms(ref), rms(rec); k=min(len(a),len(r)); return pearsonr(a[:k], r[:k])[0]
def mel_r(mrec):
    T=min(m_ref.shape[1],mrec.shape[1]); sp=m_ref[:,:T].mean(0)>np.median(m_ref[:,:T].mean(0))
    return np.corrcoef(mrec[:,:T][:,sp].ravel(), m_ref[:,:T][:,sp].ravel())[0,1]
er_bl, mr_bl = env_r(blind), mel_r(m_bl)
er_lk, mr_lk = env_r(leaky), mel_r(m_lk)

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor=DARK)
fig.suptitle('Blind vs leaky speech_presence  —  chunk_002 [0–60 s]\n'
             'reference touches the powerline only in the LEAKY arm; both score WER = 100% (empty)',
             color=TEXT, fontsize=13, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.18,
                       left=0.06, right=0.97, top=0.9, bottom=0.07)

def mp(ax, M, title, cmap='magma'):
    ax.set_facecolor(PANEL)
    t = np.linspace(0, DUR, M.shape[1])
    ax.pcolormesh(t, np.arange(M.shape[0]), M, shading='auto', cmap=cmap,
                  vmin=np.percentile(M,5), vmax=np.percentile(M,99), rasterized=True)
    ax.set_title(title, color=TEXT, fontsize=10, loc='left', pad=4)
    ax.set_ylabel('mel bin', color=MUT, fontsize=8)
    ax.tick_params(colors=MUT, labelsize=7)

mp(fig.add_subplot(gs[0,0]), m_ref, 'Reference (CLEAN) mel — what was actually said')
mp(fig.add_subplot(gs[0,1]), pl_mel, 'Powerline AM-sideband mel (raw input to both arms)', 'viridis')
mp(fig.add_subplot(gs[1,0]), m_bl, f'BLIND reconstruction (no reference)   env r={er_bl:.2f}  mel r={mr_bl:.2f}  WER=100%')
mp(fig.add_subplot(gs[1,1]), m_lk, f'LEAKY reconstruction (reference-injected)   env r={er_lk:.2f}  mel r={mr_lk:.2f}  WER=100%')

# envelope overlay
ax = fig.add_subplot(gs[2,:]); ax.set_facecolor(PANEL)
for sig, c, lab in ((ref,BLUE,'Reference RMS'),(blind,GRN,'BLIND RMS'),(leaky,PURP,'LEAKY RMS')):
    e = rms(sig); e = e/(e.max()+1e-9); ax.plot(np.arange(len(e))*0.05, e, color=c, lw=1.6, label=lab, alpha=0.9)
ax.set_title('Envelope overlay (50 ms RMS) — LEAKY tracks the reference (injected envelope); '
             'BLIND does not. Neither yields words.', color=TEXT, fontsize=10, loc='left', pad=4)
ax.set_xlabel('time (s)', color=MUT, fontsize=8); ax.set_ylabel('norm. RMS', color=MUT, fontsize=8)
ax.set_xlim(0, DUR); ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.tick_params(colors=MUT, labelsize=7)
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
ax.text(0.99, 0.92, f'env r: BLIND {er_bl:.2f}  vs  LEAKY {er_lk:.2f}\n(0.71 was reference leakage)',
        transform=ax.transAxes, color=YLW, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', fc=PANEL, ec=GRID))

out = os.path.join(SD, 'speech_presence_blind.png')
fig.savefig(out, dpi=140, bbox_inches='tight', facecolor=DARK)
print('saved', out)
print(f'BLIND env_r={er_bl:.3f} mel_r={mr_bl:.3f} | LEAKY env_r={er_lk:.3f} mel_r={mr_lk:.3f}')
