#!/usr/bin/env python3
"""inspect_2900.py — focused look at the 2900 kHz candidate carrier.

Lag-aligns the AM-demodulated envelope against the reference audio (the survey's
overlay does NOT apply the lag), shows the zoom PSD + demod spectrogram + aligned
envelope overlay, and writes the demodulated baseband to a WAV so you can listen.
"""
import argparse, os, wave
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as dsp
from math import gcd

AUD_SR = 22_050
DARK, PANEL, GRID = '#0d0d14', '#1e1e2e', '#313244'
TEXT, MUTED = '#cdd6f4', '#a6adc8'
BLUE, PURP, ORNG, YLW = '#89b4fa', '#cba6f7', '#fab387', '#f9e2af'

ap = argparse.ArgumentParser()
ap.add_argument('--capture', default='wideband_8m_8s.bin')
ap.add_argument('--samp-rate', type=float, default=7692308)
ap.add_argument('--audio', default='hello5.wav')
ap.add_argument('--fc', type=float, default=2900e3)
ap.add_argument('--bw', type=float, default=20e3)
ap.add_argument('--out-png', default='inspect_2900.png')
ap.add_argument('--out-wav', default='demod_2900.wav')
args = ap.parse_args()

def resample_to(x, src, dst):
    if src == dst: return x.astype(np.float32)
    g = gcd(int(src), int(dst)); return dsp.resample_poly(x, dst//g, src//g).astype(np.float32)

def read_wav(p):
    w = wave.open(p,'rb'); sr=w.getframerate(); ch=w.getnchannels()
    d=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
    w.close()
    if ch>1: d=d.reshape(-1,ch).mean(1)
    return d, sr

def write_wav(p, x, sr):
    x = x/(np.abs(x).max()+1e-9)
    wave.open(p,'wb')
    w=wave.open(p,'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((x*32767).astype(np.int16).tobytes()); w.close()

sr = args.samp_rate
cap = np.fromfile(args.capture, dtype=np.float32)
ref, ref_sr = read_wav(args.audio)
print(f'[cap] {len(cap):,} samp {len(cap)/sr:.2f}s  ref {len(ref)/ref_sr:.2f}s')

# AM demod around fc
lo, hi = args.fc-args.bw, args.fc+args.bw
sos = dsp.butter(4, [lo, hi], 'bp', fs=sr, output='sos')
bp = dsp.sosfiltfilt(sos, cap - cap.mean())
env = np.abs(dsp.hilbert(bp)); env = env - env.mean()
print(f'[demod] fc={args.fc/1e3:.0f}kHz bw=±{args.bw/1e3:.0f}kHz')

# to audio rate
env_a = resample_to(env, int(sr), AUD_SR)
ref_a = resample_to(ref, int(ref_sr), AUD_SR)
write_wav(args.out_wav, env_a, AUD_SR)
print(f'[wav] {args.out_wav}')

# frame-energy envelopes for lag search
fr = int(0.02*AUD_SR)
ne = len(env_a)//fr; nr = len(ref_a)//fr
ee = np.array([np.sqrt(np.mean(env_a[i*fr:(i+1)*fr]**2)) for i in range(ne)])
rr = np.array([np.sqrt(np.mean(ref_a[i*fr:(i+1)*fr]**2)) for i in range(nr)])
een=(ee-ee.mean())/(ee.std()+1e-12); rrn=(rr-rr.mean())/(rr.std()+1e-12)
# full cross-correlation, allow ref to sit anywhere inside the longer demod
xc = dsp.correlate(een, rrn, mode='full'); lags = np.arange(-(len(rrn)-1), len(een))
xc = xc/min(len(een),len(rrn))
k = np.argmax(xc); best_lag = lags[k]; best_r = xc[k]
print(f'[align] best lag = {best_lag*20} ms   lag-aligned r = {best_r:.3f}')

# build aligned overlay (frame domain)
s = best_lag
if s>=0: a_e=een[s:s+len(rrn)]; a_r=rrn[:len(a_e)]
else: a_r=rrn[-s:]; a_e=een[:len(a_r)]
m=min(len(a_e),len(a_r)); a_e,a_r=a_e[:m],a_r[:m]; t=np.arange(m)*0.02

fig=plt.figure(figsize=(15,11),facecolor=DARK)
fig.suptitle(f'2900 kHz candidate — lag-aligned r={best_r:.3f} @ {best_lag*20} ms  (±{args.bw/1e3:.0f} kHz)',
             color=TEXT,fontsize=13,fontweight='bold')
gs=gridspec.GridSpec(3,2,figure=fig,hspace=0.4,wspace=0.25,left=0.07,right=0.97,top=0.92,bottom=0.07)

def style(ax,t,xl,yl,grid=True):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED,labelsize=8); ax.set_title(t,color=TEXT,fontsize=9,loc='left')
    ax.set_xlabel(xl,color=MUTED,fontsize=8); ax.set_ylabel(yl,color=MUTED,fontsize=8)
    if grid: ax.grid(True,color=GRID,lw=0.5,alpha=0.6)

# zoom PSD around carrier
f,P=dsp.welch(cap-cap.mean(),fs=sr,nperseg=1<<16,window='blackman'); Pdb=10*np.log10(P+1e-30)
ax=fig.add_subplot(gs[0,:]); m=np.abs(f-args.fc)<=6*args.bw
ax.plot((f[m]-args.fc)/1e3,Pdb[m],color=ORNG,lw=1.0); ax.axvline(0,color=YLW,ls='--',lw=0.8)
ax.axvline(-args.bw/1e3,color=GRID,ls=':'); ax.axvline(args.bw/1e3,color=GRID,ls=':')
style(ax,'Zoom PSD @ 2900 kHz (demod band between dotted lines)','Offset from carrier (kHz)','PSD (dB)')

# demod spectrogram
ax=fig.add_subplot(gs[1,0])
fsp,tsp,S=dsp.spectrogram(env_a,fs=AUD_SR,nperseg=1024,noverlap=768,window='hann')
fm=fsp<=5000; Sdb=10*np.log10(S[fm]+1e-30)
ax.pcolormesh(tsp,fsp[fm],Sdb,shading='gouraud',cmap='inferno',
              vmin=np.percentile(Sdb,5),vmax=np.percentile(Sdb,99),rasterized=True)
style(ax,'Demodulated envelope spectrogram','Time (s)','Audio freq (Hz)',grid=False)

# reference spectrogram for comparison
ax=fig.add_subplot(gs[1,1])
fsp2,tsp2,S2=dsp.spectrogram(ref_a,fs=AUD_SR,nperseg=1024,noverlap=768,window='hann')
fm2=fsp2<=5000; S2db=10*np.log10(S2[fm2]+1e-30)
ax.pcolormesh(tsp2,fsp2[fm2],S2db,shading='gouraud',cmap='inferno',
              vmin=np.percentile(S2db,5),vmax=np.percentile(S2db,99),rasterized=True)
style(ax,'REFERENCE audio spectrogram (hello5.wav)','Time (s)','Audio freq (Hz)',grid=False)

# aligned envelope overlay
ax=fig.add_subplot(gs[2,:])
ax.plot(t,(a_r-a_r.min())/(a_r.max()-a_r.min()+1e-9),color=BLUE,lw=1.6,label='reference audio env')
ax.plot(t,(a_e-a_e.min())/(a_e.max()-a_e.min()+1e-9),color=PURP,lw=1.4,alpha=0.85,label='demod 2900 kHz env (lag-aligned)')
style(ax,'Lag-aligned envelope overlay','Time (s)','norm. env')
ax.legend(fontsize=9,facecolor=PANEL,edgecolor=GRID,labelcolor=TEXT)

fig.savefig(args.out_png,dpi=140,bbox_inches='tight',facecolor=DARK)
print(f'[png] {args.out_png}')
