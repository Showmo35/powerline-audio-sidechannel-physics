import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as dsp
import neural_vocoder as nv
from am_reconstruct import detect_mains

cap = nv.load_capture('capture.bin')
aud = nv.decode_audio('hello5.wav', nv.AUD_SR)
dur = min(len(cap)/nv.CAP_SR, len(aud)/nv.AUD_SR)
cap = cap[:int(dur*nv.CAP_SR)]; aud = aud[:int(dur*nv.AUD_SR)]
cap_ds = nv.downsample(cap, nv.CAP_SR, nv.AUD_SR)[:len(aud)]
fm = detect_mains(cap_ds, nv.AUD_SR, 60.0)

# High-res STFT, both signals on same grid
nps = 8192; nov = nps*7//8
f, t, Ya = dsp.stft(aud,    fs=nv.AUD_SR, nperseg=nps, noverlap=nov, window='hann')
_, _, Yc = dsp.stft(cap_ds, fs=nv.AUD_SR, nperseg=nps, noverlap=nov, window='hann')
Aa, Ac = np.abs(Ya), np.abs(Yc)
df = f[1]-f[0]

# voiced frames: top 30% energy frames of reference
en = Aa.sum(0); voiced = en > np.percentile(en, 70)
print(f'voiced frames: {voiced.sum()}/{len(en)}  df={df:.2f} Hz')

# Sideband-shifted capture: audio(f) <- capture(fm + f)
shift = int(round(fm/df))
Ac_sb = np.zeros_like(Ac); Ac_sb[:Ac.shape[0]-shift] = Ac[shift:]

band = (f>=200)&(f<=3500)   # formant band
def shape_r(X, Y):
    x = X[band][:,voiced]; y = Y[band][:,voiced]
    x = x - x.mean(0, keepdims=True); y = y - y.mean(0, keepdims=True)
    return np.corrcoef(x.ravel(), y.ravel())[0,1]

print(f'[hi-res, formant band, envelope-removed shape r]')
print(f'  capture (no shift) vs audio : {shape_r(Ac,    Aa):+.3f}')
print(f'  capture SIDEBAND-shifted     : {shape_r(Ac_sb, Aa):+.3f}')
print(f'  (audio vs audio sanity = 1.0): {shape_r(Aa,    Aa):+.3f}')

# Per-voiced-frame shifted-sideband correlation distribution
rs=[]
for j in np.where(voiced)[0]:
    a=Aa[band,j]; c=Ac_sb[band,j]
    a=a-a.mean(); c=c-c.mean()
    s=(a.std()*c.std())
    if s>0: rs.append((a*c).mean()/s)
rs=np.array(rs)
print(f'  per-frame sideband shape r: mean={rs.mean():+.3f} median={np.median(rs):+.3f} '
      f'frac>0.3={np.mean(rs>0.3):.2f}')

# Figure: average voiced spectrum, audio vs shifted-sideband capture
fig,ax=plt.subplots(2,1,figsize=(11,8),facecolor='#0d0d14')
mb=(f>=0)&(f<=3500)
aa=Aa[:,voiced].mean(1); cc=Ac_sb[:,voiced].mean(1)
aa=aa/aa[mb].max(); cc=cc/cc[mb].max()
for a in ax: a.set_facecolor('#1e1e2e'); a.tick_params(colors='#a6adc8')
ax[0].plot(f[mb], 20*np.log10(aa[mb]+1e-6), color='#89b4fa', lw=1, label='reference audio')
ax[0].plot(f[mb], 20*np.log10(cc[mb]+1e-6), color='#cba6f7', lw=1, alpha=.8, label='capture, sideband-shifted down by f_mains')
ax[0].set_title('Mean voiced spectrum: do sideband peaks land on formants?', color='#cdd6f4')
ax[0].set_xlabel('Audio freq (Hz)',color='#a6adc8'); ax[0].set_ylabel('dB',color='#a6adc8'); ax[0].legend()
# raw capture spectrum around first mains harmonics (look for sideband combs)
mb2=(f>=0)&(f<=600)
cap_raw=Ac[:,voiced].mean(1); cap_raw=cap_raw/cap_raw[mb2].max()
ax[1].plot(f[mb2], 20*np.log10(cap_raw[mb2]+1e-6), color='#fab387', lw=1)
for n in range(1,11):
    if n*fm<600: ax[1].axvline(n*fm, color='#f9e2af', ls='--', lw=.6, alpha=.5)
ax[1].set_title(f'Raw capture spectrum (voiced avg), mains harmonics @ {fm:.1f} Hz dashed', color='#cdd6f4')
ax[1].set_xlabel('Capture freq (Hz)',color='#a6adc8'); ax[1].set_ylabel('dB',color='#a6adc8')
fig.tight_layout(); fig.savefig('sideband_diagnostic.png', dpi=140, facecolor='#0d0d14')
print('saved -> sideband_diagnostic.png')
