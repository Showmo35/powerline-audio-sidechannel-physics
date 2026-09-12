#!/usr/bin/env python3
"""scan_band.py — find narrowband peaks inside a frequency window and rank them."""
import argparse
import numpy as np
from scipy import signal as dsp

ap = argparse.ArgumentParser()
ap.add_argument('--capture', default='wideband_8m_8s.bin')
ap.add_argument('--samp-rate', type=float, default=7692308)
ap.add_argument('--lo', type=float, default=300e3)
ap.add_argument('--hi', type=float, default=400e3)
args = ap.parse_args()

cap = np.fromfile(args.capture, dtype=np.float32)
f, P = dsp.welch(cap - cap.mean(), fs=args.samp_rate, nperseg=1 << 16, window='blackman')
Pdb = 10 * np.log10(P + 1e-30)
win = max(11, (len(Pdb)//200) | 1)
base = dsp.medfilt(Pdb, win)
excess = Pdb - base
band = (f >= args.lo) & (f <= args.hi)
idx = np.where(band)[0]
df = f[1]-f[0]
print(f'band {args.lo/1e3:.0f}-{args.hi/1e3:.0f} kHz  bins={len(idx)}  df={df:.1f} Hz')
print(f'  PSD in band: min {Pdb[idx].min():.1f}  max {Pdb[idx].max():.1f} dB')
peaks, props = dsp.find_peaks(excess[idx], prominence=4.0, distance=max(3, len(idx)//100))
if len(peaks)==0:
    print('  NO peaks with prominence >=4 dB in this band')
else:
    order = np.argsort(props['prominences'])[::-1][:10]
    print(f'  {"freq(kHz)":>10} {"psd(dB)":>9} {"prom(dB)":>9}')
    for p in peaks[order]:
        gi = idx[p]
        print(f'  {f[gi]/1e3:>10.2f} {Pdb[gi]:>9.1f} {excess[gi]:>9.1f}')
