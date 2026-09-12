#!/usr/bin/env python3
"""
phonetic_probe.py — NOT "can we classify the two words" (envelope timing alone wins
that).  The real question: is there PHONETIC DETAIL on the carrier — i.e. does the
demodulated baseband track the true audio SPECTRAL envelope, and up to what audio
frequency?  Formants/fricatives live at 1-9 kHz; the planted cue is spectral
(which -> 1.5-4 kHz affricate, this -> 4-9 kHz sibilant), RMS-matched so loudness
carries no class info.

Two decisive, envelope-free measurements per channel:

  1. PHONETIC-BANDWIDTH CURVE.  For each audio (mel) band, correlate — ACROSS the
     60 tokens — the channel's per-token band energy with the CLEAN stimulus's
     per-token band energy.  High r in a band = that audio frequency is recovered.
     The frequency where r falls into the shuffle-null = the channel's phonetic
     bandwidth.  Needs no labels; measures spectral fidelity directly.

  2. SPECTRAL-ONLY vs ENVELOPE-ONLY class discrimination.  Split the features:
       - TEMPORAL: 1-D loudness-envelope shape (timing only, no spectrum).
       - SPECTRAL: per-frame-energy-normalized mel spectrum (spectrum only, no
         loudness/timing).  If SPECTRAL separates which/this above the permutation
         null, real phonetic detail survived; if only TEMPORAL works, the channel
         is envelope-limited and "detection" was never phonetic.
"""
import argparse, json
import numpy as np
from scipy import signal as dsp
import wave
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DARK,PANEL,GRID='#0d0d14','#1e1e2e','#313244'; TEXT,MUTED='#cdd6f4','#a6adc8'
BLUE,RED,GRN,YLW,PURP,ORNG='#89b4fa','#f38ba8','#a6e3a1','#f9e2af','#cba6f7','#fab387'

def read_wav(p):
    w=wave.open(p,'rb'); sr=w.getframerate()
    d=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float64)/32768; w.close()
    return d,sr

def power_env(cap,fs,dec=2000,block=8_000_000):
    hp=dsp.butter(2,50,'high',fs=fs,output='sos'); zih=dsp.sosfilt_zi(hp)*0
    lp=dsp.butter(4,30,'low',fs=fs,output='sos');  zil=dsp.sosfilt_zi(lp)*0
    out=[]; n0=0
    for s in range(0,len(cap),block):
        blk=cap[s:s+block].astype(np.float64)
        y,zih=dsp.sosfilt(hp,blk,zi=zih); p,zil=dsp.sosfilt(lp,y*y,zi=zil)
        out.append(p[(-n0)%dec::dec]); n0+=len(blk)
    return np.sqrt(np.clip(np.concatenate(out),0,None)),fs/dec

def demod_carrier(cap,fs,fc,half_bw=10e3,dec=80,block=4_000_000):
    sos=dsp.butter(8,half_bw,'low',fs=fs,output='sos'); zi=dsp.sosfilt_zi(sos).astype(complex)*0
    out=[]; n0=0; ppr=-2j*np.pi*fc/fs
    for s in range(0,len(cap),block):
        blk=cap[s:s+block]; n=np.arange(n0,n0+len(blk))
        y,zi=dsp.sosfilt(sos,blk*np.exp(ppr*n),zi=zi)
        out.append(y[(-n0)%dec::dec]); n0+=len(blk)
    return np.abs(np.concatenate(out)),fs/dec

def audioband(cap,fs,dec=160,block=8_000_000):
    sos=dsp.butter(8,12000,'low',fs=fs,output='sos'); zi=dsp.sosfilt_zi(sos)*0
    out=[]; n0=0
    for s in range(0,len(cap),block):
        blk=cap[s:s+block].astype(np.float64)
        y,zi=dsp.sosfilt(sos,blk,zi=zi); out.append(y[(-n0)%dec::dec]); n0+=len(blk)
    return np.abs(np.concatenate(out)),fs/dec

def mel_bank(n_fft,sr,n_mel,fmin=200,fmax=9000):
    m=lambda f:2595*np.log10(1+f/700); im=lambda x:700*(10**(x/2595)-1)
    pts=im(np.linspace(m(fmin),m(min(fmax,sr/2*0.98)),n_mel+2))
    b=np.floor((n_fft+1)*pts/sr).astype(int); fb=np.zeros((n_mel,n_fft//2+1))
    for k in range(1,n_mel+1):
        l,c,r=b[k-1],b[k],b[k+1]; c=max(c,l+1); r=max(r,c+1)
        for j in range(l,c): fb[k-1,j]=(j-l)/max(c-l,1)
        for j in range(c,r): fb[k-1,j]=(r-j)/max(r-c,1)
    cf=im((m(fmin)+m(min(fmax,sr/2*0.98)))/1)  # unused
    centers=pts[1:-1]
    return fb,centers

def token_mel(sig,fs,n_mel=20,n_t=8):
    """return (n_mel,) mean band energy(log) and (n_mel,n_t) per-frame-normalized shape."""
    nfft=512
    f,t,Z=dsp.stft(sig,fs=fs,nperseg=nfft,noverlap=nfft*3//4,window='hann')
    P=np.abs(Z)**2; fb,cen=mel_bank(nfft,fs,n_mel)
    M=fb@P
    if M.shape[1]<2: return np.zeros(n_mel),np.zeros((n_mel,n_t)),cen
    idx=np.linspace(0,M.shape[1]-1,n_t).astype(int); Mt=M[:,idx]
    band_e=np.log(M.mean(1)+1e-8)
    # spectral shape: normalize each FRAME to unit energy -> removes loudness/timing
    shape=np.log(Mt+1e-8); shape=shape-shape.mean(0,keepdims=True)
    return band_e,shape,cen

def evaluate(X,y,n_perm=400,seed=0):
    X=np.nan_to_num(X); y=np.array(y); rng=np.random.default_rng(seed)
    def cv(yy):
        skf=StratifiedKFold(5,shuffle=True,random_state=1); pr=np.zeros_like(yy)
        for tr,te in skf.split(X,yy):
            pr[te]=LDA(solver='lsqr',shrinkage='auto').fit(X[tr],yy[tr]).predict(X[te])
        return balanced_accuracy_score(yy,pr)
    acc=np.mean([cv(y) for _ in range(3)]); null=np.array([cv(rng.permutation(y)) for _ in range(n_perm)])
    return acc,(np.sum(null>=acc)+1)/(n_perm+1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--capture',default='word_capture.bin'); ap.add_argument('--samp-rate',type=float,default=4e6)
    ap.add_argument('--stim',default='word_stimulus.wav'); ap.add_argument('--labels',default='word_labels.json')
    ap.add_argument('--carriers',default='1460e3,309.4e3,880e3,1230e3')
    a=ap.parse_args(); fs=a.samp_rate
    lab=json.load(open(a.labels)); toks=lab['labels']; y=[0 if t['word']=='which' else 1 for t in toks]
    cap=np.fromfile(a.capture,dtype=np.float32)
    stim,ssr=read_wav(a.stim)

    env_cap,efs=power_env(cap,fs)
    def env_at(x,fsx,tofs):
        n=int(round(fsx/tofs)); m=len(x)//n
        return np.sqrt(np.array([np.mean(x[i*n:(i+1)*n]**2) for i in range(m)]))
    es=env_at(stim,ssr,efs)
    A=(env_cap-env_cap.mean())/(env_cap.std()+1e-9); B=(es-es.mean())/(es.std()+1e-9)
    Lm=min(len(A),len(B)); xc=np.correlate(A[:Lm],B[:Lm],'full'); lag_s=(np.argmax(xc)-(Lm-1))/efs
    print(f'[align] lag={lag_s:.3f}s env_r={xc.max()/Lm:.3f}   tokens={len(toks)}')

    win_pre,win_len=0.05,0.60
    def windows(sig,sfs):
        out=[]
        for t in toks:
            on=t['onset_s']+lag_s
            i0=max(0,int((on-win_pre)*sfs)); i1=min(len(sig),int((on-win_pre+win_len)*sfs))
            out.append(sig[i0:i1])
        return out

    channels={'audioband<12k':audioband(cap,fs)}
    for fc in [float(eval(c)) for c in a.carriers.split(',')]:
        channels[f'{fc/1e3:.0f}k']=demod_carrier(cap,fs,fc)

    # stimulus reference per-token band energies
    stim_be=[]; cen=None
    for t in toks:
        s=stim[int(t['onset_s']*ssr):int((t['onset_s']+win_len)*ssr)]
        be,_,cen=token_mel(s,ssr); stim_be.append(be)
    stim_be=np.array(stim_be)  # (ntok,nmel)

    print(f'\n{"channel":<14} {"phon_BW":>8} {"specAcc":>8} {"specP":>7} {"tempAcc":>8} {"tempP":>7}  verdict')
    print('-'*74)
    curves={}; rng=np.random.default_rng(3)
    for name,(sig,sfs) in channels.items():
        segs=windows(sig,sfs)
        BE=[]; SH=[]; TE=[]
        for s in segs:
            be,sh,_=token_mel(s,sfs); BE.append(be); SH.append(sh.ravel())
            # temporal envelope shape (loudness over time)
            if len(s)>8:
                ev=np.interp(np.linspace(0,len(s)-1,24),np.arange(len(s)),np.abs(s))
                TE.append((ev-ev.mean())/(ev.std()+1e-9))
            else: TE.append(np.zeros(24))
        BE=np.array(BE); SH=np.array(SH); TE=np.array(TE)
        # per-band correlation demod vs stimulus, across tokens
        rband=np.array([np.corrcoef(BE[:,k],stim_be[:,k])[0,1] for k in range(BE.shape[1])])
        rband=np.nan_to_num(rband)
        # shuffle null for band corr
        null=np.array([[np.corrcoef(BE[rng.permutation(len(BE)),k],stim_be[:,k])[0,1]
                        for k in range(BE.shape[1])] for _ in range(200)])
        thr=np.nanpercentile(null,95,axis=0)
        # phonetic bandwidth = highest band center where r>thr and r>0.2 (contiguous from low)
        good=(rband>thr)&(rband>0.2)
        phonBW=cen[np.where(good)[0].max()] if good.any() else 0.0
        curves[name]=(cen,rband,thr)
        specAcc,specP=evaluate(SH,y); tempAcc,tempP=evaluate(TE,y)
        v='PHONETIC' if (specP<0.05 and specAcc>0.6) else ('envelope-only' if tempP<0.05 else 'none')
        print(f'{name:<14} {phonBW:>7.0f}H {specAcc:>8.3f} {specP:>7.3f} {tempAcc:>8.3f} {tempP:>7.3f}  {v}')

    # verdict summary
    print('\n== PHONETIC-DETAIL VERDICT '+'='*40)
    print('  phon_BW = top audio freq the channel tracks across tokens (band r>null & >0.2)')
    print('  specAcc = which/this from SPECTRAL SHAPE only (loudness/timing removed)')
    print('  tempAcc = which/this from loudness-envelope TIMING only (no spectrum)')
    print('  PHONETIC only if spectral-shape beats chance; else channel is envelope-limited.')
    make_fig(curves,y,channels)

def make_fig(curves,y,channels):
    fig,ax=plt.subplots(figsize=(11,6.5),facecolor=DARK); ax.set_facecolor(PANEL)
    cols=[BLUE,GRN,YLW,PURP,ORNG,RED]
    for i,(name,(cen,rband,thr)) in enumerate(curves.items()):
        ax.plot(cen,rband,'o-',color=cols[i%len(cols)],lw=1.6,ms=4,label=name)
    ax.plot(cen,thr,'--',color=MUTED,lw=1,label='shuffle null (95%)')
    ax.axhspan(-0.3,0.2,color=RED,alpha=.06)
    ax.axvspan(1500,4000,color=YLW,alpha=.05); ax.axvspan(4000,9000,color=GRN,alpha=.05)
    ax.text(2500,0.9,'/tʃ/ "which"',color=YLW,fontsize=8,ha='center')
    ax.text(6000,0.9,'/s/ "this"',color=GRN,fontsize=8,ha='center')
    ax.set_xlabel('audio frequency (Hz)  — formants/fricatives live here',color=MUTED)
    ax.set_ylabel('per-band corr(demod, true audio) across tokens',color=MUTED)
    ax.set_title('Phonetic-bandwidth curve: how much audio SPECTRUM survives on each carrier',
                 color=TEXT,fontsize=11,fontweight='bold',loc='left')
    ax.set_ylim(-0.3,1.0); ax.tick_params(colors=MUTED); [s.set_edgecolor(GRID) for s in ax.spines.values()]
    ax.grid(True,color=GRID,lw=.5,alpha=.5); ax.legend(facecolor=PANEL,edgecolor=GRID,labelcolor=TEXT,fontsize=8,ncol=2)
    fig.savefig('phonetic_probe.png',dpi=140,facecolor=DARK,bbox_inches='tight'); print('[viz] phonetic_probe.png')

if __name__=='__main__':
    main()
