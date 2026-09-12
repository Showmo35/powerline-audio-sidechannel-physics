#!/usr/bin/env python3
"""
make_viz.py — side-by-side real vs PLF-generated mels + Griffin-Lim audio for a
few held-out utterances (generated mels read straight from gen_mels/*.npy).
Writes viz/{utt}_mels.png, {utt}_real.wav, {utt}_glreal.wav, {utt}_glgen.wav, meta.json.
"""
import json, os, wave
import numpy as np
import torch, torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import CFG, MANIFEST, GEN_DIR, MODULE_DIR
import dataset as D

VIZ = os.path.join(MODULE_DIR, 'viz')
os.makedirs(VIZ, exist_ok=True)
PAGE, PANEL, INK, MUTED = '#101318', '#171B22', '#E9E7E2', '#9AA0AB'


def gl_invert(logmel, n_iter=64):
    powmel = (logmel.exp() - CFG.log_eps).clamp(min=0)
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=CFG.n_fft // 2 + 1, f_min=CFG.fmin, f_max=CFG.fmax,
        n_mels=CFG.n_mels, sample_rate=CFG.ref_sr, norm=None, mel_scale='htk')
    spec = (torch.linalg.pinv(fb.T) @ powmel).clamp(min=0)
    gl = torchaudio.transforms.GriffinLim(n_fft=CFG.n_fft, n_iter=n_iter,
                                          win_length=CFG.win_length,
                                          hop_length=CFG.hop, power=2.0)
    w = gl(spec)
    return (w / (w.abs().max() + 1e-8)).numpy()


def save_wav(path, x, sr=16000):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    rows = D.load_rows()
    test = CFG.test_chunks()
    cands = [r for r in rows if r['chunk'] in test and 5.0 <= r['dur_s'] <= 9.0
             and os.path.exists(os.path.join(GEN_DIR, r['utt_id'] + '.npy'))]
    rng = np.random.RandomState(7)
    by_chunk = {}
    for r in cands:
        by_chunk.setdefault(r['chunk'], []).append(r)
    chunks = rng.choice(sorted(by_chunk), size=3, replace=False)
    picks = [by_chunk[c][rng.randint(len(by_chunk[c]))] for c in chunks]

    ds = D.MelText(picks, 'real')
    meta = []
    for i, r in enumerate(picks):
        gen = torch.from_numpy(np.load(os.path.join(GEN_DIR, r['utt_id'] + '.npy')).astype(np.float32))
        real = ds._mel(r)
        T = min(gen.shape[-1], real.shape[-1])
        gen, real = gen[:, :T], real[:, :T]
        mel_r = float(np.corrcoef(gen.numpy().ravel(), real.numpy().ravel())[0, 1])
        dur = T / CFG.fps

        tag = r['utt_id']
        wav_real = D.read_wav_window(CFG.wav_path(r['chunk']), r['start_s'], dur, CFG.ref_sr)
        save_wav(os.path.join(VIZ, f'{tag}_real.wav'), wav_real)
        save_wav(os.path.join(VIZ, f'{tag}_glreal.wav'), gl_invert(real))
        save_wav(os.path.join(VIZ, f'{tag}_glgen.wav'), gl_invert(gen))

        fig, axes = plt.subplots(2, 1, figsize=(10.2, 4.6), dpi=135, sharex=True)
        fig.patch.set_facecolor(PAGE)
        vmin, vmax = real.min().item(), real.max().item()
        for ax, m, title in ((axes[0], real, 'real mel (reference audio)'),
                             (axes[1], gen, 'PLF-generated mel (from powerline capture)')):
            ax.imshow(m.numpy(), origin='lower', aspect='auto', cmap='magma',
                      vmin=vmin, vmax=vmax, extent=[0, dur, 0, CFG.n_mels],
                      interpolation='nearest')
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=INK, fontsize=9.5, loc='left', pad=4)
            ax.set_ylabel('mel bin', color=MUTED, fontsize=8)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for s in ax.spines.values():
                s.set_color(MUTED); s.set_linewidth(0.4)
        axes[1].set_xlabel('seconds', color=MUTED, fontsize=8)
        fig.tight_layout(pad=1.0)
        fig.savefig(os.path.join(VIZ, f'{tag}_mels.png'),
                    facecolor=PAGE, bbox_inches='tight')
        plt.close(fig)

        meta.append({'utt_id': tag, 'chunk': r['chunk'], 'dur_s': round(dur, 2),
                     'mel_r': round(mel_r, 3), 'text': r['text'].lower()})
        print(f'[viz] {tag} ({r["chunk"]}, {dur:.1f}s, mel_r={mel_r:.3f})')

    json.dump(meta, open(os.path.join(VIZ, 'meta.json'), 'w'), indent=1)
    print('[viz] wrote', VIZ)


if __name__ == '__main__':
    main()
