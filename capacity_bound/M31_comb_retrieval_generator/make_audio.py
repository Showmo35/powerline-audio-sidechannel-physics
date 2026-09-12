#!/usr/bin/env python3
"""
make_audio.py — listen to it: Griffin-Lim audio from REAL vs M31-GENERATED mels.

The fair comparison is GL(real mel) vs GL(generated mel): both go through the same
vocoder, so any difference you hear is the MEL difference, not Griffin-Lim artifacts.
The true original wav is saved too as the reference.

Vocodes at the generator's NATIVE resolution (80 mels x 100 fps over the 4 s window) —
NOT the T=64 time-normalized retrieval mel, which is far too time-squashed to vocode.

Per random held-out word (same seed as viz_gen_vs_real.py):
  <w>_0_orig_4s.wav        true audio, 4 s window
  <w>_1_realGL_4s.wav      GL(real mel)        <- the GL ceiling (should be intelligible)
  <w>_2_gen_raw_4s.wav     GL(generated mel, raw arm)
  <w>_3_gen_comb_4s.wav    GL(generated mel, comb arm)
  ...and the same four cropped to the word itself (_word.wav)

Output: outputs/audio/
"""
import os, sys, argparse
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CFG, RC, OUT_DIR
import models_comb as M
import dataset_words as D
import data_io_raw as io
from train import FrontEnd

AUD = os.path.join(OUT_DIR, 'audio')
os.makedirs(AUD, exist_ok=True)

_melfn = torchaudio.transforms.MelSpectrogram(
    CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop, win_length=CFG.win_length,
    n_mels=CFG.n_mels, f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)
_fb = torchaudio.functional.melscale_fbanks(CFG.n_fft // 2 + 1, CFG.fmin, CFG.fmax,
                                            CFG.n_mels, CFG.ref_sr, norm=None, mel_scale='htk')
_pinv = torch.linalg.pinv(_fb.T)
_gl = torchaudio.transforms.GriffinLim(CFG.n_fft, n_iter=64, win_length=CFG.win_length,
                                       hop_length=CFG.hop, power=2.0)


def mel_to_wav(logmel):
    """(n_mels, frames) log-mel -> waveform via pinv(mel filterbank) + Griffin-Lim."""
    m = torch.as_tensor(logmel).float()
    spec = (_pinv @ (m.exp() - CFG.log_eps).clamp(min=0)).clamp(min=0)
    w = _gl(spec)
    return (w / (w.abs().max() + 1e-8)).numpy()


def save(name, x):
    torchaudio.save(os.path.join(AUD, name),
                    torch.from_numpy(np.asarray(x, np.float32))[None], CFG.ref_sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arms', nargs='+', default=['raw', 'comb'])
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = (dev == 'cuda')

    te_items, words = D.build_items('test')
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(te_items), args.n, replace=False)      # same seed as the figure
    items = [te_items[i] for i in pick]
    labels = [words[it[3]] for it in items]
    print(f'[audio] words: {labels}', flush=True)

    dl = DataLoader(D.WordSet(items), batch_size=len(items), shuffle=False,
                    num_workers=4, collate_fn=D.collate)
    batch = next(iter(dl))

    gens = {}
    for arm in args.arms:
        p = os.path.join(OUT_DIR, arm, 'best.pt')
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location=dev, weights_only=False)
        model = M.build(CFG, frontend=arm).to(dev)
        model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
        mean, std = ck['mel_mean'], ck['mel_std']
        front = FrontEnd(arm, dev)
        with torch.no_grad():
            cond = front(batch['raw'].to(dev), batch['f0'].to(dev))
            torch.manual_seed(0)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                g = model.sample(cond, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
            gens[arm] = (g.float() * std + mean).cpu().numpy()   # (n, 80, 400) @100fps

    for i, (ch, s, e, y) in enumerate(items):
        w = labels[i]
        # the generated window covers [s-ctx, s-ctx+win_s] at 100 fps
        t0 = s - RC.ctx_s
        frs = int(batch['frs'][i])
        # --- true original audio (4 s window + word crop) ---
        a4 = io.read_wav_window(CFG.wav_path(ch), t0, CFG.win_s, CFG.ref_sr)
        aw = io.read_wav_window(CFG.wav_path(ch), t0, (e - s) + 2 * RC.ctx_s, CFG.ref_sr)
        save(f'{w}_{i}_0_orig_4s.wav', a4)
        save(f'{w}_{i}_0_orig_word.wav', aw)
        # --- GL(real mel) : the vocoder ceiling ---
        rm4 = torch.log(_melfn(torch.from_numpy(np.asarray(a4, np.float32))) + CFG.log_eps)
        save(f'{w}_{i}_1_realGL_4s.wav', mel_to_wav(rm4))
        save(f'{w}_{i}_1_realGL_word.wav', mel_to_wav(rm4[:, :frs]))
        # --- GL(generated mel) per arm ---
        for arm in gens:
            gm = torch.from_numpy(gens[arm][i])                  # (80, 400)
            save(f'{w}_{i}_2_gen_{arm}_4s.wav', mel_to_wav(gm))
            save(f'{w}_{i}_2_gen_{arm}_word.wav', mel_to_wav(gm[:, :frs]))
        print(f'[audio] {w}: wrote orig / realGL / gen({",".join(gens)})', flush=True)

    print(f'\n[audio] -> {AUD}', flush=True)
    print('[audio] LISTEN IN THIS ORDER: _0_orig (truth) -> _1_realGL (vocoder ceiling)'
          ' -> _2_gen_* (what the powerline reconstructs)', flush=True)


if __name__ == '__main__':
    main()
