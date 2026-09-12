#!/usr/bin/env python3
"""
words_all_harm.py — FULL-SCALE word detection from ALL mains harmonics (GPU).

The question: with every harmonic demodulated coherently on the CORRECT 60.00 Hz grid
(amplitude + phase), can we detect words better than the loudness/prosody control?

This is the decisive scale-up of M28's which/this pilot, and it is directly comparable
to M20 (same 30-word vocab, same fixed 1.5 s window centred on the word midpoint, same
chunk split, same macro-accuracy metric):

    M20 wide (200 kHz STFT, 48.8 Hz bins) : 31.1%   <- could not resolve the 60 Hz comb
    M20 env  (spectrum flattened)         : 39.0%   <- the loudness control WON
    chance                                : 3.3%
    audio ceiling (wav2vec2)              : 65.0%

M20's conclusion "harmonics add nothing over the envelope" was drawn from a WIDEBAND
STFT, never from per-harmonic coherent demodulation on a correct grid. That is what
this measures.

Features per word: (2, K, T) = [AM, PM] x all K harmonics x T frames.
Control:           envelope-only (the AM gram collapsed over harmonics) -> same model.

Stage 1 caches grams to outputs/cache/. Stage 2 trains a small CNN.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

PROOT = '<REPO_ROOT>'
sys.path.insert(0, os.path.join(PROOT, 'M15_soundbar_melgen_word'))
import data_io as io
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comb import estimate_mains
from comb_gpu import harmonic_gram_gpu, gram_features_gpu, CAP_SR

INDEX = os.path.join(PROOT, 'M20_powerline_word_classifier', 'word_index.json')
BIN = os.path.join(PROOT, 'Powerline_Data_Captures', 'soundbar_bin_captures')
CACHE = os.path.join(HERE, 'outputs', 'cache')
os.makedirs(CACHE, exist_ok=True)

WIN_S = 1.5           # M20 framing: fixed window centred on the word midpoint
PAD_FRAC = 0.2        # tukey taper fraction each side -> flat centre = central 60%
VOCAB_K = 30
MIN_OCC = 20
MAX_PER_WORD = 400
TEST_EVERY = 12


def is_test(ch):
    return int(ch.split('_')[1]) % TEST_EVERY == 0


def build_items():
    occ = json.load(open(INDEX))
    cand = [w for w in occ if len(occ[w]) >= MIN_OCC]
    words = sorted(cand, key=lambda w: -len(occ[w]))[:VOCAB_K]
    wid = {w: i for i, w in enumerate(words)}
    rng = np.random.RandomState(0)
    tr, te = [], []
    for w in words:
        rows = [r for r in occ[w]]
        rng.shuffle(rows)
        a = [r for r in rows if not is_test(r[0])][:MAX_PER_WORD]
        b = [r for r in rows if is_test(r[0])]
        tr += [(c, s, e, wid[w]) for c, s, e in a]
        te += [(c, s, e, wid[w]) for c, s, e in b]
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te, words


def cache_split(items, tag, K, T, bs, dev):
    """Demodulate ALL harmonics for every occurrence; cache (N,2,K,T) fp16."""
    path = os.path.join(CACHE, f'{tag}_K{K}_T{T}.npy')
    ypath = os.path.join(CACHE, f'{tag}_y.npy')
    if os.path.exists(path) and os.path.exists(ypath):
        print(f'[cache] {tag}: reuse {path}', flush=True)
        return np.load(path, mmap_mode='r'), np.load(ypath)
    pad = WIN_S * PAD_FRAC / (1 - 2 * PAD_FRAC)      # pad so flat centre == WIN_S
    tot = WIN_S + 2 * pad
    N = int(round(tot * CAP_SR))
    f0c = {}
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16,
                                    shape=(len(items), 2, K, T))
    ys = np.zeros(len(items), np.int64)
    t0 = time.time(); buf, meta, n = [], [], 0
    for i, (ch, s, e, y) in enumerate(items):
        if ch not in f0c:
            f0c[ch] = estimate_mains(io.read_bin_window(
                os.path.join(BIN, f'{ch}.bin'), 60.0, 30.0, CAP_SR), CAP_SR)
        lag = io.read_lag_ms(os.path.join(BIN, f'{ch}.lag')) / 1000.0
        mid = 0.5 * (s + e)
        x = io.read_bin_window(os.path.join(BIN, f'{ch}.bin'),
                               mid - tot / 2 + lag, tot, CAP_SR)
        if len(x) < N:
            x = np.pad(x, (0, N - len(x)))
        x = x[:N] / (np.std(x[:N]) + 1e-8)
        buf.append(x); meta.append((i, f0c[ch], y))
        if len(buf) == bs or i == len(items) - 1:
            xb = torch.from_numpy(np.stack(buf)).float().to(dev)
            fb = torch.tensor([m[1] for m in meta], device=dev)
            H = harmonic_gram_gpu(xb, fb, CAP_SR, K=K, T=T, pad_frac=PAD_FRAC)
            G = gram_features_gpu(H).cpu().numpy().astype(np.float16)
            for j, (ii, _, yy) in enumerate(meta):
                out[ii] = G[j]; ys[ii] = yy
            n += len(buf); buf, meta = [], []
            if n % (bs * 20) < bs:
                print(f'[cache:{tag}] {n}/{len(items)}  {time.time()-t0:.0f}s', flush=True)
    out.flush(); np.save(ypath, ys)
    print(f'[cache:{tag}] done {len(items)} in {time.time()-t0:.0f}s -> {path}', flush=True)
    return np.load(path, mmap_mode='r'), ys


class CombNet(nn.Module):
    """Small CNN over the (2, K, T) harmonic-gram. Strides hard over the harmonic axis
    (K~1600) the way M20's WideWordNet strides over frequency."""
    def __init__(self, n_cls, k_in=2):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(k_in, 32, (7, 3), stride=(4, 1), padding=(3, 1)), nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (7, 3), stride=(4, 1), padding=(3, 1)), nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, (7, 3), stride=(4, 1), padding=(3, 1)), nn.GELU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, (5, 3), stride=(2, 1), padding=(2, 1)), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                               nn.Linear(128 * 16, 256), nn.GELU(), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.h(self.f(x))


def macro_acc(yt, yp, K):
    accs = []
    for c in range(K):
        m = yt == c
        if m.sum():
            accs.append((yp[m] == c).mean())
    return float(np.mean(accs))


def run(Xtr, ytr, Xte, yte, nwords, dev, tag, epochs, k_in, bs=64):
    torch.manual_seed(0)
    model = CombNet(nwords, k_in=k_in).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 3e-3, total_steps=epochs * max(1, len(Xtr) // bs))
    best = 0.0
    for ep in range(epochs):
        model.train(); idx = np.random.permutation(len(Xtr))
        for b in range(0, len(idx) - bs + 1, bs):
            j = np.sort(idx[b:b + bs])
            xb = torch.from_numpy(np.asarray(Xtr[j], np.float32)).to(dev)
            yb = torch.from_numpy(ytr[j]).to(dev)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        model.eval(); preds = []
        with torch.no_grad():
            for b in range(0, len(Xte), bs):
                xb = torch.from_numpy(np.asarray(Xte[b:b + bs], np.float32)).to(dev)
                preds.append(model(xb).argmax(1).cpu().numpy())
        p = np.concatenate(preds)
        m = macro_acc(yte, p, nwords)
        best = max(best, m)
        print(f'  [{tag}] ep{ep+1}/{epochs} macro={m:.4f} best={best:.4f} '
              f'(overall={float((p==yte).mean()):.4f})', flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=1600, help='ALL harmonics to Nyquist')
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=30)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[m28-words] device={dev} K={args.K} (all harmonics to {args.K*60/1000:.0f} kHz) '
          f'T={args.T}', flush=True)

    tr, te, words = build_items()
    print(f'[m28-words] vocab={len(words)} train={len(tr)} test={len(te)}', flush=True)
    print(f'[m28-words] words={words}', flush=True)

    Xtr, ytr = cache_split(tr, 'train', args.K, args.T, args.bs, dev)
    Xte, yte = cache_split(te, 'test', args.K, args.T, args.bs, dev)

    res = {}
    print('\n── FULL harmonic-gram (AM+PM, all harmonics) ──', flush=True)
    res['all_harm_AM_PM'] = run(Xtr, ytr, Xte, yte, len(words), dev, 'AM+PM',
                                args.epochs, k_in=2)

    print('\n── ENVELOPE control (AM collapsed over harmonics) ──', flush=True)
    # collapse the harmonic axis -> pure loudness contour (the prosody control)
    def env_of(X):
        out = np.zeros((len(X), 1, 1, X.shape[-1]), np.float32)
        for j in range(len(X)):
            out[j, 0, 0] = np.asarray(X[j][0], np.float32).mean(0)
        return out
    Etr, Ete = env_of(Xtr), env_of(Xte)
    res['envelope_control'] = run(Etr, ytr, Ete, yte, len(words), dev, 'env',
                                  args.epochs, k_in=1)

    print('\n══ M28 FULL-SCALE WORD DETECTION (all harmonics) ══════════════')
    print(f'  all-harmonic AM+PM : {res["all_harm_AM_PM"]:.4f}')
    print(f'  envelope control   : {res["envelope_control"]:.4f}')
    print(f'  refs: M20 wide=0.311  M20 env=0.390  chance=0.033  audio(wav2vec2)=0.650')
    print('════════════════════════════════════════════════════════════════')
    json.dump({'K': args.K, 'T': args.T, 'n_train': len(tr), 'n_test': len(te),
               'words': words, 'results': res,
               'refs': {'M20_wide': 0.311, 'M20_env': 0.390, 'chance': 0.033,
                        'audio_wav2vec2': 0.650}},
              open(os.path.join(HERE, 'outputs', 'm28_words_all_harm.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
