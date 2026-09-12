#!/usr/bin/env python3
"""
viz_wrong_grid.py — WORD-LEVEL M14 analog of M22's "wrong grid".

Same figure as M22's m22_wrong_grid.png, but the generator is M14 (PowerLine-Flow)
instead of the M22 retrieval-trained generator.  The retrieval space is identical
and model-independent: real word log-mels, mean-centered + L2-normalized, so cosine
== Pearson r (exactly M22's `embed`).  The ONLY thing that changes is who produced
the generated mel.

For a sample of held-out WORD occurrences we:
  * read the 4 s powerline window at (word_onset - ctx) + lag  (M14 generator input),
  * sample the full-window mel with M14, crop the first `frs` frames (the word sits
    at the window start) and time-normalize to (n_mels, T) — the generated word mel,
  * retrieve the nearest real word in a gallery of train words (kNN, Pearson r),
  * keep the cases where the nearest gallery word is the WRONG word.

We render three mels per row:
  [ correct word (real) ]  [ M14 generated ]  [ retrieved WRONG word (real) ]
labelled with r(gen, correct) and r(gen, impostor).

Words come from  M20_powerline_word_classifier/word_index.json  (same source M22 uses).
Output -> outputs/m14_wrong_grid.png  (run on GPU; see run_wrong.slurm)
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import CFG, PROJECT_ROOT
import data_io as io
import models as M

# ── word-retrieval constants (identical to M22's RC) ───────────────────────────
WORD_INDEX = os.path.join(PROJECT_ROOT, 'M20_powerline_word_classifier', 'word_index.json')
CTX_S   = 0.10           # seconds of context each side of a word
T_FR    = 64             # time-normalized frames per word (kNN space)
MIN_DUR = 0.08
MAX_DUR = 2.00
GALLERY_PER_WORD = 50
EVAL_STEPS = 32
CFG_SCALE  = 2.0

_TEST = CFG.test_chunks()
def is_test(ch):
    return ch in _TEST


def build_vocab():
    occ = json.load(open(WORD_INDEX))
    words = sorted(occ.keys())
    return words, occ


def build_items(split, words, occ):
    """List of (chunk, s, e, label) over ALL words for split ('train'|'test')."""
    wid = {w: i for i, w in enumerate(words)}
    rng = np.random.RandomState(0)
    items = []
    for w in words:
        rows = [r for r in occ[w]
                if is_test(r[0]) == (split == 'test')
                and MIN_DUR <= (r[2] - r[1]) <= MAX_DUR]
        for ch, s, e in rows:
            items.append((ch, float(s), float(e), wid[w]))
    rng.shuffle(items)
    return items


def cap_per_label(items, k, seed=0):
    if k is None or k <= 0:
        return items
    rng = np.random.RandomState(seed)
    by = {}
    for it in items:
        by.setdefault(it[3], []).append(it)
    out = []
    for y, rows in by.items():
        if len(rows) > k:
            idx = rng.choice(len(rows), size=k, replace=False)
            rows = [rows[i] for i in idx]
        out += rows
    return out


_MELSPEC = torchaudio.transforms.MelSpectrogram(
    sample_rate=CFG.ref_sr, n_fft=CFG.n_fft, hop_length=CFG.hop,
    win_length=CFG.win_length, n_mels=CFG.n_mels,
    f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)


def logmel(wav):
    m = _MELSPEC(torch.from_numpy(np.asarray(wav, np.float32)))
    return torch.log(m + CFG.log_eps)


def word_mel(ch, s, e):
    """REAL log-mel of [s-ctx, e+ctx] -> (n_mels, T)."""
    wdur = (e - s) + 2 * CTX_S
    ww = io.read_wav_window(CFG.wav_path(ch), s - CTX_S, wdur, CFG.ref_sr)
    mw = logmel(ww)
    return F.interpolate(mw[None, None], size=(CFG.n_mels, T_FR),
                         mode='bilinear', align_corners=False)[0, 0], wdur


def embed(mel_bt):
    """(B, n_mels, T) -> mean-centered, L2-normalized.  cosine == Pearson r."""
    z = mel_bt.reshape(mel_bt.shape[0], -1)
    z = z - z.mean(dim=1, keepdim=True)
    return F.normalize(z, dim=1)


def gen_word_mel(gen_full, frs):
    """gen_full (B, n_mels, n_frames) -> (B, n_mels, T): slice [:, :frs] then time-norm."""
    outs = []
    for i in range(gen_full.shape[0]):
        w = gen_full[i:i + 1, :, :int(frs[i])]
        w = F.interpolate(w[None], size=(CFG.n_mels, T_FR),
                          mode='bilinear', align_corners=False)[0]
        outs.append(w)
    return torch.cat(outs, dim=0)


class GallerySet(Dataset):
    """Real word mels only (cheap gallery path)."""
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        ch, s, e, y = self.items[i]
        mw, _ = word_mel(ch, s, e)
        return {'mel_word': mw, 'label': y}


class QuerySet(Dataset):
    """M14 generator input (4 s powerline @ onset-ctx) + word frame count + label."""
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        ch, s, e, y = self.items[i]
        lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0
        raw = io.read_bin_window(CFG.bin_path(ch), (s - CTX_S) + lag, CFG.win_s,
                                 CFG.cap_sr, CFG.in_sr)
        raw = np.asarray(raw, np.float32)
        raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
        x = np.zeros(CFG.in_len, np.float32)
        x[:min(len(raw), CFG.in_len)] = raw[:CFG.in_len]
        wdur = (e - s) + 2 * CTX_S
        frs = max(6, int(round(wdur * CFG.fps)))
        return {'raw': torch.from_numpy(x), 'frs': frs, 'label': y}


def collate_q(b):
    return {'raw': torch.stack([x['raw'] for x in b]),
            'frs': torch.tensor([x['frs'] for x in b]),
            'label': torch.tensor([x['label'] for x in b])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', type=int, default=800)
    ap.add_argument('--gcap', type=int, default=GALLERY_PER_WORD)
    ap.add_argument('--n', type=int, default=6, help='failure examples to show')
    ap.add_argument('--ckpt', default='outputs/cluster_a100/best.pt')
    ap.add_argument('--tag', default='M14')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default='outputs/m14_wrong_grid.png')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    use_amp = (dev == 'cuda')

    words, occ = build_vocab()
    model = M.build(CFG).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
    mean, std = ck['mel_mean'], ck['mel_std']
    print(f'[wrong] loaded {args.ckpt} step={ck.get("step")} metrics={ck.get("metrics")}',
          flush=True)

    # gallery: real train word mels (capped/word)
    tr_items = build_items('train', words, occ)
    gal_items = cap_per_label(tr_items, args.gcap)
    gal_dl = DataLoader(GallerySet(gal_items), batch_size=256, shuffle=False, num_workers=8,
                        collate_fn=lambda bb: {'mel_word': torch.stack([x['mel_word'] for x in bb]),
                                               'label': torch.tensor([x['label'] for x in bb])})
    print('[wrong] building gallery …', flush=True)
    embs, ys = [], []
    with torch.no_grad():
        for b in gal_dl:
            embs.append(embed(b['mel_word'].to(dev)).half())
            ys.append(b['label'].to(dev))
    G = torch.cat(embs); gy = torch.cat(ys)
    gallery_words = set(gy.tolist())
    print(f'[wrong] gallery={G.shape[0]} over {len(gallery_words)} words', flush=True)

    # queries: held-out words that exist in the gallery vocab
    te_items = build_items('test', words, occ)
    te_items = [it for it in te_items if it[3] in gallery_words]
    rng = np.random.RandomState(0)
    if len(te_items) > args.queries:
        te_items = [te_items[i] for i in rng.choice(len(te_items), args.queries, replace=False)]
    te_dl = DataLoader(QuerySet(te_items), batch_size=args.batch, shuffle=False,
                       num_workers=8, collate_fn=collate_q)

    fails = []
    qptr = 0
    with torch.no_grad():
        for b in te_dl:
            raw = b['raw'].to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                gen = model.sample(raw, steps=EVAL_STEPS, cfg_scale=CFG_SCALE)
            gen = gen.float() * std + mean
            gw = gen_word_mel(gen, b['frs'])                      # (b,n_mels,T)
            q = embed(gw).half()
            sims = q @ G.t()                                     # (b,Ng)
            for i in range(len(gw)):
                t = int(b['label'][i])
                same = (gy == t)
                top1 = int(sims[i].argmax())
                pred = int(gy[top1])
                if pred == t:
                    continue                                     # correct -> skip
                r_imp = float(sims[i][top1])
                cidx = int(torch.where(same)[0][sims[i][same].argmax()])
                r_cor = float(sims[i][cidx])
                fails.append({
                    'true': words[t], 'pred': words[pred], 'r_cor': r_cor, 'r_imp': r_imp,
                    'gen': gw[i].cpu().numpy(),
                    'q_item': te_items[qptr + i],                 # correct word occurrence
                    'imp_item': gal_items[top1],                  # matched wrong occurrence
                })
            qptr += len(gw)

    print(f'[wrong] {len(fails)}/{len(te_items)} wrong retrievals collected', flush=True)
    # most confident, varied confusions: high impostor corr, dedupe by true word
    fails.sort(key=lambda d: -d['r_imp'])
    chosen, used = [], set()
    for d in fails:
        if d['true'] in used:
            continue
        used.add(d['true']); chosen.append(d)
        if len(chosen) == args.n:
            break

    def real_of(item):
        return word_mel(item[0], item[1], item[2])[0].numpy()

    # ── render: n rows x 3 cols [correct | generated | retrieved-wrong] ────────
    n = len(chosen)
    fig = plt.figure(figsize=(8.4, 1.15 * n + 1.1))
    gs = GridSpec(n, 3, figure=fig, hspace=0.55, wspace=0.08,
                  left=0.02, right=0.99, top=0.88, bottom=0.03)
    fig.suptitle(f'{args.tag}: generated mels that correlate with the WRONG word\n'
                 f'(held-out words · generated mel matches an impostor better than its own)',
                 fontsize=10, y=0.985, va='top')
    col_titles = ['correct word (real)', f'{args.tag} generated', 'retrieved WRONG word (real)']
    for row, d in enumerate(chosen):
        real_c = real_of(d['q_item']); real_i = real_of(d['imp_item']); gen_m = d['gen']
        vmin = min(real_c.min(), gen_m.min(), real_i.min())
        vmax = max(real_c.max(), gen_m.max(), real_i.max())
        panels = [(real_c, f'“{d["true"]}”'),
                  (gen_m,  f'r={d["r_cor"]:.2f} to correct\nr={d["r_imp"]:.2f} to wrong'),
                  (real_i, f'“{d["pred"]}”')]
        for c, (m, sub) in enumerate(panels):
            ax = fig.add_subplot(gs[row, c])
            ax.imshow(m, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[c], fontsize=8.5, pad=3)
            ax.set_xlabel(sub, fontsize=8,
                          color=('#b2182b' if c == 1 else 'black'))
        fig.axes[-1].patch.set_edgecolor('#b2182b')
        for sp in fig.axes[-1].spines.values():
            sp.set_color('#b2182b'); sp.set_linewidth(1.6)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print('[wrong] wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
