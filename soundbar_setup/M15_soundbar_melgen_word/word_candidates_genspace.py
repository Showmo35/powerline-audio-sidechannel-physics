#!/usr/bin/env python3
"""
word_candidates_genspace.py — test the hypothesis that M15 generates into its own
"encoded space", so a TEST word should match TRAIN word mels *generated the same way*
(gen<->gen), not real mels (gen<->real).

For one sentence's words: rank the M15-generated test word against, on the SAME bounded
vocabulary, TWO galleries of centroids:
  REAL  = centroid of each word's REAL train mels          (the old word_candidates.png)
  GEN   = centroid of each word's M15-GENERATED train mels (gen<->gen, the hypothesis)
Shows top-10 + true-word rank under each. If GEN ranks the true word far better than
REAL, the failure was the cross-modal gap, not the generator.

Bounded vocab (generating gen-centroids for all 25k words is too expensive): a sample of
words with >=6 train occ, plus the sentence words and the wife-candidate set forced in.
Utterance generations are cached (many words share an utterance).
"""
import os, json, numpy as np, torch
import torch.nn.functional as F
import torchaudio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from config import CFG
import models as M
import data_io as io

PROOT = '<REPO_ROOT>'
WIDX = os.path.join(PROOT, 'M20_powerline_word_classifier', 'word_index.json')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CTX = 0.10; T = 64
VOCAB_N = 800; K_REAL = 8; K_GEN = 5
FORCE = ['wife', 'master', 'hounds', 'beg', 'platte', 'back', 'dock', 'dive',
         'drug', 'bought', 'bet', 'pack', 'debt', 'sounds', 'minister']

ck = torch.load('outputs/run1/last.pt', map_location=dev, weights_only=False)
model = M.build(CFG).to(dev); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
melfn = torchaudio.transforms.MelSpectrogram(CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop,
                                             win_length=CFG.win_length, n_mels=CFG.n_mels,
                                             f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)
occ = json.load(open(WIDX))
man = json.load(open('full_manifest.json'))
by_chunk = {}
for r in man:
    by_chunk.setdefault(r['chunk'], []).append(r)
def is_test(ch): return int(ch.split('_')[1]) % 12 == 0

def find_utt(ch, ws, we):
    best = None
    for r in by_chunk.get(ch, []):
        if r['start_s'] <= ws and r['end_s'] >= we and 2.0 <= r['dur_s'] <= CFG.max_dur_s:
            if best is None or r['dur_s'] < best['dur_s']:
                best = r
    return best

def norm64(m): return F.interpolate(torch.as_tensor(m)[None, None].float(), size=(80, T),
                                    mode='bilinear', align_corners=False)[0, 0]
def embed(m64):
    z = m64.reshape(-1); z = z - z.mean(); return z / (z.norm() + 1e-8)

def real_mel64(ch, s, e):
    w = io.read_wav_window(CFG.wav_path(ch), s - CTX, (e - s) + 2 * CTX, CFG.ref_sr).astype(np.float32)
    return norm64(torch.log(melfn(torch.from_numpy(w)) + CFG.log_eps))

_UC = {}
@torch.no_grad()
def gen_utt(u):
    k = (u['chunk'], round(u['start_s'], 3))
    if k not in _UC:
        lag = io.read_lag_ms(CFG.lag_path(u['chunk'])) / 1000.0
        dur = min(u['dur_s'], CFG.max_dur_s)
        raw = io.read_bin_window(CFG.bin_path(u['chunk']), u['start_s'] + lag, dur, CFG.cap_sr).astype(np.float32)
        raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
        x = np.zeros(CFG.a_len, np.float32); x[:min(len(raw), CFG.a_len)] = raw[:CFG.a_len]
        _UC[k] = (model.sample(torch.from_numpy(x)[None].to(dev))[0].float().cpu() * ms + mm)
    return _UC[k]

def gen_mel64(ch, s, e):
    u = find_utt(ch, s, e)
    if u is None: return None
    g = gen_utt(u)
    f0 = max(0, int((s - CTX - u['start_s']) * CFG.fps)); f1 = min(g.shape[1], int((e + CTX - u['start_s']) * CFG.fps))
    if f1 - f0 < 3: f1 = min(g.shape[1], f0 + 3)
    return norm64(g[:, f0:f1])

# ── vocab ──
rng = np.random.RandomState(0)
cand = [w for w in occ if sum(not is_test(c) for c, s, e in occ[w]) >= 6]
rng.shuffle(cand)
vocab = list(dict.fromkeys([w for w in FORCE if w in occ] + cand[:VOCAB_N]))
print(f'[vocab] {len(vocab)} words (forced {sum(w in FORCE for w in vocab)})', flush=True)

# ── build REAL + GEN centroids ──
Creal, Cgen, cw = [], [], []
t0 = __import__('time').time()
for i, w in enumerate(vocab):
    tr = [(c, s, e) for (c, s, e) in occ[w] if not is_test(c)]
    rmels = [real_mel64(*x) for x in tr[:K_REAL]]
    gmels = []
    for x in tr:
        g = gen_mel64(*x)
        if g is not None: gmels.append(g)
        if len(gmels) >= K_GEN: break
    if not rmels or not gmels:
        continue
    cr = torch.stack([embed(m) for m in rmels]).mean(0); cr = cr / (cr.norm() + 1e-8)
    cg = torch.stack([embed(m) for m in gmels]).mean(0); cg = cg / (cg.norm() + 1e-8)
    Creal.append(cr); Cgen.append(cg); cw.append(w)
    if i % 100 == 0:
        print(f'  {i}/{len(vocab)}  utt-cache={len(_UC)}  {__import__("time").time()-t0:.0f}s', flush=True)
Creal = torch.stack(Creal).to(dev); Cgen = torch.stack(Cgen).to(dev); cw = np.array(cw)
print(f'[centroids] {len(cw)} words (real+gen), utt gens={len(_UC)}', flush=True)

# ── test sentence words (chunk_084) ──
test = [r for r in man if is_test(r['chunk']) and 2.0 <= r['dur_s'] <= 3.8 and len(r['text'].split()) >= 4]
rng2 = np.random.RandomState(1); rng2.shuffle(test); utt = test[2]
ch, us = utt['chunk'], utt['start_s']
seg = sorted([(float(s), float(e), w) for w in occ for (c, s, e) in occ[w]
              if c == ch and s >= us - 0.05 and e <= utt['end_s'] + 0.05], key=lambda t: t[0])

def top10_rank(q, C, w):
    sims = (q @ C.t()).float().cpu().numpy(); order = np.argsort(-sims)
    top = [(cw[j], float(sims[j])) for j in order[:10]]
    rank = int(np.where(cw[order] == w)[0][0]) + 1 if w in cw else -1
    return top, rank

rows = []; gmels = []
print(f'\nSENTENCE [{ch}]: "{utt["text"]}"', flush=True)
for (s, e, w) in seg:
    gm = gen_mel64(ch, s, e)
    if gm is None: continue
    gmels.append(gm.numpy())
    q = embed(gm).to(dev)
    tR, rR = top10_rank(q, Creal, w)
    tG, rG = top10_rank(q, Cgen, w)
    rows.append({'word': w, 'rankReal': rR, 'rankGen': rG, 'topReal': tR, 'topGen': tG})
    print(f'\n  TRUE "{w}"  |  REAL-gallery rank={rR}   GEN-gallery rank={rG}  (of {len(cw)})', flush=True)
    print('    REAL top10:', ' '.join(f'{a}({b:.2f})' for a, b in tR), flush=True)
    print('    GEN  top10:', ' '.join(f'{a}({b:.2f})' for a, b in tG), flush=True)

json.dump({'chunk': ch, 'text': utt['text'], 'n_vocab': int(len(cw)), 'rows': rows},
          open('outputs/word_candidates_genspace.json', 'w'), indent=1)

# ── figure ──
fig, axes = plt.subplots(len(rows), 1, figsize=(13, 1.5 * len(rows)))
if len(rows) == 1: axes = [axes]
for ax, r, gm in zip(axes, rows, gmels):
    sub = ax.inset_axes([0.0, 0.0, 0.13, 1.0]); sub.imshow(gm, origin='lower', aspect='auto', cmap='magma')
    sub.set_xticks([]); sub.set_yticks([]); ax.axis('off')
    cR = '#b2182b' if not (0 < r['rankReal'] <= 10) else '#b58900'
    cG = '#1b7837' if (0 < r['rankGen'] <= 10) else '#b2182b'
    ax.text(0.15, 0.80, f'TRUE "{r["word"]}"', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.text(0.15, 0.50, f'REAL gallery  rank {r["rankReal"]}/{len(cw)}:  ' +
            '  '.join(f'{a}·{b:.2f}' for a, b in r['topReal'][:8]), transform=ax.transAxes,
            fontsize=7.5, family='monospace', color=cR)
    ax.text(0.15, 0.18, f'GEN gallery   rank {r["rankGen"]}/{len(cw)}:  ' +
            '  '.join(f'{a}·{b:.2f}' for a, b in r['topGen'][:8]), transform=ax.transAxes,
            fontsize=7.5, family='monospace', color=cG)
fig.suptitle(f'M15 word candidates — REAL-mel gallery (gen↔real) vs GEN-mel gallery (gen↔gen)  ·  vocab={len(cw)}\n'
             f'[{ch}] "{utt["text"]}"   — does matching test-gen to TRAIN-gen (same encoded space) recover the word?',
             fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig('outputs/word_candidates_genspace.png', dpi=140, bbox_inches='tight')
print('\n[done] wrote outputs/word_candidates_genspace.png', flush=True)
