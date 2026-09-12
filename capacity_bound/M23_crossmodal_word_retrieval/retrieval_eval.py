#!/usr/bin/env python3
"""
retrieval_eval.py — M23 stage 2 (CPU). THE DECIDING EXPERIMENT.

k-NN query-by-example word retrieval: each query mel is matched (cosine) against
the gallery of real word-mels; predicted word = vote of its nearest neighbours.

Three feature conditions, applied identically to gallery and query
(per-item transform, gallery-standardised columns, L2-normalised rows):

  A  full-mel        flatten(T x 80)                 prosody + any phonetics
  B  envelope-null   per-frame(freq)-demeaned mel    phonetics ALONE
  C  envelope-only   energy contour (mean/freq) + logdur   prosody ALONE
  Cd duration-only   logdur scalar                   nearest-duration baseline

Query sets:
  gen   = M14-generated-from-powerline mels  (THE RESULT — cross-modal)
  real  = real test mels                     (ceiling — same-modality retrieval)

Deciding rules (chance = 1/30 = 3.33%):
  B(gen) >> chance      -> phonetics survive  (would overturn CHIRP verdict)
  A(gen) ~ C(gen)       -> retrieval is prosodic, spectrum adds nothing
Plus a vocabulary-scaling curve (accuracy vs #classes) and a confusion matrix
annotated with per-word mean duration (prosodic-collision check).
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

VOCAB = os.environ.get('M23_VOCAB', 'freq')
_base = os.path.dirname(os.path.abspath(__file__))
D   = os.path.join(_base, 'data' if VOCAB == 'freq' else 'data_content')
OUT = os.path.join(_base, 'results' if VOCAB == 'freq' else 'results_content')
os.makedirs(OUT, exist_ok=True)

gal  = np.load(f'{D}/gal_mel.npy'); galy = np.load(f'{D}/gal_y.npy'); gald = np.load(f'{D}/gal_dur.npy')
qgen = np.load(f'{D}/q_gen.npy');   qy   = np.load(f'{D}/q_y.npy');   qd   = np.load(f'{D}/q_dur.npy')
qreal = np.load(f'{D}/q_real.npy')
words = json.load(open(f'{D}/words.json')); K = len(words)
CHANCE = 1.0 / K
print(f'[eval] gallery={len(galy)} query={len(qy)} vocab={K} chance={CHANCE:.4f}', flush=True)


# ── feature transforms (M: (N,T,80), d: (N,) durations) ──────────────────────
def featA(M, d):                       # full mel
    return M.reshape(len(M), -1)

def featB(M, d):                       # envelope-null: remove per-frame loudness
    return (M - M.mean(axis=2, keepdims=True)).reshape(len(M), -1)

def featC(M, d):                       # envelope-only: energy contour + logdur
    e = M.mean(axis=2)                                  # (N,T) log-energy contour
    return np.concatenate([e, np.log(d)[:, None]], axis=1)

def featCd(M, d):                      # duration only
    return np.log(d)[:, None]

FEATS = {'A_fullmel': featA, 'B_envnull': featB, 'C_envonly': featC, 'Cd_duronly': featCd}


def prep(Fg, Fq):
    mu = Fg.mean(0, keepdims=True); sd = Fg.std(0, keepdims=True) + 1e-8
    Fg = (Fg - mu) / sd; Fq = (Fq - mu) / sd
    Fg = Fg / (np.linalg.norm(Fg, axis=1, keepdims=True) + 1e-8)
    Fq = Fq / (np.linalg.norm(Fq, axis=1, keepdims=True) + 1e-8)
    return Fg, Fq


def macro_f1(y, p, k=K):
    fs = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs))


def knn(Fg, gy, Fq, qy_, ks=(1, 3, 5), topk=5, k_classes=K):
    S = Fq @ Fg.T
    order = np.argsort(-S, axis=1)[:, :max(max(ks), topk)]
    res = {}
    pred1 = None
    for k in ks:
        pred = np.array([np.bincount(gy[order[i, :k]], minlength=k_classes).argmax()
                         for i in range(len(qy_))])
        if k == 1:
            pred1 = pred
        res[f'top1_k{k}'] = float((pred == qy_).mean())
        res[f'macroF1_k{k}'] = macro_f1(qy_, pred, k_classes)
    res['top5'] = float(np.mean([qy_[i] in gy[order[i, :topk]] for i in range(len(qy_))]))
    return res, pred1


# ── main sweep: {gen, real} x {A,B,C,Cd} ─────────────────────────────────────
QUERIES = {'gen': qgen, 'real': qreal}
summary, preds = {}, {}
for qname, Q in QUERIES.items():
    summary[qname] = {}
    for fname, fn in FEATS.items():
        Fg, Fq = prep(fn(gal, gald), fn(Q, qd))
        r, p1 = knn(Fg, galy, Fq, qy)
        summary[qname][fname] = r
        preds[(qname, fname)] = p1
        print(f'[{qname:4s}|{fname:11s}] top1_k1={r["top1_k1"]:.3f}  '
              f'top1_k5={r["top1_k5"]:.3f}  macroF1_k5={r["macroF1_k5"]:.3f}  top5={r["top5"]:.3f}',
              flush=True)

# ── permutation / chance band (shuffle gallery labels, A/gen) ─────────────────
rngperm = np.random.RandomState(0)
Fg, Fq = prep(featA(gal, gald), featA(qgen, qd))
perm_acc = []
for _ in range(20):
    r, _ = knn(Fg, rngperm.permutation(galy), Fq, qy, ks=(1,), topk=1)
    perm_acc.append(r['top1_k1'])
perm = {'mean': float(np.mean(perm_acc)), 'std': float(np.std(perm_acc)), 'analytic_chance': CHANCE}
print(f'[perm] shuffled-label top1 = {perm["mean"]:.4f} +/- {perm["std"]:.4f}  (1/K={CHANCE:.4f})', flush=True)

# ── vocabulary-scaling curve: accuracy vs #classes (by frequency rank) ────────
order_by_freq = [w for w, _ in sorted([(c, np.sum(galy == c)) for c in range(K)], key=lambda t: -t[1])]
scaling = {'n_classes': [], 'A_gen': [], 'C_gen': [], 'A_real': [], 'chance': []}
for n in [2, 5, 10, 20, K]:
    keep = set(order_by_freq[:n])
    gm = np.isin(galy, list(keep)); qm = np.isin(qy, list(keep))
    remap = {c: i for i, c in enumerate(order_by_freq[:n])}
    gy2 = np.array([remap[c] for c in galy[gm]]); qy2 = np.array([remap[c] for c in qy[qm]])
    row = {}
    for label, Qset, feat in [('A_gen', qgen, featA), ('C_gen', qgen, featC), ('A_real', qreal, featA)]:
        Fg, Fq = prep(feat(gal[gm], gald[gm]), feat(Qset[qm], qd[qm]))
        r, _ = knn(Fg, gy2, Fq, qy2, ks=(1,), topk=1, k_classes=n)
        row[label] = r['top1_k1']
    scaling['n_classes'].append(n)
    for k in ('A_gen', 'C_gen', 'A_real'):
        scaling[k].append(row[k])
    scaling['chance'].append(1.0 / n)
    print(f'[scale] n={n:2d}  A_gen={row["A_gen"]:.3f}  C_gen={row["C_gen"]:.3f}  '
          f'A_real={row["A_real"]:.3f}  chance={1.0/n:.3f}', flush=True)

# ── per-word mean duration (prosodic-collision reference) ─────────────────────
word_dur = {words[c]: float(np.mean(gald[galy == c])) for c in range(K)}

# ── save results ──────────────────────────────────────────────────────────────
out = {'vocab': words, 'chance': CHANCE, 'n_gallery': int(len(galy)), 'n_query': int(len(qy)),
       'summary': summary, 'permutation': perm, 'scaling': scaling, 'word_mean_dur': word_dur}
json.dump(out, open(f'{OUT}/results.json', 'w'), indent=1)

# ── figure 1: condition bar chart (gen vs real, top1_k5) ─────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
conds = list(FEATS.keys()); x = np.arange(len(conds)); w = 0.38
gv = [summary['gen'][c]['top1_k5'] for c in conds]
rv = [summary['real'][c]['top1_k5'] for c in conds]
ax.bar(x - w/2, gv, w, label='query = powerline-generated (RESULT)', color='#e08a10')
ax.bar(x + w/2, rv, w, label='query = real mel (ceiling)', color='#2563d6')
ax.axhline(CHANCE, ls='--', c='#888', lw=1, label=f'chance 1/{K}')
ax.set_xticks(x); ax.set_xticklabels(conds, rotation=12); ax.set_ylabel('top-1 accuracy (k=5)')
ax.set_title('M23 cross-modal word retrieval — deciding conditions'); ax.legend(fontsize=7)
for i, v in enumerate(gv): ax.text(i - w/2, v + .005, f'{v:.2f}', ha='center', fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/conditions.png'); plt.close(fig)

# ── figure 2: scaling curve ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=130)
nc = scaling['n_classes']
ax.plot(nc, scaling['A_gen'], 'o-', c='#e08a10', label='A full-mel (gen)')
ax.plot(nc, scaling['C_gen'], 's-', c='#d64545', label='C envelope-only (gen)')
ax.plot(nc, scaling['A_real'], '^-', c='#2563d6', label='A full-mel (real, ceiling)')
ax.plot(nc, scaling['chance'], '--', c='#888', label='chance 1/n')
ax.set_xlabel('vocabulary size (# classes)'); ax.set_ylabel('top-1 accuracy'); ax.set_yscale('log')
ax.set_title('Vocabulary-scaling collapse'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/scaling.png'); plt.close(fig)

# ── figure 3: confusion matrix (gen, condition C, k=1), words sorted by duration
p = preds[('gen', 'C_envonly')]
dorder = sorted(range(K), key=lambda c: word_dur[words[c]])
Cm = np.zeros((K, K))
for t, pr in zip(qy, p):
    Cm[t, pr] += 1
Cm = Cm[np.ix_(dorder, dorder)]
Cm = Cm / (Cm.sum(1, keepdims=True) + 1e-9)
fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=130)
im = ax.imshow(Cm, cmap='magma', vmin=0, vmax=min(1.0, Cm.max()))
lab = [f'{words[c]} ({word_dur[words[c]]:.2f}s)' for c in dorder]
ax.set_xticks(range(K)); ax.set_xticklabels(lab, rotation=90, fontsize=5)
ax.set_yticks(range(K)); ax.set_yticklabels(lab, fontsize=5)
ax.set_title('Confusion (gen, envelope-only), rows/cols sorted by duration\n'
             'block structure near diagonal = prosodic collision')
ax.set_xlabel('predicted'); ax.set_ylabel('true'); fig.colorbar(im, fraction=0.046)
fig.tight_layout(); fig.savefig(f'{OUT}/confusion_envonly.png'); plt.close(fig)

# ── verdict ──────────────────────────────────────────────────────────────────
Bg = summary['gen']['B_envnull']['top1_k1']; Ag = summary['gen']['A_fullmel']['top1_k1']
Cg = summary['gen']['C_envonly']['top1_k1']
print('\n================ M23 VERDICT ================', flush=True)
print(f'chance                 = {CHANCE:.3f}', flush=True)
print(f'A full-mel   (gen)     = {Ag:.3f}', flush=True)
print(f'B env-null   (gen)     = {Bg:.3f}   <- phonetics test', flush=True)
print(f'C env-only   (gen)     = {Cg:.3f}', flush=True)
b_over = (Bg - perm['mean']) / (perm['std'] + 1e-9)
print(f'B over shuffled-chance = {b_over:.1f} sigma', flush=True)
if b_over > 3 and Bg > 2 * CHANCE:
    print('=> PHONETICS PRESENT: B beats chance -> revisit CHIRP verdict.', flush=True)
elif Ag > 1.15 * Cg:
    print('=> SPECTRUM ADDS (non-phonetic): A > C but B~chance.', flush=True)
else:
    print('=> PROSODIC RETRIEVAL ONLY: A~C, B~chance -> prosody-limited, '
          'phonetics-absent (consistent with M14/M20/CHIRP).', flush=True)
print('=> see results/ for json + figures', flush=True)
