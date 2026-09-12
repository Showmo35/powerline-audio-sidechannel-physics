#!/usr/bin/env python3
"""
M25 — M15 (full-utterance) generator as a word predictor via mel-similarity kNN.

The M23/M24 pipeline generated each word from an ISOLATED per-word powerline
window, which produces blurry envelope-only mels and a weak retrieval ceiling
(the "34% top-100 coverage" claim). M15 instead ingests the FULL 200 kHz
utterance, generates the whole utterance mel with context/alignment, then we crop
the word region. Its own analysis showed the generated word-mel correlates with
the TRUE word (full 0.54 / env 0.65 / PHON 0.48) but only 0.087 with the
Whisper-on-Griffin-Lim predicted word (~= random 0.085). So the content is there;
the Whisper-GL prediction path threw it away.

This script scores M15's generated mels the RIGHT way: query-by-example kNN over a
gallery of REAL train word-mels, in the same three feature conditions M23 used:

  A full-mel   flatten(64x80)                prosody + any phonetics
  B env-null   per-frame(freq)-demeaned mel  PHONETICS ALONE  (the deciding test)
  C env-only   log-energy contour + logdur   prosody ALONE
  Cd dur-only  logdur scalar

Query = M15-generated test word-mels (THE RESULT).  Control = real test mels
(same-modality ceiling).  Gallery = real train word-mels reused from M23 (real
mels are generator-independent).  Direct comparison against the M14 per-word
baseline in M23/results.

Runs both vocabularies M23 used:
  freq     top-30 frequent words          (M23 data/,          baseline gen A=0.270)
  content  top-30 content words (no stop) (M23 data_content/,  baseline gen A=0.325)

Output -> M25_m15_retrieval/results_<vocab>/  (results.json + conditions.png)
and saved M15 query mels q_genM15.npy for a later open-vocab top-100 follow-up.
"""
import os, sys, json, time, importlib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

PROOT = '<REPO_ROOT>'
M20   = os.path.join(PROOT, 'M20_powerline_word_classifier')
M15D  = os.path.join(PROOT, 'M15_soundbar_melgen_word')
M23   = os.path.join(PROOT, 'M23_crossmodal_word_retrieval')
HERE  = os.path.dirname(os.path.abspath(__file__))

CTX = 0.10          # context each side of the word — MUST match M23 gallery build
T   = 64            # time-normalised frames per word — MUST match M23
FPS = 100.0         # mel frames per second (hop 160 @ 16 kHz)
GEN_BS = 4          # unique-utterance generation batch (3.2M-sample input each)
STEPS  = 32         # M15 flow sample steps
CFGW   = 3.0        # M15 CFG scale

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[m25] device={dev}', flush=True)

# stopword list identical to M23 content vocab construction
STOP = set((
    "the a an and or but if of to in on at by for with as is are was were be been "
    "being it its this that these those he she they we you i me my we us our your his "
    "her him them their theirs hers ours not no nor so than then there here which who "
    "whom whose what when where why how all any some more most other another into onto "
    "upon about will would could should shall can may might must have has had do does "
    "did done said say says saying very well only such own same each every both few "
    "many much lot lots from down up out off over under again further once now just "
    "even also too very still yet ever never always often back away around through "
    "before after above below between within without because while during until unless "
    "though although however therefore thus hence otherwise meanwhile besides moreover "
    "am been being were are was is be being it he she they them him her his hers theirs "
    "one two three four five six seven eight nine ten first second third last next "
    "little more most less least good great long").split())

# ── M20: word occurrences, freq vocab, split, is_test, lag ────────────────────
sys.path.insert(0, M20)
C20 = importlib.import_module('config').CFG
INDEX = importlib.import_module('config').INDEX
D20 = importlib.import_module('dataset')
read_lag_s = D20.read_lag_s

def freq_items():
    te, words = D20.build_items(split='test')
    return te, words

def content_items():
    occ = json.load(open(INDEX))
    cand = [w for w in occ if len(w) >= 4 and w not in STOP and len(occ[w]) >= 50]
    words = sorted(cand, key=lambda w: -len(occ[w]))[:C20.vocab_k]
    wid = {w: i for i, w in enumerate(words)}
    rng = np.random.RandomState(0)
    it = []
    for w in words:
        rows = [r for r in occ[w] if C20.is_test(r[0])]
        rng.shuffle(rows)
        for ch, s, e in rows:
            it.append((ch, s, e, wid[w]))
    return it, words

# purge M20 modules before importing M15's identically-named modules
for m in ('config', 'dataset', 'models', 'data_io', 'text'):
    sys.modules.pop(m, None)
sys.path.remove(M20)

# ── M15 model + IO ────────────────────────────────────────────────────────────
sys.path.insert(0, M15D)
c15 = importlib.import_module('config').CFG
Mm = importlib.import_module('models')
io15 = importlib.import_module('data_io')

model = Mm.build(c15).to(dev)
ck = torch.load(os.path.join(M15D, 'outputs/run1/last.pt'), map_location=dev, weights_only=False)
model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
MM, MS = ck['mel_mean'], ck['mel_std']
print(f'[m25] M15 loaded step={ck.get("step")} ({model.count_params()/1e6:.1f}M) '
      f'a_len={c15.a_len} n_frames={c15.n_frames}', flush=True)

# utterance manifest (exact timing) for full-utterance context
man = json.load(open(os.path.join(M15D, 'full_manifest.json')))
by_chunk = {}
for r in man:
    by_chunk.setdefault(r['chunk'], []).append(r)

def find_utt(ch, ws, we):
    best = None
    for r in by_chunk.get(ch, []):
        if r['start_s'] <= ws and r['end_s'] >= we and c15.min_dur_s <= r['dur_s'] <= c15.max_dur_s:
            if best is None or r['dur_s'] < best['dur_s']:   # tightest containing utterance
                best = r
    return best

# mel front-end IDENTICAL to M23 gallery / M15 target
melfn = torchaudio.transforms.MelSpectrogram(
    16000, 1024, hop_length=160, win_length=640, n_mels=80,
    f_min=0, f_max=8000, power=2.0)

def to_T(m):                                    # (80, frames) -> (T,80)
    m = F.interpolate(m[None, None], size=(80, T), mode='bilinear', align_corners=False)[0, 0]
    return m.T.contiguous().numpy().astype(np.float32)

def real_mel(ch, s, e):
    t0, dur = s - CTX, (e - s) + 2 * CTX
    w = io15.read_wav_window(c15.wav_path(ch), t0, dur, 16000).astype(np.float32)
    m = torch.log(melfn(torch.from_numpy(w)) + 1e-5)
    return to_T(m)

# full-utterance M15 generation, cached per utterance
_UCACHE = {}
def _utt_key(u): return (u['chunk'], round(u['start_s'], 3))

def gen_utt_batch(utts):
    """Generate full-utterance mels (80,1600 raw log-mel) for a list of manifest rows."""
    raws = []
    for u in utts:
        lag = io15.read_lag_ms(c15.lag_path(u['chunk'])) / 1000.0
        dur = min(u['dur_s'], c15.max_dur_s)
        x = io15.read_bin_window(c15.bin_path(u['chunk']), u['start_s'] + lag, dur, c15.cap_sr).astype(np.float32)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8)
        out = np.zeros(c15.a_len, np.float32); out[:min(len(x), c15.a_len)] = x[:c15.a_len]
        raws.append(out)
    X = torch.from_numpy(np.stack(raws)).to(dev)
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
        gen = model.sample(X, steps=STEPS, cfg_scale=CFGW)
    gen = (gen.float().cpu() * MS + MM)          # (B,80,1600) raw log-mel
    return [gen[i] for i in range(len(utts))]

def ensure_gen(utts):
    todo = [u for u in utts if _utt_key(u) not in _UCACHE]
    # de-dup todo
    seen = {}; uniq = []
    for u in todo:
        k = _utt_key(u)
        if k not in seen:
            seen[k] = 1; uniq.append(u)
    for b in range(0, len(uniq), GEN_BS):
        bu = uniq[b:b + GEN_BS]
        for u, g in zip(bu, gen_utt_batch(bu)):
            _UCACHE[_utt_key(u)] = g

def build_queries(items, tag):
    """M15 full-utterance gen (cropped) + real control, aligned. Returns mels_gen, mels_real, y, dur, n_fallback."""
    # find utterances for all items first, then batch-generate uniques
    utts = []
    for (ch, s, e, y) in items:
        u = find_utt(ch, s, e)
        utts.append(u)
    uniq_utts = []
    seen = set()
    for u in utts:
        if u is not None and _utt_key(u) not in seen:
            seen.add(_utt_key(u)); uniq_utts.append(u)
    print(f'[{tag}] {len(items)} words in {len(uniq_utts)} unique utterances '
          f'({sum(u is None for u in utts)} words with no containing utterance -> fallback)', flush=True)
    t0 = time.time()
    for b in range(0, len(uniq_utts), 64):
        ensure_gen(uniq_utts[b:b + 64])
        print(f'[{tag}] generated {min(b+64,len(uniq_utts))}/{len(uniq_utts)} utts  {time.time()-t0:.0f}s', flush=True)

    N = len(items)
    mg = np.zeros((N, T, 80), np.float32)
    mr = np.zeros((N, T, 80), np.float32)
    yy = np.zeros(N, np.int64)
    dd = np.zeros(N, np.float32)
    nfb = 0
    for i, ((ch, s, e, y), u) in enumerate(zip(items, utts)):
        mr[i] = real_mel(ch, s, e); yy[i] = y; dd[i] = (e - s) + 2 * CTX
        if u is not None:
            g = _UCACHE[_utt_key(u)]                       # (80,1600) full utt
            f0 = max(0, int((s - CTX - u['start_s']) * FPS))
            f1 = min(g.shape[1], int((e + CTX - u['start_s']) * FPS))
            if f1 - f0 < 3:
                f1 = min(g.shape[1], f0 + 3)
            mg[i] = to_T(g[:, f0:f1])
        else:                                              # fallback: gen the word window alone
            nfb += 1
            uu = {'chunk': ch, 'start_s': max(0.0, s - CTX), 'dur_s': max(c15.min_dur_s, (e - s) + 2 * CTX)}
            gg = gen_utt_batch([uu])[0]
            w = int(((e - s) + 2 * CTX) * FPS)
            mg[i] = to_T(gg[:, :max(3, w)])
    return mg, mr, yy, dd, nfb


# ── retrieval eval (feature conditions + kNN) — same math as M23 retrieval_eval ─
def featA(M, d): return M.reshape(len(M), -1)
def featB(M, d): return (M - M.mean(axis=2, keepdims=True)).reshape(len(M), -1)   # env-null
def featC(M, d):
    e = M.mean(axis=2); return np.concatenate([e, np.log(d)[:, None]], axis=1)
def featCd(M, d): return np.log(d)[:, None]
FEATS = {'A_fullmel': featA, 'B_envnull': featB, 'C_envonly': featC, 'Cd_duronly': featCd}

def prep(Fg, Fq):
    mu = Fg.mean(0, keepdims=True); sd = Fg.std(0, keepdims=True) + 1e-8
    Fg = (Fg - mu) / sd; Fq = (Fq - mu) / sd
    Fg /= (np.linalg.norm(Fg, axis=1, keepdims=True) + 1e-8)
    Fq /= (np.linalg.norm(Fq, axis=1, keepdims=True) + 1e-8)
    return Fg, Fq

def macro_f1(y, p, k):
    fs = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs))

def knn(Fg, gy, Fq, qy_, K, ks=(1, 5), topk=5):
    S = Fq @ Fg.T
    order = np.argsort(-S, axis=1)[:, :max(max(ks), topk)]
    res = {}
    for k in ks:
        pred = np.array([np.bincount(gy[order[i, :k]], minlength=K).argmax() for i in range(len(qy_))])
        res[f'top1_k{k}'] = float((pred == qy_).mean())
        res[f'macroF1_k{k}'] = macro_f1(qy_, pred, K)
    res['top5'] = float(np.mean([qy_[i] in gy[order[i, :topk]] for i in range(len(qy_))]))
    return res


def run_vocab(mode):
    print(f'\n######## VOCAB={mode} ########', flush=True)
    if mode == 'freq':
        items, words = freq_items(); galdir = os.path.join(M23, 'data')
    else:
        items, words = content_items(); galdir = os.path.join(M23, 'data_content')
    K = len(words)
    OUT = os.path.join(HERE, f'results_{mode}'); os.makedirs(OUT, exist_ok=True)

    gal = np.load(f'{galdir}/gal_mel.npy'); galy = np.load(f'{galdir}/gal_y.npy'); gald = np.load(f'{galdir}/gal_dur.npy')
    gwords = json.load(open(f'{galdir}/words.json'))
    assert gwords == words, f'vocab mismatch {mode}: gallery {gwords[:3]}.. vs {words[:3]}..'
    print(f'[{mode}] gallery={len(galy)} over {K} words; queries={len(items)}', flush=True)

    qgen, qreal, qy, qd, nfb = build_queries(items, mode)
    np.save(f'{OUT}/q_genM15.npy', qgen); np.save(f'{OUT}/q_real.npy', qreal)
    np.save(f'{OUT}/q_y.npy', qy); np.save(f'{OUT}/q_dur.npy', qd)
    json.dump(words, open(f'{OUT}/words.json', 'w'))

    summary = {'genM15': {}, 'real': {}}
    for qname, Q in [('genM15', qgen), ('real', qreal)]:
        for fname, fn in FEATS.items():
            Fg, Fq = prep(fn(gal, gald), fn(Q, qd))
            summary[qname][fname] = knn(Fg, galy, Fq, qy, K)
            r = summary[qname][fname]
            print(f'[{qname:7s}|{fname:11s}] top1={r["top1_k1"]:.3f} '
                  f'top1_k5={r["top1_k5"]:.3f} top5={r["top5"]:.3f} macroF1_k5={r["macroF1_k5"]:.3f}', flush=True)

    CHANCE = 1.0 / K
    out = {'vocab': words, 'chance': CHANCE, 'n_gallery': int(len(galy)),
           'n_query': int(len(qy)), 'n_fallback': int(nfb), 'summary': summary}
    json.dump(out, open(f'{OUT}/results.json', 'w'), indent=1)

    # comparison figure vs M14 baseline (from M23 results)
    base_p = os.path.join(M23, 'results' if mode == 'freq' else 'results_content', 'results.json')
    base = json.load(open(base_p)) if os.path.exists(base_p) else None
    conds = list(FEATS.keys()); x = np.arange(len(conds)); w = 0.27
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=140)
    m15g = [summary['genM15'][c]['top1_k1'] for c in conds]
    realv = [summary['real'][c]['top1_k1'] for c in conds]
    if base:
        m14g = [base['summary']['gen'][c]['top1_k1'] for c in conds]
        ax.bar(x - w, m14g, w, label='M14 per-word gen (baseline)', color='#b0b0b0')
    ax.bar(x, m15g, w, label='M15 full-utterance gen (RESULT)', color='#e08a10')
    ax.bar(x + w, realv, w, label='real mel (same-modality ceiling)', color='#2563d6')
    ax.axhline(CHANCE, ls='--', c='#444', lw=1, label=f'chance 1/{K}')
    ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylabel('top-1 accuracy (k=1)')
    ax.set_title(f'M25 — M15 full-utterance generator as word predictor (vocab={mode}, K={K})\n'
                 f'B_envnull = phonetics-alone test  ·  {len(qy)} held-out query words', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for i, v in enumerate(m15g): ax.text(i, v + .006, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{OUT}/conditions.png', bbox_inches='tight'); plt.close(fig)

    # verdict
    Ag = summary['genM15']['A_fullmel']['top1_k1']; Bg = summary['genM15']['B_envnull']['top1_k1']
    Cg = summary['genM15']['C_envonly']['top1_k1']; Ar = summary['real']['A_fullmel']['top1_k1']
    print(f'\n==== M25 VERDICT ({mode}) ====', flush=True)
    print(f'chance={CHANCE:.3f}  M15gen A={Ag:.3f} B(env-null,phon)={Bg:.3f} C(env-only)={Cg:.3f}  real ceiling A={Ar:.3f}', flush=True)
    if base:
        print(f'M14 baseline gen A={base["summary"]["gen"]["A_fullmel"]["top1_k1"]:.3f} '
              f'B={base["summary"]["gen"]["B_envnull"]["top1_k1"]:.3f}', flush=True)
    if Bg > 2 * CHANCE and Ag > 1.15 * Cg:
        print('=> PHONETICS RECOVERED: env-null >> chance AND full-mel > env-only. '
              'M15 generator carries phonetic content the per-word pipeline discarded.', flush=True)
    elif Ag > 1.15 * Cg:
        print('=> spectrum adds over envelope, but env-null near chance.', flush=True)
    else:
        print('=> prosody-limited even for M15 (A~C, B~chance).', flush=True)
    print(f'   ({nfb} fallback words without a containing utterance)', flush=True)
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['freq', 'content']
    for mode in which:
        run_vocab(mode)
    print('\n[m25] done', flush=True)
