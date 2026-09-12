#!/usr/bin/env python3
"""
build_mels.py — M23 stage 1 (GPU).

Builds the retrieval corpus for cross-modal query-by-example word spotting:

  GALLERY (enrollment)  = REAL log-mels of every TRAIN word occurrence.
  QUERY  (test)         = M14-GENERATED log-mels from the POWERLINE .bin
                          for every TEST word occurrence  (the "generate" step).
  QUERY-REAL (control)  = REAL log-mels of every TEST occurrence (real->real
                          ceiling — how well the representation retrieves at all).

Word occurrences, vocab (top-30 frequent words) and the chunk split come straight
from M20 (dataset.build_items); the generator is M14 PowerLine-Flow (last.pt).
Each word is time-normalised to T=64 frames x 80 mel bins; the raw duration is
kept as a separate scalar so the retrieval stage can add/remove the duration cue.

No leakage: gallery = train chunks, queries = held-out test chunks (M20 split,
every 12th chunk). Output arrays -> M23_.../data/.
"""
import sys, os, json, importlib, time
import numpy as np, torch, wave, torchaudio
from scipy.signal import resample_poly

PROOT = '<REPO_ROOT>'
M20   = os.path.join(PROOT, 'M20_powerline_word_classifier')
M14   = os.path.join(PROOT, 'M14_new_setup_soundbar_melgen')
M22   = os.path.join(PROOT, 'M22_retrieval_generator')
# GEN: which generator produces the query mels.
#   'm14' = original reconstruction generator (default, baseline);
#   'm22' = fresh open-vocab retrieval-optimized generator (M22/outputs/best.pt).
GEN   = os.environ.get('M23_GEN', 'm14')
SUF   = '' if GEN == 'm14' else f'_gen{GEN.upper()}'   # e.g. m22 -> '_genM22'
# VOCAB mode: 'freq' = top-K frequent (function words, original run);
#             'content' = top-K CONTENT words (stopwords removed) -> attack-relevant.
VOCAB = os.environ.get('M23_VOCAB', 'freq')
OUT   = os.path.join(PROOT, 'M23_crossmodal_word_retrieval',
                     ('data' if VOCAB == 'freq' else 'data_content') + SUF)
os.makedirs(OUT, exist_ok=True)

# function/closed-class words to EXCLUDE when building the content vocabulary
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
    "little more most less least good great long"
).split())

CTX = 0.10      # seconds of context each side of the word (matches M20 make_before)
T   = 64        # time-normalised frames per word
GEN_BS = 16     # generator batch size

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[m23] device={dev}', flush=True)

# ── M20: word occurrences, vocab, split, paths ────────────────────────────────
sys.path.insert(0, M20)
from config import CFG as C20          # noqa: E402
import dataset as D                    # noqa: E402  (binds M20 config internally)

# ── generator (PowerLine-Flow architecture; M14 or M22 weights) ───────────────
# M14 and M22 share the identical DiT architecture, so the same gen_mels() works;
# only the config/models module, checkpoint, and mel mean/std differ. M22 was
# trained on RMS-normalized input, so RMS_NORM is enabled for it (M14 was fed raw
# here originally, so its path stays byte-identical for baseline reproducibility).
_GDIR = M22 if GEN == 'm22' else M14
_CKPT = os.path.join(M22, 'outputs/best.pt') if GEN == 'm22' \
        else os.path.join(M14, 'outputs/cluster_a100/last.pt')
RMS_NORM = (GEN == 'm22')
for _m in ('config', 'models'):
    sys.modules.pop(_m, None)
sys.path.insert(0, _GDIR)
c14 = importlib.import_module('config').CFG
_GM = importlib.import_module('models')
m14 = _GM.build(c14).to(dev)
_ck = torch.load(_CKPT, map_location=dev, weights_only=False)
m14.load_state_dict(_ck.get('ema', _ck['model'])); m14.eval()
mm, msd = _ck['mel_mean'], _ck['mel_std']
print(f'[m23] GEN={GEN} loaded ({m14.count_params()/1e6:.1f}M params) from {_CKPT} '
      f'| win_s={c14.win_s} in_sr={c14.in_sr} fps={c14.fps} rms_norm={RMS_NORM}', flush=True)

# ── mel front-end (identical to M14 target / M20 make_before) ────────────────
melfn = torchaudio.transforms.MelSpectrogram(
    16000, 1024, hop_length=160, win_length=640, n_mels=80,
    f_min=0, f_max=8000, power=2.0)


def read_wav(ch, t0, dur):
    with wave.open(C20.wav_path(ch), 'rb') as w:
        sr = w.getframerate()
        w.setpos(min(max(0, int(t0 * sr)), w.getnframes()))
        x = np.frombuffer(w.readframes(int(dur * sr)), np.int16).astype(np.float32) / 32768.0
    return resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x


def to_T(m):
    """(80, frames) log-mel -> (T, 80) time-normalised."""
    m = torch.nn.functional.interpolate(
        m[None, None], size=(80, T), mode='bilinear', align_corners=False)[0, 0]
    return m.T.contiguous().numpy().astype(np.float32)          # (T, 80)


def real_mel(ch, s, e):
    t0, dur = s - CTX, (e - s) + 2 * CTX
    aw = read_wav(ch, t0, dur)
    m = torch.log(melfn(torch.from_numpy(aw)) + 1e-5)           # (80, frames)
    return to_T(m), dur


_plc = {}
def _pl(ch):
    if ch not in _plc:
        if len(_plc) > 4:
            _plc.clear()
        _plc[ch] = np.memmap(C20.bin_path(ch), dtype=np.float32, mode='r')
    return _plc[ch]


@torch.no_grad()
def gen_mels(items):
    """M14-generated log-mels for a batch of (ch,s,e,y) items -> list[(T,80)]."""
    raws, frs = [], []
    L = int(c14.win_s * C20.cap_sr)
    for ch, s, e, _ in items:
        t0, dur = s - CTX, (e - s) + 2 * CTX
        lag = D.read_lag_s(ch)
        a = int((t0 + lag) * C20.cap_sr)
        raw = np.array(_pl(ch)[max(0, a):a + L], np.float32)
        raw = np.pad(raw, (0, max(0, L - len(raw))))
        r = resample_poly(raw, c14.in_sr, C20.cap_sr).astype(np.float32)
        if RMS_NORM:                                    # match M22 training input
            r = r / (np.sqrt(np.mean(r ** 2)) + 1e-8)
        raws.append(r)
        frs.append(max(6, int(round(dur * c14.fps))))
    X = torch.from_numpy(np.stack(raws)).to(dev)
    torch.manual_seed(0)
    gen = (m14.sample(X).float().cpu() * msd + mm)               # (B, 80, 400)
    return [to_T(gen[i, :, :frs[i]]) for i in range(len(items))]


def build_real(items, tag):
    mels = np.zeros((len(items), T, 80), np.float32)
    durs = np.zeros(len(items), np.float32)
    ys   = np.zeros(len(items), np.int64)
    t0 = time.time()
    for i, (ch, s, e, y) in enumerate(items):
        mels[i], durs[i] = real_mel(ch, s, e); ys[i] = y
        if i % 2000 == 0:
            print(f'[real:{tag}] {i}/{len(items)}  {time.time()-t0:.0f}s', flush=True)
    return mels, durs, ys


def build_gen(items, tag):
    mels = np.zeros((len(items), T, 80), np.float32)
    durs = np.zeros(len(items), np.float32)
    ys   = np.zeros(len(items), np.int64)
    t0 = time.time()
    for b in range(0, len(items), GEN_BS):
        bi = items[b:b + GEN_BS]
        for j, m in enumerate(gen_mels(bi)):
            mels[b + j] = m
        for j, (ch, s, e, y) in enumerate(bi):
            durs[b + j] = (e - s) + 2 * CTX; ys[b + j] = y
        if b % (GEN_BS * 20) == 0:
            print(f'[gen:{tag}] {b}/{len(items)}  {time.time()-t0:.0f}s', flush=True)
    return mels, durs, ys


def content_items():
    """Top-30 CONTENT words (stopwords removed, len>=4, >=50 occ); same chunk split."""
    occ = json.load(open(D.INDEX))
    cand = [w for w in occ if len(w) >= 4 and w not in STOP and len(occ[w]) >= 50]
    words = sorted(cand, key=lambda w: -len(occ[w]))[:C20.vocab_k]
    wid = {w: i for i, w in enumerate(words)}
    rng = np.random.RandomState(0)

    def mk(split):
        it = []
        for w in words:
            rows = [r for r in occ[w] if (C20.is_test(r[0]) == (split == 'test'))]
            rng.shuffle(rows)
            if split == 'train':
                rows = rows[:C20.max_per_word]
            for ch, s, e in rows:
                it.append((ch, s, e, wid[w]))
        return it
    return mk('train'), mk('test'), words


def main():
    if VOCAB == 'content':
        tr_items, te_items, words = content_items()
    else:
        tr_items, words = D.build_items(split='train')
        te_items, _     = D.build_items(split='test')
    print(f'[m23] VOCAB={VOCAB}  words={words}', flush=True)
    print(f'[m23] vocab={len(words)}  train={len(tr_items)}  test={len(te_items)}', flush=True)
    json.dump(words, open(os.path.join(OUT, 'words.json'), 'w'))

    # gallery = real train mels
    g_mel, g_dur, g_y = build_real(tr_items, 'gallery')
    np.save(f'{OUT}/gal_mel.npy', g_mel); np.save(f'{OUT}/gal_dur.npy', g_dur); np.save(f'{OUT}/gal_y.npy', g_y)

    # query controls = real test mels
    r_mel, r_dur, r_y = build_real(te_items, 'query-real')
    np.save(f'{OUT}/q_real.npy', r_mel); np.save(f'{OUT}/q_dur.npy', r_dur); np.save(f'{OUT}/q_y.npy', r_y)

    # query = M14-generated test mels (the "generate" step)
    q_mel, q_dur, q_y = build_gen(te_items, 'query-gen')
    assert np.array_equal(q_y, r_y), 'gen/real query order mismatch'
    np.save(f'{OUT}/q_gen.npy', q_mel)

    print('[m23] stage-1 done ->', OUT, flush=True)


if __name__ == '__main__':
    main()
