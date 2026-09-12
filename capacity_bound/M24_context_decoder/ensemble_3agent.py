#!/usr/bin/env python3
"""
ensemble_3agent.py — how many TEST content words are perfectly detected when 3 agents
(M22, M15, M24) each vote and the majority is accepted?

Agents (all leak-free; lattice = M22-gen on TEST sentences, M15 not trained on test):
  M22 : powerline per-word channel retrieval top-1  = lattice topk[0]
  M15 : M15 full-utterance gen -> crop word -> nearest of the slot's candidates
        (cosine to REAL train centroid) top-1
  M24 : Qwen-1.5B language-model context rerank of the slot's candidates
        (prefix = sentence with function words + prior content = channel-top1; honest)

Accepted = majority word (>=2 agents agree); if all 3 differ -> M24's LM pick.
Perfect detection = accepted == true word. Reports each agent + ensemble + union
(any-agent-correct upper bound), on in-vocab content and on ALL content words.
"""
import os, sys, json, importlib, time
import numpy as np
import torch, torch.nn.functional as F
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
PROOT = '<REPO_ROOT>'
M15D = os.path.join(PROOT, 'M15_soundbar_melgen_word')
WIDX = os.path.join(PROOT, 'M20_powerline_word_classifier', 'word_index.json')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CTX = 0.10; T = 64
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# ── lattice (M22 channel candidates + timing, TEST sentences) ──
lat = json.load(open(os.path.join(HERE, 'data/lattice_genM22_big.json')))
print(f'[lat] {len(lat["sentences"])} test sentences', flush=True)

# ── M15 generator ──
sys.path.insert(0, M15D)
CFG = importlib.import_module('config').CFG
Mm = importlib.import_module('models')
io15 = importlib.import_module('data_io')
ck = torch.load(os.path.join(M15D, 'outputs/run1/last.pt'), map_location=dev, weights_only=False)
m15 = Mm.build(CFG).to(dev); m15.load_state_dict(ck.get('ema', ck['model'])); m15.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
melfn = torchaudio.transforms.MelSpectrogram(CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop,
                                             win_length=CFG.win_length, n_mels=CFG.n_mels,
                                             f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)
man = json.load(open(os.path.join(M15D, 'full_manifest.json')))
utt2meta = {r['utt_id']: r for r in man}
print(f'[m15] loaded; manifest utts={len(utt2meta)}', flush=True)

def norm64(m): return F.interpolate(torch.as_tensor(m)[None, None].float(), size=(80, T),
                                    mode='bilinear', align_corners=False)[0, 0]
def embed(m64):
    z = m64.reshape(-1); z = z - z.mean(); return z / (z.norm() + 1e-8)

# ── REAL centroids for every candidate word (train occ) ──
occ = json.load(open(WIDX))
def is_test(ch): return int(ch.split('_')[1]) % 12 == 0
cand_words = set()
for s in lat['sentences']:
    for sl in s['slots']:
        if sl.get('is_content') and 'topk' in sl:
            for w, _ in sl['topk']:
                cand_words.add(w)
print(f'[cent] building real centroids for {len(cand_words)} candidate words …', flush=True)
def real64(ch, s, e):
    w = io15.read_wav_window(CFG.wav_path(ch), s - CTX, (e - s) + 2 * CTX, CFG.ref_sr).astype(np.float32)
    return norm64(torch.log(melfn(torch.from_numpy(w)) + CFG.log_eps))
CENT = {}
t0 = time.time()
for i, w in enumerate(sorted(cand_words)):
    tr = [(c, s, e) for (c, s, e) in occ.get(w, []) if not is_test(c)][:6]
    embs = []
    for (c, s, e) in tr:
        try: embs.append(embed(real64(c, s, e)))
        except Exception: pass
    if embs:
        v = torch.stack(embs).mean(0); CENT[w] = (v / (v.norm() + 1e-8))
    if i % 500 == 0: print(f'  cent {i}/{len(cand_words)} {time.time()-t0:.0f}s', flush=True)
print(f'[cent] {len(CENT)} centroids', flush=True)

# ── M15 gen per sentence utterance (cached), crop each content slot ──
_UC = {}
@torch.no_grad()
def gen_utt(ch, us, dur):
    k = (ch, round(us, 3))
    if k not in _UC:
        lag = io15.read_lag_ms(CFG.lag_path(ch)) / 1000.0
        raw = io15.read_bin_window(CFG.bin_path(ch), us + lag, min(dur, CFG.max_dur_s), CFG.cap_sr).astype(np.float32)
        raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
        x = np.zeros(CFG.a_len, np.float32); x[:min(len(raw), CFG.a_len)] = raw[:CFG.a_len]
        _UC[k] = (m15.sample(torch.from_numpy(x)[None].to(dev))[0].float().cpu() * ms + mm)
    return _UC[k]

# ── Qwen LM ──
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()
PRE = 'A sentence from a classic English novel: '
@torch.no_grad()
def lm_firsttoken_logits(prefix):
    ids = tok(prefix, return_tensors='pt').input_ids.to(dev)
    out = lm(ids).logits[0, -1]
    return torch.log_softmax(out.float(), -1)

# ── run agents over content slots ──
res = []  # per in-vocab content slot: dict(w, m22, m15, m24)
n_content = 0; n_m15_ok_utt = 0
t0 = time.time()
for s in lat['sentences']:
    meta = utt2meta.get(s['utt_id'])
    # left-to-right context words for the LM (function=known word, content=channel-top1)
    ctx_words = []
    gutt = None
    if meta is not None:
        try: gutt = gen_utt(meta['chunk'], meta['start_s'], meta['dur_s'])
        except Exception: gutt = None
    for sl in s['slots']:
        if not sl.get('is_content'):
            ctx_words.append(sl['w']); continue
        n_content += 1
        if 'topk' not in sl or sl.get('unknown'):
            ctx_words.append('thing'); continue
        w = sl['w']; cands = [c[0] for c in sl['topk']]; chsc = np.array([c[1] for c in sl['topk']])
        m22 = cands[0]
        # M24: LM first-token logprob over candidates given honest prefix
        prefix = PRE + ' '.join(ctx_words) + (' ' if ctx_words else '')
        logp = lm_firsttoken_logits(prefix)
        lm_s = np.array([float(logp[tok(' ' + c, add_special_tokens=False).input_ids[0]]) for c in cands])
        chz = (chsc - chsc.mean()) / (chsc.std() + 1e-6); lmz = (lm_s - lm_s.mean()) / (lm_s.std() + 1e-6)
        m24 = cands[int(np.argmax(chz + lmz))]
        # M15: crop word from gen utt, score vs candidate centroids
        m15w = None
        if gutt is not None and meta is not None and 'start' in sl:
            f0 = max(0, int((sl['start'] - CTX - meta['start_s']) * CFG.fps))
            f1 = min(gutt.shape[1], int((sl['start'] + sl['dur'] + CTX - meta['start_s']) * CFG.fps))
            if f1 - f0 >= 3:
                q = embed(norm64(gutt[:, f0:f1]))
                best = -9;
                for c in cands:
                    if c in CENT:
                        sc = float(q @ CENT[c])
                        if sc > best: best, m15w = sc, c
        res.append({'w': w, 'm22': m22, 'm24': m24, 'm15': m15w})
        ctx_words.append(m22)   # honest running context = channel top-1
    if len(res) and len(res) % 200 < 20:
        print(f'  slots={len(res)} utt-cache={len(_UC)} {time.time()-t0:.0f}s', flush=True)

# ── ensemble vote ──
def vote(r):
    votes = [x for x in (r['m22'], r['m15'], r['m24']) if x is not None]
    from collections import Counter
    cnt = Counter(votes)
    top, k = cnt.most_common(1)[0]
    return top if k >= 2 else r['m24']    # tie/all-diff -> LM-informed pick

N = len(res)
acc = lambda key: sum(r[key] == r['w'] for r in res)
ens = sum(vote(r) == r['w'] for r in res)
union = sum(r['w'] in (r['m22'], r['m15'], r['m24']) for r in res)
allc = n_content
def pc(x, d): return f'{x} ({100*x/d:.1f}%)'
print('\n==== 3-AGENT ENSEMBLE (perfect detection) ====', flush=True)
print(f'in-vocab content slots scored = {N}   (all content words = {allc})', flush=True)
print(f'  M22 alone      : {pc(acc("m22"),N)} of in-vocab | {100*acc("m22")/allc:.1f}% of all content', flush=True)
print(f'  M15 alone      : {pc(acc("m15"),N)} of in-vocab | {100*acc("m15")/allc:.1f}% of all content', flush=True)
print(f'  M24 alone      : {pc(acc("m24"),N)} of in-vocab | {100*acc("m24")/allc:.1f}% of all content', flush=True)
print(f'  ENSEMBLE (vote): {pc(ens,N)} of in-vocab | {100*ens/allc:.1f}% of all content', flush=True)
print(f'  union (any correct, upper bound): {pc(union,N)} of in-vocab | {100*union/allc:.1f}% of all content', flush=True)
json.dump({'N_invocab': N, 'N_all_content': allc,
           'm22': acc('m22'), 'm15': acc('m15'), 'm24': acc('m24'),
           'ensemble': ens, 'union': union},
          open(os.path.join(HERE, 'outputs/ensemble_3agent.json'), 'w'), indent=1)
print('[done] wrote outputs/ensemble_3agent.json', flush=True)
