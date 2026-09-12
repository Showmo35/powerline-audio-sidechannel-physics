#!/usr/bin/env python3
"""
build_lattice.py — M24 Stage 2: build a RICHER attack lattice to raise the ceiling
that Stage-1 hit (candidate-slot oracle 0.130, coverage 0.62).

Two changes vs. M23 attack_build:
  * FULL-VOCAB gallery — enroll many more content-word types (lower MIN_OCC, higher
    MAX_TYPES) → raises COVERAGE (fraction of GT content words that are in the gallery).
  * TOP-100 candidates per content slot (vs 10) → raises the ORACLE ceiling
    (fraction of slots whose true word is in the candidate list at all).

Generator = M22 (set M23_GEN=m22 in the environment; we import M23's build_mels for
the generator + mel readers). Output lattice has the SAME schema decode.py reads.

Output -> M24_context_decoder/data/lattice_<tag>.json
"""
import os, sys, json, time, re, argparse
from collections import defaultdict
import numpy as np

os.environ.setdefault('HF_HUB_OFFLINE', '1')
PROOT = '<REPO_ROOT>'
M23 = os.path.join(PROOT, 'M23_crossmodal_word_retrieval')
sys.path.insert(0, M23)
import build_mels as B                      # generator (M23_GEN), real_mel, gen_mels, C20, D, STOP, T

MANIFEST = os.path.join(PROOT, 'Powerline_Data_Captures', 'full_manifest.json')
HERE = os.path.dirname(os.path.abspath(__file__))


def is_content(w):
    w = w.lower()
    return len(w) >= 4 and w not in B.STOP


def toks(t):
    return re.sub(r'[^A-Za-z ]', ' ', t).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-occ', type=int, default=5)      # was 25 -> more coverage
    ap.add_argument('--max-types', type=int, default=12000)  # was 2000
    ap.add_argument('--cap', type=int, default=20)         # exemplars per type
    ap.add_argument('--n-sent', type=int, default=250)
    ap.add_argument('--topk', type=int, default=100)       # was 10 -> higher oracle
    ap.add_argument('--gen-bs', type=int, default=16)
    ap.add_argument('--tag', default='genM22_big')
    args = ap.parse_args()
    LEN_LO, LEN_HI = 5, 25
    OUT = os.path.join(HERE, 'data'); os.makedirs(OUT, exist_ok=True)
    print(f'[lattice] GEN={B.GEN}  min_occ={args.min_occ} max_types={args.max_types} '
          f'cap={args.cap} topk={args.topk}', flush=True)

    occ = json.load(open(B.D.INDEX))

    # ── gallery vocabulary (content types by train frequency) ─────────────────
    train_occ = {}
    for w, rows in occ.items():
        if not is_content(w):
            continue
        tr = [r for r in rows if not B.C20.is_test(r[0])]
        if len(tr) >= args.min_occ:
            train_occ[w] = tr
    types = sorted(train_occ, key=lambda w: -len(train_occ[w]))[:args.max_types]
    tid = {w: i for i, w in enumerate(types)}
    print(f'[lattice] gallery types={len(types)}', flush=True)

    # ── gallery real mels grouped by type (contiguous) ────────────────────────
    rng = np.random.RandomState(0)
    g_mels, bounds = [], []
    t0 = time.time()
    for i, w in enumerate(types):
        rows = train_occ[w][:]; rng.shuffle(rows); rows = rows[:args.cap]
        bounds.append(len(g_mels))
        for ch, s, e in rows:
            m, _ = B.real_mel(ch, s, e); g_mels.append(m)
        if i % 500 == 0:
            print(f'[gallery] {i}/{len(types)} types, {len(g_mels)} mels, {time.time()-t0:.0f}s', flush=True)
    G = np.stack(g_mels).astype(np.float32)
    bounds = np.array(bounds)
    print(f'[lattice] gallery mels={len(G)}  {time.time()-t0:.0f}s', flush=True)

    Graw = G.reshape(len(G), -1)
    mu = Graw.mean(0, keepdims=True); sd = Graw.std(0, keepdims=True) + 1e-8
    Gn = (Graw - mu) / sd
    Gn /= (np.linalg.norm(Gn, axis=1, keepdims=True) + 1e-8)

    def retrieve(qmels):
        Q = qmels.reshape(len(qmels), -1)
        Qn = (Q - mu) / sd
        Qn /= (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-8)
        S = Qn @ Gn.T
        Stype = np.maximum.reduceat(S, bounds, axis=1)
        idx = np.argsort(-Stype, axis=1)[:, :args.topk]
        return idx, np.take_along_axis(Stype, idx, axis=1)

    # ── chunk -> sorted [(s,e,w)] ─────────────────────────────────────────────
    chunk_words = {}
    for w, rows in occ.items():
        for ch, s, e in rows:
            chunk_words.setdefault(ch, []).append((s, e, w))
    for ch in chunk_words:
        chunk_words[ch].sort()

    manifest = json.load(open(MANIFEST))
    utts = [u for u in manifest if B.C20.is_test(u['chunk'])
            and LEN_LO <= len(u['text'].split()) <= LEN_HI]
    rng.shuffle(utts); utts = utts[:args.n_sent]
    print(f'[lattice] held-out sentences={len(utts)}', flush=True)

    sents, slot_items, slot_ref = [], [], []
    eps = 0.15
    for si, u in enumerate(utts):
        span = [x for x in chunk_words.get(u['chunk'], [])
                if x[0] >= u['start_s'] - eps and x[1] <= u['end_s'] + eps]
        byword = defaultdict(list)
        for s, e, w in span:
            byword[w.lower()].append((s, e))
        for w in byword:
            byword[w].sort()
        used = defaultdict(int); slots = []
        for tok in toks(u['text']):
            lw = tok.lower(); c = is_content(lw); slot = {'w': lw, 'is_content': c}
            if c:
                times = byword.get(lw, []); k = used[lw]
                if k < len(times):
                    s, e = times[k]; used[lw] += 1
                    slot['start'] = round(s, 3); slot['dur'] = round(e - s, 3)
                    slot_ref.append((si, len(slots))); slot_items.append((u['chunk'], s, e, 0))
                else:
                    slot['unknown'] = True
            slots.append(slot)
        sents.append({'utt_id': u['utt_id'], 'text': u['text'], 'slots': slots})

    print(f'[lattice] content slots to generate={len(slot_items)}', flush=True)
    t0 = time.time(); all_q = []
    for b in range(0, len(slot_items), args.gen_bs):
        all_q.extend(B.gen_mels(slot_items[b:b + args.gen_bs]))
        if b % (args.gen_bs * 20) == 0:
            print(f'[gen-slots] {b}/{len(slot_items)}  {time.time()-t0:.0f}s', flush=True)
    all_q = np.stack(all_q).astype(np.float32)
    idx, sim = retrieve(all_q)

    hit = 0
    for n, (si, sj) in enumerate(slot_ref):
        cands = [[types[int(idx[n, r])], float(sim[n, r])] for r in range(args.topk)]
        sents[si]['slots'][sj]['topk'] = cands
        gt = sents[si]['slots'][sj]['w']
        o = gt in [c[0] for c in cands]
        sents[si]['slots'][sj]['oracle'] = bool(o); hit += int(o)
    oracle = hit / max(1, len(slot_ref))
    in_gal = float(np.mean([sents[si]['slots'][sj]['w'] in tid for si, sj in slot_ref]))
    print(f'[lattice] oracle top-{args.topk} content recall = {oracle:.3f}  '
          f'(coverage = {in_gal:.3f})', flush=True)

    out = os.path.join(OUT, f'lattice_{args.tag}.json')
    json.dump({'params': {'min_occ': args.min_occ, 'max_types': args.max_types,
                          'cap': args.cap, 'topk': args.topk, 'gen': B.GEN,
                          'gallery_types': len(types), 'gallery_mels': int(len(G))},
               'oracle_topk_content_recall': oracle,
               'content_slots_in_vocab': in_gal,
               'sentences': sents}, open(out, 'w'), indent=1)
    print('[lattice] wrote', out, flush=True)


if __name__ == '__main__':
    main()
