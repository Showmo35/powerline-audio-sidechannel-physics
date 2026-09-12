#!/usr/bin/env python3
"""
attack_build.py — M23 attack stage 1 (GPU).

Builds the material for the sentence-reconstruction attack:

  GALLERY : REAL mels of many CONTENT-word types (train chunks), up to CAP
            exemplars each -> the enrolled acoustic dictionary.
  LATTICE : for each held-out (test-chunk) SENTENCE, reconstruct its aligned
            word/time sequence (word_index.json), mark content vs function
            slots; for every CONTENT slot generate the powerline mel (M14) and
            retrieve the top-K candidate words + cosine confidence from the
            gallery. Function slots are gaps the LLM will fill.

Output -> data_attack/sentences.json  (+ oracle top-K content recall).
Ground-truth text from full_manifest.json (sentence level).
"""
import os, sys, json, time, re
from collections import defaultdict
import numpy as np, torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_mels as B                      # C20, D, m14, gen_mels, real_mel, STOP, T

PROOT = '<REPO_ROOT>'
MANIFEST = os.path.join(PROOT, 'Powerline_Data_Captures', 'full_manifest.json')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_attack' + B.SUF)
os.makedirs(OUT, exist_ok=True)

MIN_OCC   = 25     # min TRAIN occurrences to enroll a content word type
MAX_TYPES = 2000   # gallery vocabulary size (top types by train frequency)
CAP       = 25     # exemplars per type
N_SENT    = 250    # held-out sentences to attack
LEN_LO, LEN_HI = 5, 25
TOPK      = 10
GEN_BS    = 16


def is_content(w):
    w = w.lower()
    return len(w) >= 4 and w not in B.STOP


def toks(t):
    return re.sub(r'[^A-Za-z ]', ' ', t).split()


def main():
    occ = json.load(open(B.D.INDEX))

    # ── gallery vocabulary (content types, train frequency) ───────────────────
    train_occ = {}
    for w, rows in occ.items():
        if not is_content(w):
            continue
        tr = [r for r in rows if not B.C20.is_test(r[0])]
        if len(tr) >= MIN_OCC:
            train_occ[w] = tr
    types = sorted(train_occ, key=lambda w: -len(train_occ[w]))[:MAX_TYPES]
    tid = {w: i for i, w in enumerate(types)}
    print(f'[attack] gallery types={len(types)}  (e.g. {types[:12]})', flush=True)

    # ── build gallery real mels, grouped by type (contiguous) ─────────────────
    rng = np.random.RandomState(0)
    g_mels, g_type, bounds = [], [], []
    t0 = time.time()
    for i, w in enumerate(types):
        rows = train_occ[w][:]; rng.shuffle(rows); rows = rows[:CAP]
        bounds.append(len(g_mels))
        for ch, s, e in rows:
            m, _ = B.real_mel(ch, s, e)
            g_mels.append(m); g_type.append(tid[w])
        if i % 250 == 0:
            print(f'[gallery] {i}/{len(types)} types, {len(g_mels)} mels, {time.time()-t0:.0f}s', flush=True)
    G = np.stack(g_mels).astype(np.float32)              # (Ng, T, 80)
    bounds = np.array(bounds)
    print(f'[attack] gallery mels={len(G)}  {time.time()-t0:.0f}s', flush=True)

    # ── standardised, L2-normalised gallery feature matrix (condition A) ──────
    Graw = G.reshape(len(G), -1)
    mu = Graw.mean(0, keepdims=True); sd = Graw.std(0, keepdims=True) + 1e-8
    Gn = (Graw - mu) / sd
    Gn /= (np.linalg.norm(Gn, axis=1, keepdims=True) + 1e-8)

    def retrieve(qmels):
        Q = qmels.reshape(len(qmels), -1)
        Qn = (Q - mu) / sd
        Qn /= (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-8)
        S = Qn @ Gn.T                                     # (nq, Ng)
        Stype = np.maximum.reduceat(S, bounds, axis=1)    # (nq, Ntypes)
        idx = np.argsort(-Stype, axis=1)[:, :TOPK]
        return idx, np.take_along_axis(Stype, idx, axis=1)

    # ── chunk -> sorted [(s,e,w)] index ───────────────────────────────────────
    chunk_words = {}
    for w, rows in occ.items():
        for ch, s, e in rows:
            chunk_words.setdefault(ch, []).append((s, e, w))
    for ch in chunk_words:
        chunk_words[ch].sort()

    # ── held-out sentences ────────────────────────────────────────────────────
    manifest = json.load(open(MANIFEST))
    utts = [u for u in manifest if B.C20.is_test(u['chunk'])
            and LEN_LO <= len(u['text'].split()) <= LEN_HI]
    rng.shuffle(utts); utts = utts[:N_SENT]
    print(f'[attack] held-out sentences={len(utts)}', flush=True)

    # gather content slots across sentences (for batched generation).
    # Slot sequence follows the MANIFEST ground-truth word order (authoritative);
    # word_index only supplies the timing of content words (function words are
    # gaps the LLM fills; content words the aligner missed become unknown "{?}").
    sents = []
    slot_items, slot_ref = [], []           # (ch,s,e,0) and (sent_idx, slot_idx)
    eps = 0.15
    for si, u in enumerate(utts):
        span = [x for x in chunk_words.get(u['chunk'], [])
                if x[0] >= u['start_s'] - eps and x[1] <= u['end_s'] + eps]
        byword = defaultdict(list)
        for s, e, w in span:
            byword[w.lower()].append((s, e))
        for w in byword:
            byword[w].sort()
        used = defaultdict(int)
        slots = []
        for tok in toks(u['text']):
            lw = tok.lower(); c = is_content(lw)
            slot = {'w': lw, 'is_content': c}
            if c:
                times = byword.get(lw, []); k = used[lw]
                if k < len(times):
                    s, e = times[k]; used[lw] += 1
                    slot['start'] = round(s, 3); slot['dur'] = round(e - s, 3)
                    slot_ref.append((si, len(slots)))
                    slot_items.append((u['chunk'], s, e, 0))
                else:
                    slot['unknown'] = True         # content word, no alignment -> {?}
            slots.append(slot)
        sents.append({'utt_id': u['utt_id'], 'text': u['text'], 'slots': slots})

    print(f'[attack] content slots to generate={len(slot_items)}', flush=True)
    # batched M14 generation + retrieval
    t0 = time.time()
    all_q = []
    for b in range(0, len(slot_items), GEN_BS):
        all_q.extend(B.gen_mels(slot_items[b:b + GEN_BS]))
        if b % (GEN_BS * 20) == 0:
            print(f'[gen-slots] {b}/{len(slot_items)}  {time.time()-t0:.0f}s', flush=True)
    all_q = np.stack(all_q).astype(np.float32)
    idx, sim = retrieve(all_q)

    # attach top-k to slots + oracle
    hit = 0
    for n, (si, sj) in enumerate(slot_ref):
        cands = [[types[int(idx[n, r])], float(sim[n, r])] for r in range(TOPK)]
        sents[si]['slots'][sj]['topk'] = cands
        gt = sents[si]['slots'][sj]['w']
        o = gt in [c[0] for c in cands]
        sents[si]['slots'][sj]['oracle'] = bool(o); hit += int(o)
    oracle = hit / max(1, len(slot_ref))
    in_gal = np.mean([sents[si]['slots'][sj]['w'] in tid for si, sj in slot_ref])
    print(f'[attack] oracle top-{TOPK} content recall = {oracle:.3f}  '
          f'(content slots in gallery vocab = {in_gal:.3f})', flush=True)

    json.dump({'params': {'MIN_OCC': MIN_OCC, 'MAX_TYPES': MAX_TYPES, 'CAP': CAP,
                          'N_SENT': len(utts), 'TOPK': TOPK, 'gallery_types': len(types),
                          'gallery_mels': int(len(G))},
               'oracle_topk_content_recall': oracle,
               'content_slots_in_vocab': float(in_gal),
               'sentences': sents},
              open(f'{OUT}/sentences.json', 'w'), indent=1)
    print('[attack] stage-1 done ->', f'{OUT}/sentences.json', flush=True)


if __name__ == '__main__':
    main()
