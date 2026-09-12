#!/usr/bin/env python3
"""
word_candidates_centroid.py — one test sentence, M15 word-by-word top-10 candidates
by CENTROID mel correlation.

For each word: M15 generates the full-utterance mel (from the powerline), we crop the
word region and embed it (time-norm 64 -> flatten -> mean-center -> L2, so cosine ==
Pearson r). The candidate score is cosine to the CENTROID of each word type's real-mel
distribution (mean of that word's train embeddings), NOT to a single nearest instance.
Prints the true word + its top-10 candidate words and the true word's rank.

Gallery/centroids: real TRAIN word-mels (M22's dataset). Query sentence: held-out TEST
(M15 never trained on it). Same 16 kHz mel geometry for both, so the space is shared.
"""
import os, sys, json, importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

PROOT = '<REPO_ROOT>'
M22 = os.path.join(PROOT, 'M22_retrieval_generator')
M15 = os.path.join(PROOT, 'M15_soundbar_melgen_word')
WIDX = os.path.join(PROOT, 'M20_powerline_word_classifier', 'word_index.json')
MAN = os.path.join(M15, 'full_manifest.json')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CTX = 0.10; T = 64

# ── Phase A: build real-mel gallery + word centroids (M22 modules) ──
sys.path.insert(0, M22)
D22 = importlib.import_module('dataset_words')
TR = importlib.import_module('train_retrieval')
tr_items, words = D22.build_items('train')
gal_items = D22.cap_per_label(tr_items, 30)
gal_ds = D22.WordSet(gal_items, real_only=True)
gal_dl = DataLoader(gal_ds, batch_size=256, shuffle=False, num_workers=8,
                    collate_fn=lambda bb: {'mel_word': torch.stack([x['mel_word'] for x in bb]),
                                           'label': torch.tensor([x['label'] for x in bb])})
print(f'[gallery] embedding {len(gal_items)} real word-mels over {len(words)} words …', flush=True)
G, gy = TR.build_gallery(gal_dl, dev)                       # (Ng,D) fp16, (Ng,)
# centroids per word type
present = torch.unique(gy).tolist()
C = torch.zeros(len(present), G.shape[1], device=dev)
cent_word = []
for i, c in enumerate(present):
    v = G[gy == c].float().mean(0)
    C[i] = v / (v.norm() + 1e-8)
    cent_word.append(words[c])
print(f'[centroids] {len(present)} word centroids', flush=True)

# purge M22 modules before importing M15's same-named ones
for m in ('config', 'models', 'data_io', 'dataset_words', 'train_retrieval', 'text'):
    sys.modules.pop(m, None)
sys.path.remove(M22)

# ── Phase B: M15 generator ──
sys.path.insert(0, M15)
CFG = importlib.import_module('config').CFG
Mm = importlib.import_module('models')
io15 = importlib.import_module('data_io')
ck = torch.load(os.path.join(M15, 'outputs/run1/last.pt'), map_location=dev, weights_only=False)
model = Mm.build(CFG).to(dev); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
print(f'[m15] loaded step={ck.get("step")}', flush=True)


def embed_word(m80xf):
    """(80, f) log-mel -> centered/L2 embedding in the kNN space (cosine==Pearson r)."""
    z = F.interpolate(torch.as_tensor(m80xf)[None, None].float(), size=(80, T),
                      mode='bilinear', align_corners=False)[0, 0]
    z = z.reshape(-1); z = z - z.mean()
    return (z / (z.norm() + 1e-8))


# ── pick the sentence (same seed as the mel figures -> picks[2] = chunk_084) ──
man = json.load(open(MAN))
def is_test(ch): return int(ch.split('_')[1]) % 12 == 0
test = [r for r in man if is_test(r['chunk']) and 2.0 <= r['dur_s'] <= 3.8 and len(r['text'].split()) >= 4]
rng = np.random.RandomState(1); rng.shuffle(test)
utt = test[2]
ch, us, ue = utt['chunk'], utt['start_s'], utt['end_s']
print(f'\nSENTENCE [{ch}]: "{utt["text"]}"', flush=True)

# word timeline of the sentence from word_index
occ = json.load(open(WIDX))
seg = sorted([(float(s), float(e), w) for w in occ for (c, s, e) in occ[w]
              if c == ch and s >= us - 0.05 and e <= ue + 0.05], key=lambda t: t[0])

# generate the full-utterance mel once
lag = io15.read_lag_ms(CFG.lag_path(ch)) / 1000.0
dur = min(utt['dur_s'], CFG.max_dur_s)
raw = io15.read_bin_window(CFG.bin_path(ch), us + lag, dur, CFG.cap_sr).astype(np.float32)
raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
x = np.zeros(CFG.a_len, np.float32); x[:min(len(raw), CFG.a_len)] = raw[:CFG.a_len]
with torch.no_grad():
    gen = (model.sample(torch.from_numpy(x)[None].to(dev))[0].float().cpu() * ms + mm)   # (80,1600)

cent_word_arr = np.array(cent_word)
rows = []
gen_mels = []
for (s, e, w) in seg:
    f0 = max(0, int((s - CTX - us) * CFG.fps)); f1 = min(gen.shape[1], int((e + CTX - us) * CFG.fps))
    if f1 - f0 < 3:
        f1 = min(gen.shape[1], f0 + 3)
    gm = gen[:, f0:f1]
    gen_mels.append(gm.numpy())
    q = embed_word(gm).to(dev).half()
    sims = (q @ C.t().half()).float().cpu().numpy()
    order = np.argsort(-sims)
    top10 = [(cent_word[j], float(sims[j])) for j in order[:10]]
    rank = int(np.where(cent_word_arr[order] == w)[0][0]) + 1 if w in cent_word else -1
    rows.append({'word': w, 'rank': rank, 'top10': top10})
    tag = f'rank={rank}' if rank > 0 else 'NOT in vocab'
    print(f'\n  TRUE "{w}"  ({tag})', flush=True)
    print('    top-10 by centroid corr: ' + ' | '.join(f'{cw}({sc:.2f})' for cw, sc in top10), flush=True)

in1 = np.mean([1 for r in rows if r['rank'] == 1])
in10 = np.mean([1 for r in rows if 0 < r['rank'] <= 10])
n = len(rows)
print(f'\n==== SUMMARY: {ch} "{utt["text"]}" ====', flush=True)
print(f'  words={n}  true-in-top1={sum(r["rank"]==1 for r in rows)}/{n}  '
      f'true-in-top10={sum(0<r["rank"]<=10 for r in rows)}/{n}', flush=True)
json.dump({'chunk': ch, 'text': utt['text'], 'rows': rows}, open('outputs/word_candidates.json', 'w'), indent=1)

# ── figure: per word, gen mel + its top-10 candidate list (true word in green if present) ──
fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.15 * len(rows)))
if len(rows) == 1:
    axes = [axes]
for ax, r, gm in zip(axes, rows, gen_mels):
    sub = ax.inset_axes([0.0, 0.0, 0.16, 1.0])
    sub.imshow(gm, origin='lower', aspect='auto', cmap='magma'); sub.set_xticks([]); sub.set_yticks([])
    ax.axis('off')
    cand = '   '.join((f'{cw}·{sc:.2f}') for cw, sc in r['top10'])
    col = '#1b7837' if r['rank'] == 1 else ('#b58900' if 0 < r['rank'] <= 10 else '#b2182b')
    ax.text(0.18, 0.62, f'TRUE: "{r["word"]}"   (rank {r["rank"] if r["rank"]>0 else "—"})',
            transform=ax.transAxes, fontsize=10, fontweight='bold', color=col)
    ax.text(0.18, 0.18, 'top-10: ' + cand, transform=ax.transAxes, fontsize=8, family='monospace')
fig.suptitle(f'M15 word-by-word top-10 candidates by CENTROID mel-correlation\n[{ch}] "{utt["text"]}"'
             f'   —   true-in-top1 {sum(r["rank"]==1 for r in rows)}/{n}, top10 {sum(0<r["rank"]<=10 for r in rows)}/{n}',
             fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('outputs/word_candidates.png', dpi=140, bbox_inches='tight')
print('[done] wrote outputs/word_candidates.png', flush=True)
