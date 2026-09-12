#!/usr/bin/env python3
"""
Grid of 6 word-classification examples (3 correct, 3 incorrect) from best-detected
words. Each row: powerline ENVELOPE (what the classifier uses) | M16 mel (M14-
generated mel from the powerline). Uses the env-only M20 classifier (best, 39%).
"""
import sys, os, json, importlib
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import resample_poly

from config import CFG, OUT_DIR, BIN
import dataset as D
import models as M

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- M20 env classifier ----
ck = torch.load(os.path.join(OUT_DIR, 'env', 'best.pt'), map_location=dev, weights_only=False)
words, mean, std = ck['words'], ck['mean'], ck['std']
K = len(words)
clf = M.build(K).to(dev)
with torch.no_grad():
    _ = clf(torch.zeros(2, CFG.n_wbins, CFG.n_frames, device=dev))     # init LazyLinear
clf.load_state_dict(ck['model']); clf.eval()

te_items, _ = D.build_items(split='test')
te_ds = D.CachedWords('test', flatten_freq=True)
dl = torch.utils.data.DataLoader(te_ds, batch_size=64, collate_fn=D.collate, num_workers=4)
preds = []
with torch.no_grad():
    for b in dl:
        x = ((b['wide'] - mean) / std).to(dev)
        preds.append(clf(x).argmax(1).cpu().numpy())
preds = np.concatenate(preds)
ytrue = te_ds.y

# per-word accuracy -> best-detected words
acc = {c: (preds[ytrue == c] == c).mean() for c in range(K) if (ytrue == c).sum() > 3}
best_words = sorted(acc, key=lambda c: -acc[c])[:12]         # focus on well-detected words
bw = set(best_words)

rng = np.random.RandomState(0)
corr_idx = [i for i in range(len(ytrue)) if ytrue[i] in bw and preds[i] == ytrue[i]]
wrong_idx = [i for i in range(len(ytrue)) if ytrue[i] in bw and preds[i] != ytrue[i]]
rng.shuffle(corr_idx); rng.shuffle(wrong_idx)
picks = [(i, True) for i in corr_idx[:3]] + [(i, False) for i in wrong_idx[:3]]

# ---- M14 for the "M16 mel" (generated mel from powerline) ----
# evict M20's config/models from the import cache so M14's versions load (name clash)
for _m in ('config', 'models'):
    sys.modules.pop(_m, None)
sys.path.insert(0, '<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen')
cfg14 = importlib.import_module('config').CFG
m14 = importlib.import_module('models').build(cfg14).to(dev)
ck14 = torch.load('<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen/'
                  'outputs/cluster_a100/last.pt', map_location=dev, weights_only=False)
m14.load_state_dict(ck14.get('ema', ck14['model'])); m14.eval()
m_mean, m_std = ck14['mel_mean'], ck14['mel_std']
plc = {}

def m14_mel(ch, s, e):
    if ch not in plc:
        plc[ch] = np.memmap(f'{BIN}/{ch}.bin', dtype=np.float32, mode='r')
    lag = D.read_lag_s(ch)
    a = int((s + lag) * CFG.cap_sr); raw = np.array(plc[ch][a:a + int(cfg14.win_s * CFG.cap_sr)], np.float32)
    raw = np.pad(raw, (0, max(0, int(cfg14.win_s * CFG.cap_sr) - len(raw))))
    xr = resample_poly(raw, cfg14.in_sr, CFG.cap_sr).astype(np.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        g = m14.sample(torch.from_numpy(xr)[None].to(dev))[0].float().cpu() * m_std + m_mean
    fr = max(6, int(round((e - s) * cfg14.fps)))
    return g[:, :fr].numpy()

# ---- plot ----
fig, ax = plt.subplots(6, 2, figsize=(9, 13), dpi=130,
                       gridspec_kw={'width_ratios': [1, 1.1]})
for r, (i, correct) in enumerate(picks):
    ch, s, e, y = te_items[i]
    tw, pw = words[ytrue[i]], words[preds[i]]
    env = np.asarray(te_ds.X[i], np.float32)
    contour = np.log(np.exp(env).mean(0) + 1e-9)          # per-frame loudness (model's input)
    mel = m14_mel(ch, s, e)
    ax[r, 0].plot(contour, color='#2c7fb8', lw=2); ax[r, 0].set_yticks([]); ax[r, 0].set_xticks([])
    ax[r, 0].set_ylabel(('✓ ' if correct else '✗ ') + f'true="{tw}"\npred="{pw}"',
                        fontsize=11, color=('#1a7a3a' if correct else '#b02020'))
    ax[r, 1].imshow(mel, origin='lower', aspect='auto', cmap='magma'); ax[r, 1].set_xticks([]); ax[r, 1].set_yticks([])
    if r == 0:
        ax[r, 0].set_title('powerline ENVELOPE (classifier input)', fontsize=11)
        ax[r, 1].set_title('M16 mel (M14-generated from powerline)', fontsize=11)
    if r == 2:
        ax[r, 0].annotate('— correct —', xy=(0, -0.25), xycoords='axes fraction', color='#1a7a3a')
fig.suptitle('Word classification examples: 3 correct (top) + 3 incorrect (bottom)\n'
             'envelope carries the signal; the generated mel is uninformative', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(OUT_DIR, 'analysis_examples.png')
fig.savefig(out, bbox_inches='tight'); print('[saved]', out)
print('best-detected words:', [(words[c], round(acc[c], 2)) for c in best_words])
for i, correct in picks:
    print(f'  {"OK " if correct else "ERR"} true={words[ytrue[i]]:10} pred={words[preds[i]]:10} ({te_items[i][0]})')
