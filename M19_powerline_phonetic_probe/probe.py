#!/usr/bin/env python3
"""
probe.py — regress the true audio mel from a powerline feature family and report
how much PHONETIC (envelope-removed) spectral shape is recoverable.

  --feature wide  : wideband 200 kHz log-STFT  (contains harmonics + AM sidebands)
  --feature env   : per-frame loudness only     (baseline; env-removed r ≈ 0)

Metrics on held-out chunks:
  mel_r        full log-mel Pearson (folds in the envelope — the old number)
  env_r        loudness only: corr of per-frame MEAN energy
  PHON_r       ENVELOPE-REMOVED: corr after subtracting each frame's mean →
               the fraction of audio SPECTRAL SHAPE (phonetics) recovered ← key
  perbin_r     mean per-mel-bin corr of the envelope-removed prediction
A shuffled-pairing control gives the PHON_r chance floor.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, OUT_DIR
import dataset as D


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return float((a * b).sum().item() / d) if d > 0 else 0.0


class Probe(nn.Module):
    """Depthwise-ish temporal conv stack: [B, Cin, T] → [B, n_mels, T]."""
    def __init__(self, c_in, cfg=CFG, width=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, cfg.n_mels, 1))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, dl, device, mmean, mstd):
    model.eval()
    full = env = phon = 0.0; perbin = np.zeros(CFG.n_mels); nb = 0
    P_all, M_all = [], []
    for b in dl:
        x = b[KEY].to(device); mel = b['mel']
        pred = (model(x).cpu() * mstd + mmean)
        for i in range(pred.shape[0]):
            P, M = pred[i], mel[i]                                  # [n_mels, T]
            full += _corr(P.flatten(), M.flatten())
            env += _corr(P.mean(0), M.mean(0))
            Pc, Mc = P - P.mean(0, keepdim=True), M - M.mean(0, keepdim=True)
            phon += _corr(Pc.flatten(), Mc.flatten())
            for k in range(CFG.n_mels):
                perbin[k] += _corr(Pc[k], Mc[k])
            nb += 1
        P_all.append(pred); M_all.append(mel)
    # shuffled-pairing chance floor for PHON_r (static average-speech-shape baseline),
    # averaged over several permutations for stability
    P = torch.cat(P_all); M = torch.cat(M_all)
    Pc_all = P - P.mean(1, keepdim=True); Mc_all = M - M.mean(1, keepdim=True)
    chance = 0.0
    for _ in range(5):
        perm = torch.randperm(len(P))
        for i in range(len(P)):
            chance += _corr(Pc_all[perm[i]].flatten(), Mc_all[i].flatten())
    chance /= (5 * len(P))
    phon = phon / nb
    return {'mel_r': full / nb, 'env_r': env / nb, 'phon_r': phon,
            'phon_r_chance': chance, 'phon_signal': phon - chance,
            'perbin_phon_r': float(perbin.mean() / nb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature', choices=('wide', 'env'), required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--max-windows', type=int, default=0, help='smoke cap on window count')
    args = ap.parse_args()
    global KEY; KEY = args.feature
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir or os.path.join(OUT_DIR, f'probe_{args.feature}')
    os.makedirs(out_dir, exist_ok=True)

    tr_ch, te_ch = D.split_chunks()
    tr_idx, te_idx = D.build_index(tr_ch, seed=0), D.build_index(te_ch, seed=1)
    if args.max_windows:
        tr_idx, te_idx = tr_idx[:args.max_windows], te_idx[:max(16, args.max_windows // 4)]
    tr_ds = D.ProbeWindows(tr_idx)
    te_ds = D.ProbeWindows(te_idx)
    print(f'[data] feature={args.feature}  train_win={len(tr_ds)} test_win={len(te_ds)}', flush=True)
    mmean, mstd = D.stats(tr_ds, 'mel'); fmean, fstd = D.stats(tr_ds, KEY)
    mmean, mstd, fmean, fstd = map(lambda v: torch.tensor(v), (mmean, mstd, fmean, fstd))
    print(f'[stats] mel {float(mmean):.2f}/{float(mstd):.2f}  {KEY} {float(fmean):.2f}/{float(fstd):.2f}', flush=True)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                                        collate_fn=D.collate, num_workers=args.workers,
                                        pin_memory=True, drop_last=True, persistent_workers=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                                        collate_fn=D.collate, num_workers=4)

    c_in = CFG.n_wbins if KEY == 'wide' else 1
    model = Probe(c_in).to(device)
    print(f'[model] probe params={sum(p.numel() for p in model.parameters()):,}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    log = []
    for ep in range(args.epochs):
        model.train()
        for b in tr_dl:
            x = ((b[KEY] - fmean) / fstd).to(device)
            y = ((b['mel'] - mmean) / mstd).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                loss = F.mse_loss(model(x), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        # eval needs feature-normalized input too
        class Wrapped(nn.Module):
            def __init__(s): super().__init__(); s.m = model
            def forward(s, x): return s.m((x - fmean.to(x.device)) / fstd.to(x.device))
        m = evaluate(Wrapped(), te_dl, device, mmean, mstd)
        m['epoch'] = ep; log.append(m)
        print(f'[eval] ep{ep}  mel_r={m["mel_r"]:.3f}  env_r={m["env_r"]:.3f}  '
              f'PHON_signal={m["phon_signal"]:+.3f} (PHON_r={m["phon_r"]:.3f} - chance {m["phon_r_chance"]:.3f})  '
              f'perbin={m["perbin_phon_r"]:.3f}', flush=True)
        json.dump(log, open(os.path.join(out_dir, 'log.json'), 'w'), indent=1)
    print(f'[done] feature={args.feature}  final PHON_signal={log[-1]["phon_signal"]:+.3f} '
          f'perbin={log[-1]["perbin_phon_r"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
