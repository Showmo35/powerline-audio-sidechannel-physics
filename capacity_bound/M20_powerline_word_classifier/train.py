#!/usr/bin/env python3
"""
train.py — train the powerline word classifier; report accuracy vs chance.

Key metric = MACRO accuracy (mean per-word recall) so "all frequent words detected"
is what's measured, not just the common ones. Chance = 1/K and majority-class.
Optional --audio-control: wav2vec2 linear probe on the same words = the ceiling.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, OUT_DIR
import dataset as D
import models as M


def macro_acc(y_true, y_pred, K):
    per = []
    for c in range(K):
        m = y_true == c
        if m.sum() > 0:
            per.append((y_pred[m] == c).mean())
    return float(np.mean(per)), np.array(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--audio-control', action='store_true')
    ap.add_argument('--input', choices=('wide', 'env'), default='wide')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.audio_control:
        tr_items, words = D.build_items(split='train'); te_items, _ = D.build_items(split='test')
        K = len(words)
        tr_dl = torch.utils.data.DataLoader(D.WordWindows(tr_items, want_audio=True), batch_size=args.batch,
                                            shuffle=False, collate_fn=D.collate, num_workers=args.workers)
        te_dl = torch.utils.data.DataLoader(D.WordWindows(te_items, want_audio=True), batch_size=args.batch,
                                            shuffle=False, collate_fn=D.collate, num_workers=4)
        audio_linear_probe(tr_dl, te_dl, K, words, device)
        return

    words = D.cache_words(); K = len(words)
    fl = (args.input == 'env')
    tr_ds = D.CachedWords('train', flatten_freq=fl); te_ds = D.CachedWords('test', flatten_freq=fl)
    print(f'[input] {args.input}{"  (envelope-only: spectrum removed)" if fl else ""}', flush=True)
    ytr, yte = tr_ds.y, te_ds.y
    print(f'[data] K={K} words, train={len(tr_ds)} test={len(te_ds)}', flush=True)
    counts = np.bincount(yte, minlength=K)
    print(f'[chance] 1/K={100/K:.1f}%  majority-class={100*counts.max()/counts.sum():.1f}%', flush=True)
    mean, std = D.stats(tr_ds)
    print(f'[stats] wide mean={mean:.2f} std={std:.2f}', flush=True)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=D.collate,
                                        num_workers=args.workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=args.batch, shuffle=False, collate_fn=D.collate,
                                        num_workers=4)

    model = M.build(K).to(device)
    with torch.no_grad():                                   # init LazyLinear on-device
        _ = model(torch.zeros(2, CFG.n_wbins, CFG.n_frames, device=device))
    print(f'[model] params={model.count_params():,}', flush=True)
    cls_w = torch.tensor(1.0 / (np.bincount(ytr, minlength=K) + 1), dtype=torch.float32)
    cls_w = (cls_w / cls_w.mean()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    best = 0.0; log = []
    for ep in range(args.epochs):
        model.train()
        for b in tr_dl:
            x = ((b['wide'] - mean) / std).to(device); y = b['y'].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                loss = F.cross_entropy(model(x), y, weight=cls_w)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        model.eval(); yt, yp = [], []
        with torch.no_grad():
            for b in te_dl:
                x = ((b['wide'] - mean) / std).to(device)
                yp.append(model(x).argmax(1).cpu().numpy()); yt.append(b['y'].numpy())
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        macro, per = macro_acc(yt, yp, K)
        overall = float((yt == yp).mean())
        log.append({'epoch': ep, 'macro_acc': macro, 'overall_acc': overall})
        print(f'[eval] ep{ep}  macro_acc={macro*100:.1f}%  overall={overall*100:.1f}%  '
              f'(chance 1/K={100/K:.1f}%)', flush=True)
        if macro > best:
            best = macro
            order = np.argsort(-per)
            print('   best per-word acc: ' + ', '.join(f'{words[i]}={per[i]*100:.0f}' for i in order[:12]), flush=True)
            torch.save({'model': model.state_dict(), 'words': words, 'mean': mean, 'std': std},
                       os.path.join(args.out_dir, 'best.pt'))
        json.dump(log, open(os.path.join(args.out_dir, 'log.json'), 'w'), indent=1)
    print(f'[done] best macro_acc={best*100:.1f}%  (chance {100/K:.1f}%)', flush=True)


@torch.no_grad()
def audio_linear_probe(tr_dl, te_dl, K, words, device):
    """wav2vec2 mean features -> logistic regression. The audio ceiling."""
    import torchaudio
    w2v = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(device).eval()

    def feats(dl):
        X, Y = [], []
        for b in dl:
            f, _ = w2v.extract_features(b['audio'].to(device))
            X.append(f[-1].mean(1).float().cpu().numpy()); Y.append(b['y'].numpy())
        return np.concatenate(X), np.concatenate(Y)
    Xtr, Ytr = feats(tr_dl); Xte, Yte = feats(te_dl)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, Ytr)
    yp = clf.predict(Xte)
    macro, per = macro_acc(Yte, yp, K)
    print(f'[AUDIO CONTROL wav2vec2 probe] macro_acc={macro*100:.1f}%  overall={100*(yp==Yte).mean():.1f}%  '
          f'(chance {100/K:.1f}%)', flush=True)


if __name__ == '__main__':
    main()
