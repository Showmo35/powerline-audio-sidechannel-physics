#!/usr/bin/env python3
"""train.py — train the heavy multi-stream word encoder; macro acc vs chance / vs env-only 39%."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG, OUT_DIR
import dataset as D
import models as M


def macro_acc(yt, yp, K):
    per = [ (yp[yt == c] == c).mean() for c in range(K) if (yt == c).sum() > 0 ]
    return float(np.mean(per)), np.array(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--amp', choices=('bf16', 'fp16'), default='bf16')  # fp16 for V100/<cluster_name>
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    amp_dtype = torch.float16 if args.amp == 'fp16' else torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    words = D.cache_words(); K = len(words)
    tr_ds, te_ds = D.CachedRaw('train'), D.CachedRaw('test')
    ytr, yte = tr_ds.y, te_ds.y
    counts = np.bincount(yte, minlength=K)
    print(f'[data] K={K} train={len(tr_ds)} test={len(te_ds)}  '
          f'chance 1/K={100/K:.1f}%  majority={100*counts.max()/counts.sum():.1f}%', flush=True)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=D.collate,
                                        num_workers=args.workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=args.batch, shuffle=False, collate_fn=D.collate,
                                        num_workers=4)

    model = M.build(K).to(device)
    print(f'[model] HeavyWordNet params={model.count_params():,}', flush=True)
    cls_w = torch.tensor(1.0 / (np.bincount(ytr, minlength=K) + 1), dtype=torch.float32)
    cls_w = (cls_w / cls_w.mean()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    total = args.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.08)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    best = 0.0; log = []; step = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for b in tr_dl:
            raw = b['raw'].to(device, non_blocking=True); y = b['y'].to(device)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
                loss = F.cross_entropy(model(raw), y, weight=cls_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 100 == 0:
                print(f'  ep{ep} step{step} loss={loss.item():.3f} ({time.time()-t0:.0f}s)', flush=True)
        model.eval(); yp = []
        with torch.no_grad():
            for b in te_dl:
                with torch.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
                    yp.append(model(b['raw'].to(device)).float().argmax(1).cpu().numpy())
        yp = np.concatenate(yp)
        macro, per = macro_acc(yte, yp, K)
        overall = float((yte == yp).mean())
        log.append({'epoch': ep, 'macro_acc': macro, 'overall_acc': overall})
        print(f'[eval] ep{ep}  macro_acc={macro*100:.1f}%  overall={overall*100:.1f}%  '
              f'(chance {100/K:.1f}%, env-only baseline 39%)', flush=True)
        if macro > best:
            best = macro
            order = np.argsort(-per)
            print('   best per-word: ' + ', '.join(f'{words[i]}={per[i]*100:.0f}' for i in order[:12]), flush=True)
            torch.save({'model': model.state_dict(), 'words': words}, os.path.join(args.out_dir, 'best.pt'))
        json.dump(log, open(os.path.join(args.out_dir, 'log.json'), 'w'), indent=1)
    print(f'[done] best macro_acc={best*100:.1f}%  (chance {100/K:.1f}%, env-only 39%)', flush=True)


if __name__ == '__main__':
    main()
