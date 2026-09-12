"""
prepare_data.py
---------------
Module 9: Prepare UNet-enhanced mel spectrograms for Whisper fine-tuning.

Pipeline:
  1. Load noisy log-mel spectrograms from Module 3's transcribe_data.npz
     (already bandpass-filtered, 16kHz, Whisper filterbank format)
  2. Run Module 4's Perceptual-Loss UNet to enhance them
  3. Decode encoded text labels back to strings
  4. Save as data/unet_whisper_data.npz

The enhanced mels are in exactly Whisper's expected format (80-bin log-mel,
same normalisation), so they can be fed directly to the Whisper encoder
without any further feature extraction.

Output:
    data/unet_whisper_data.npz
        mel_train  (N, 80, 498) float32  -- UNet-enhanced mels (5s clips)
        mel_val    (M, 80, 498) float32
        text_train (N,) object           -- transcript strings
        text_val   (M,) object
"""

import os, sys, argparse, importlib.util
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3_DATA = os.path.join(ROOT, 'Module3_BeamSearchLM', 'data', 'transcribe_data.npz')
M4_CKPT = os.path.join(ROOT, 'Module4_PerceptualLoss', 'checkpoints', 'best_model.pt')
M4_MODEL= os.path.join(ROOT, 'Module4_PerceptualLoss', 'model.py')

# ---------------------------------------------------------------------------
# Vocabulary (same as Module 3)
# ---------------------------------------------------------------------------
IDX_TO_CHAR = {0: '', 1: ' ', 28: "'"}
for _i, _c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    IDX_TO_CHAR[_i + 2] = _c


def decode_text(encoded, length):
    return ''.join(IDX_TO_CHAR.get(int(c), '') for c in encoded[:length]).strip()


def load_unet(ckpt_path, model_path, device):
    spec = importlib.util.spec_from_file_location('m4_model', model_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    PowerlineUNet = mod.PowerlineUNet

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet  = PowerlineUNet(base_ch=ckpt.get('args', {}).get('base_ch', 64)).to(device)
    unet.load_state_dict(ckpt['model_state'])
    unet.eval()
    print(f'  UNet loaded (epoch {ckpt.get("epoch","?")}, '
          f'val_loss={ckpt.get("val_loss", 0):.5f})')
    return unet


@torch.no_grad()
def run_unet(unet, mels_np, device, batch_size=64):
    """Run UNet on (N, 1, 80, T) noisy mels, return (N, 80, T) enhanced."""
    out = []
    for i in range(0, len(mels_np), batch_size):
        batch = torch.from_numpy(mels_np[i:i+batch_size]).float().to(device)
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                pred = unet(batch)
        else:
            pred = unet(batch)
        out.append(pred.squeeze(1).cpu().numpy())   # (B, 80, T)
        if i % 500 == 0:
            print(f'    {i+len(batch)}/{len(mels_np)} ...', flush=True)
    return np.concatenate(out, axis=0)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load Module 3 data
    print(f'\nLoading Module 3 data: {args.m3_data}')
    d = np.load(args.m3_data)
    noisy_train = d['noisy_train']          # (N, 1, 80, 498)
    noisy_val   = d['noisy_val']            # (M, 1, 80, 498)
    text_train  = d['text_train']           # (N, 200) int32
    text_val    = d['text_val']
    tlen_train  = d['text_len_train']       # (N,) int32
    tlen_val    = d['text_len_val']
    print(f'  Train: {len(noisy_train)}  Val: {len(noisy_val)}')
    print(f'  Mel shape: {noisy_train.shape[1:]}')

    # Decode text labels
    print('\nDecoding text labels ...')
    str_train = np.array([decode_text(text_train[i], tlen_train[i])
                          for i in range(len(text_train))], dtype=object)
    str_val   = np.array([decode_text(text_val[i],   tlen_val[i])
                          for i in range(len(text_val))],   dtype=object)
    print(f'  Sample train: "{str_train[0]}"')
    print(f'  Sample val  : "{str_val[0]}"')

    # Load Module 4 UNet
    print(f'\nLoading Perceptual-Loss UNet: {args.unet_ckpt}')
    unet = load_unet(args.unet_ckpt, args.unet_model, device)

    # Run UNet on train set
    print('\nEnhancing train mels ...')
    mel_train = run_unet(unet, noisy_train, device)   # (N, 80, 498)

    # Run UNet on val set
    print('\nEnhancing val mels ...')
    mel_val = run_unet(unet, noisy_val, device)       # (M, 80, 498)

    print(f'\nmel_train: {mel_train.shape}  range [{mel_train.min():.2f}, {mel_train.max():.2f}]')
    print(f'mel_val  : {mel_val.shape}  range [{mel_val.min():.2f}, {mel_val.max():.2f}]')

    # Save
    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_path,
        mel_train  = mel_train.astype(np.float32),
        mel_val    = mel_val.astype(np.float32),
        text_train = str_train,
        text_val   = str_val,
    )
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--m3_data',   default=M3_DATA)
    parser.add_argument('--unet_ckpt', default=M4_CKPT)
    parser.add_argument('--unet_model',default=M4_MODEL)
    parser.add_argument('--out_dir',   default=os.path.join(_HERE, 'data'))
    parser.add_argument('--out_name',  default='unet_whisper_data.npz')
    args = parser.parse_args()
    main(args)
