"""
model.py
--------
Cascade model: pretrained bandpass U-Net -> CTC encoder.

The U-Net architecture matches UNet/model_bandpass.py exactly so that
pretrained weights from checkpoints_bp/best_model.pt load cleanly.
The CTC encoder matches E2E/model.py so that results are directly comparable.

Training modes:
  phase1: UNet frozen, only CTC encoder trains.
  phase2: Both UNet and CTC train end-to-end (lower LR on UNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Vocabulary (same as E2E) ────────────────────────────────────────────────

VOCAB = {" ": 1, "'": 28}
for _i, _c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    VOCAB[_c] = _i + 2
VOCAB_SIZE = 29
BLANK_IDX = 0
IDX_TO_CHAR = {v: k for k, v in VOCAB.items()}
IDX_TO_CHAR[BLANK_IDX] = ""


# ─── Bandpass U-Net (matches UNet/model_bandpass.py exactly) ─────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.1):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d((1, 2))  # pool along time axis only

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=(1, 2), mode='bilinear',
                                align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [0, diff])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BandpassUNet(nn.Module):
    """
    Exact replica of UNet/model_bandpass.py PowerlineUNet.
    Input/Output: (B, 1, 80, T)
    """

    def __init__(self, base_ch=64):
        super().__init__()
        self.enc1 = DownBlock(1,        base_ch)
        self.enc2 = DownBlock(base_ch,  base_ch*2)
        self.enc3 = DownBlock(base_ch*2, base_ch*4)
        self.enc4 = DownBlock(base_ch*4, base_ch*8)

        self.bottleneck = nn.Sequential(
            ResBlock(base_ch*8),
            ResBlock(base_ch*8),
        )

        self.dec4 = UpBlock(base_ch*8,  base_ch*8,  base_ch*4)
        self.dec3 = UpBlock(base_ch*4,  base_ch*4,  base_ch*2)
        self.dec2 = UpBlock(base_ch*2,  base_ch*2,  base_ch)
        self.dec1 = UpBlock(base_ch,    base_ch,    base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b = self.bottleneck(x4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1) + x  # global residual


# ─── CTC Encoder (matches E2E/model.py exactly) ─────────────────────────────

class CTCEncoder(nn.Module):
    """
    Predicted mel -> character logits via conv downsampling + bidirectional GRU.

    Input : (B, 1, 80, T)
    Output: (T', B, V)  where T' = T//4   (CTC format)
    """

    def __init__(self, n_mels=80, hidden=256, n_layers=2,
                 vocab_size=VOCAB_SIZE, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        conv_out_freq = n_mels // 4
        gru_in = 32 * conv_out_freq

        self.gru = nn.GRU(
            gru_in, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        c = self.conv(x)                          # (B, 32, 20, T//4)
        B, C, M, T = c.shape
        c = c.permute(0, 3, 1, 2).reshape(B, T, C * M)  # (B, T, 640)
        out, _ = self.gru(c)                       # (B, T, 2*hidden)
        out = self.dropout(out)
        logits = self.linear(out)                  # (B, T, V)
        return logits.permute(1, 0, 2)             # (T, B, V)


# ─── Cascade Model ──────────────────────────────────────────────────────────

class CascadeModel(nn.Module):
    """
    Pretrained BandpassUNet (denoiser) -> CTCEncoder (transcriber).

    Supports two training phases:
      phase1: UNet frozen, only CTC trains.
      phase2: Both train end-to-end.
    """

    def __init__(self, unet_base_ch=64, gru_hidden=256, gru_layers=2,
                 dropout=0.2):
        super().__init__()
        self.unet = BandpassUNet(base_ch=unet_base_ch)
        self.encoder = CTCEncoder(hidden=gru_hidden, n_layers=gru_layers,
                                  dropout=dropout)

    def freeze_unet(self):
        for p in self.unet.parameters():
            p.requires_grad = False
        self.unet.eval()

    def unfreeze_unet(self):
        for p in self.unet.parameters():
            p.requires_grad = True
        self.unet.train()

    def forward(self, noisy):
        predicted = self.unet(noisy)              # (B, 1, 80, T)
        ctc_logits = self.encoder(predicted)      # (T', B, V)
        return predicted, ctc_logits

    def train(self, mode=True):
        """Override to keep UNet in eval when frozen."""
        super().train(mode)
        if not any(p.requires_grad for p in self.unet.parameters()):
            self.unet.eval()
        return self


# ─── Loss ────────────────────────────────────────────────────────────────────

class MSTFTLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, pred, target):
        loss = 0.0
        for s in self.scales:
            p = F.avg_pool2d(pred,   (1, s), (1, s)) if s > 1 else pred
            t = F.avg_pool2d(target, (1, s), (1, s)) if s > 1 else target
            sc = torch.norm(t - p, p='fro') / (torch.norm(t, p='fro') + 1e-8)
            l1 = F.l1_loss(p, t)
            loss = loss + sc + l1
        return loss / len(self.scales)


class CascadeLoss(nn.Module):
    """
    CTC loss + optional reconstruction loss for fine-tuning phase.

    total = beta * CTC  +  alpha * (L1 + MSTFT)
    Set alpha=0 for phase 1 (CTC only).
    """

    def __init__(self, alpha=0.0, beta=1.0, blank_idx=BLANK_IDX):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.mstft = MSTFTLoss()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean',
                                   zero_infinity=True)

    def forward(self, predicted, target_mel, ctc_logits, text, text_len):
        # CTC
        t_out = ctc_logits.shape[0]
        bsz   = ctc_logits.shape[1]
        log_probs  = F.log_softmax(ctc_logits, dim=-1)
        input_lens = torch.full((bsz,), t_out, dtype=torch.long,
                                device=ctc_logits.device)

        targets_flat = []
        for i in range(bsz):
            tl = int(text_len[i].item())
            if tl > 0:
                targets_flat.append(text[i, :tl])
        if targets_flat:
            targets_cat = torch.cat(targets_flat)
            ctc = self.ctc_loss(log_probs, targets_cat, input_lens, text_len)
        else:
            ctc = torch.zeros((), device=ctc_logits.device)

        # Reconstruction (optional)
        recon_l1 = torch.tensor(0.0, device=predicted.device)
        recon_mstft = torch.tensor(0.0, device=predicted.device)
        if self.alpha > 0:
            recon_l1    = F.l1_loss(predicted, target_mel)
            recon_mstft = self.mstft(predicted, target_mel)

        total = self.beta * ctc + self.alpha * (recon_l1 + recon_mstft)
        return total, ctc.item(), recon_l1.item(), recon_mstft.item()


# ─── Weight loading helper ───────────────────────────────────────────────────

def load_pretrained_unet(model, unet_ckpt_path, device='cpu'):
    """
    Load pretrained bandpass UNet weights into the cascade model.
    The checkpoint from UNet/checkpoints_bp/ stores the full model state
    under 'model_state'.
    """
    ckpt = torch.load(unet_ckpt_path, map_location=device, weights_only=False)
    unet_state = ckpt['model_state']
    model.unet.load_state_dict(unet_state)
    print(f"Loaded pretrained UNet from {unet_ckpt_path} "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.6f})")
    return model


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    m = CascadeModel()
    n_unet = sum(p.numel() for p in m.unet.parameters())
    n_ctc  = sum(p.numel() for p in m.encoder.parameters())
    print(f"UNet params:  {n_unet:,}")
    print(f"CTC params:   {n_ctc:,}")
    print(f"Total params: {n_unet + n_ctc:,}")

    x = torch.randn(2, 1, 80, 498)
    pred, logits = m(x)
    print(f"Input:   {x.shape}")
    print(f"Pred:    {pred.shape}")
    print(f"Logits:  {logits.shape}  (T'={logits.shape[0]}, B={logits.shape[1]}, V={logits.shape[2]})")
