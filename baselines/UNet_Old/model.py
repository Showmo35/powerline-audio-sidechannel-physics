"""
model.py
--------
Spectrogram U-Net for powerline-to-audio reconstruction.

Architecture:
  Encoder: 4 conv-blocks that downsample along the time axis
  Bottleneck: 2 residual blocks
  Decoder: 4 up-conv blocks with skip connections from encoder
  Output: same shape as input (1, N_MELS, T)

Input/Output shape: (batch, 1, N_MELS=64, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d -> BN -> LeakyReLU  (x2)"""
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
    """Residual block for bottleneck."""
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
    """ConvBlock + MaxPool."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d((1, 2))   # pool along time axis only

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    """Bilinear upsample + ConvBlock + skip concatenation."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=(1, 2), mode='bilinear',
                                align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes differ (due to odd time lengths)
        if x.shape != skip.shape:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [0, diff])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── U-Net ────────────────────────────────────────────────────────────────────

class PowerlineUNet(nn.Module):
    """
    Spectrogram U-Net for audio reconstruction from powerline EM.

    Input : (B, 1, 64, T)  – noisy powerline log-mel spectrogram
    Output: (B, 1, 64, T)  – reconstructed clean log-mel spectrogram
    """

    def __init__(self, base_ch=32):
        super().__init__()

        # Encoder
        self.enc1 = DownBlock(1,        base_ch)       # /2 time
        self.enc2 = DownBlock(base_ch,  base_ch*2)     # /4
        self.enc3 = DownBlock(base_ch*2, base_ch*4)    # /8
        self.enc4 = DownBlock(base_ch*4, base_ch*8)    # /16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(base_ch*8),
            ResBlock(base_ch*8),
        )

        # Decoder
        self.dec4 = UpBlock(base_ch*8,  base_ch*8,  base_ch*4)
        self.dec3 = UpBlock(base_ch*4,  base_ch*4,  base_ch*2)
        self.dec2 = UpBlock(base_ch*2,  base_ch*2,  base_ch)
        self.dec1 = UpBlock(base_ch,    base_ch,    base_ch)

        # Output head
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        # Encoder
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Combined loss ─────────────────────────────────────────────────────────────

class SpectralLoss(nn.Module):
    """
    L1 loss + Multi-resolution spectral convergence loss.
    Encourages both global and local spectral accuracy.
    """
    def __init__(self, l1_weight=0.8, sc_weight=0.2):
        super().__init__()
        self.l1_w  = l1_weight
        self.sc_w  = sc_weight

    def spectral_convergence(self, pred, target):
        diff = torch.norm(target - pred, p='fro')
        base = torch.norm(target, p='fro') + 1e-8
        return diff / base

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        sc = self.spectral_convergence(pred, target)
        return self.l1_w * l1 + self.sc_w * sc, l1.item(), sc.item()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = PowerlineUNet(base_ch=32)
    print(f"Parameters : {model.count_params():,}")
    x = torch.randn(4, 1, 64, 172)   # batch=4, mel=64, T=172 (~1 sec)
    y = model(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    loss_fn = SpectralLoss()
    loss, l1, sc = loss_fn(y, x)
    print(f"Loss: {loss:.4f}  (L1={l1:.4f}, SC={sc:.4f})")
