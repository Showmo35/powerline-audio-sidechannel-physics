"""
model.py
--------
Module 4 Soundbar V2: Perception-Aware UNet for Soundbar reconstruction.

Improvements over Module4_UNet_Soundbar:
  - CTC Perceptual Loss replaces plain L1+MSTFT (PerceptionAwareLoss)
  - base_ch=64 (was 48) for full model capacity

Contains:
  - PowerlineUNet: Spectrogram U-Net  (B, 1, 80, T) -> (B, 1, 80, T)
  - CTCPerceptualEncoder: Frozen pretrained CTC encoder as perceptual extractor
  - MultiScaleSTFTLoss: Multi-resolution spectral loss
  - PerceptionAwareLoss: L1 + MSTFT + CTC perceptual cosine-similarity loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building blocks
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dropout: float = 0.1):
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
    def __init__(self, ch: int, dropout: float = 0.1):
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
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d((1, 2))

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, [0, diff])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =============================================================================
# PowerlineUNet
# =============================================================================

class PowerlineUNet(nn.Module):
    """
    Spectrogram U-Net.  Input/Output: (B, 1, 80, T)

    Encoder:    4 DownBlocks (1 -> base_ch -> base_ch*2 -> base_ch*4 -> base_ch*8)
    Bottleneck: 2 ResBlocks
    Decoder:    4 UpBlocks with skip connections
    Global residual: output = conv(decoded) + input
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.enc1 = DownBlock(1,          base_ch)
        self.enc2 = DownBlock(base_ch,    base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 8),
            ResBlock(base_ch * 8),
        )

        self.dec4 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.dec3 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = UpBlock(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = UpBlock(base_ch,     base_ch,     base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b  = self.bottleneck(x4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1) + x

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# CTCPerceptualEncoder - frozen CTC encoder as perceptual feature extractor
# =============================================================================

class CTCPerceptualEncoder(nn.Module):
    """
    Frozen pretrained CTC encoder used as a perceptual feature extractor.

    Extracts multi-scale features (conv stem + BiGRU) and concatenates them.
    All parameters are frozen (requires_grad=False).

    Input:  (B, 1, M, T)
    Output: (B, T//4, 640 + hidden*2)
    """

    def __init__(self, n_mels: int = 80, hidden: int = 256, n_layers: int = 2,
                 vocab_size: int = 29, dropout: float = 0.2):
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
        self.hidden  = hidden

    def extract_features(self, x):
        """
        Multi-scale perceptual features for loss computation.

        Returns concatenation of conv-stem and GRU features:
          (B, T//4, 640 + hidden*2)
        """
        c = self.conv(x)
        B, C, M, T = c.shape
        conv_feat = c.permute(0, 3, 1, 2).reshape(B, T, C * M)
        gru_feat, _ = self.gru(conv_feat)
        return torch.cat([conv_feat, gru_feat], dim=-1)


# =============================================================================
# MultiScaleSTFTLoss
# =============================================================================

class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def _loss_at_scale(self, pred, target, scale):
        if scale > 1:
            pred   = F.avg_pool2d(pred,   kernel_size=scale)
            target = F.avg_pool2d(target, kernel_size=scale)
        sc = torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + 1e-8)
        l1 = F.l1_loss(pred, target)
        return sc + l1

    def forward(self, pred, target):
        loss = 0.0
        for s in self.scales:
            loss = loss + self._loss_at_scale(pred, target, s)
        return loss / len(self.scales)


# =============================================================================
# PerceptionAwareLoss
# =============================================================================

class PerceptionAwareLoss(nn.Module):
    """
    Combined loss: L1 + Multi-Scale STFT + CTC Perceptual.

    The perceptual term uses a FROZEN pretrained CTC encoder to extract
    intermediate features from both predicted and clean spectrograms,
    then penalises their cosine distance.

    Total = l1_w * L1 + mstft_w * MSTFT + percep_w * (1 - cos_sim)
    """

    def __init__(self, ctc_ckpt_path: str, l1_weight: float = 1.0,
                 mstft_weight: float = 1.0, percep_weight: float = 0.1,
                 n_mels: int = 80, hidden: int = 256, n_layers: int = 2):
        super().__init__()
        self.l1_w    = l1_weight
        self.mstft_w = mstft_weight
        self.percep_w = percep_weight

        self.mstft = MultiScaleSTFTLoss(scales=(1, 2, 4))

        self.ctc_encoder = CTCPerceptualEncoder(
            n_mels=n_mels, hidden=hidden, n_layers=n_layers, dropout=0.0)
        self._load_ctc_checkpoint(ctc_ckpt_path)
        self._freeze_ctc()

    def _load_ctc_checkpoint(self, ckpt_path: str):
        print(f"  Loading CTC perceptual encoder from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        missing, unexpected = self.ctc_encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  WARNING: missing keys in CTC encoder: {missing}")
        if unexpected:
            print(f"  INFO: unexpected keys (ignored): {unexpected}")
        print("  CTC encoder loaded successfully.")

    def _freeze_ctc(self):
        for param in self.ctc_encoder.parameters():
            param.requires_grad = False
        self.ctc_encoder.eval()
        n = sum(p.numel() for p in self.ctc_encoder.parameters())
        print(f"  CTC encoder frozen ({n:,} params)")

    def forward(self, pred, target):
        l1    = F.l1_loss(pred, target)
        mstft = self.mstft(pred, target)

        with torch.no_grad():
            self.ctc_encoder.eval()
            clean_feat = self.ctc_encoder.extract_features(target)

        self.ctc_encoder.train()
        pred_feat = self.ctc_encoder.extract_features(pred)

        perceptual = 1.0 - F.cosine_similarity(pred_feat, clean_feat, dim=-1).mean()

        loss = self.l1_w * l1 + self.mstft_w * mstft + self.percep_w * perceptual
        return loss, l1.item(), mstft.item(), perceptual.item()


if __name__ == "__main__":
    gen = PowerlineUNet(base_ch=64)
    print(f"PowerlineUNet params: {gen.count_params():,}")
    noisy = torch.randn(4, 1, 80, 98)
    pred  = gen(noisy)
    print(f"Input shape : {noisy.shape}")
    print(f"Output shape: {pred.shape}")
