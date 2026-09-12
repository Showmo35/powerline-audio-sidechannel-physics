#!/usr/bin/env python3
"""model_unet.py — small 2D U-Net: [B, 2*n_harm, 80, T] → [B, 80, T] clean log-mel."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class MelUNet(nn.Module):
    def __init__(self, in_ch, base=48):
        super().__init__()
        self.e1 = conv_block(in_ch, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.bott = conv_block(base * 4, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.d3 = conv_block(base * 4 + base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 2 + base * 2, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.d1 = conv_block(base + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _match(x, ref):
        # exact-match x to ref's spatial size: pad if smaller, crop if larger
        dh = ref.shape[-2] - x.shape[-2]
        dw = ref.shape[-1] - x.shape[-1]
        if dh > 0 or dw > 0:
            x = F.pad(x, (0, max(dw, 0), 0, max(dh, 0)))
        return x[..., :ref.shape[-2], :ref.shape[-1]]

    def forward(self, x):                      # x: [B, C, M, T]
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self._match(self.up3(b), e3), e3], 1))
        d2 = self.d2(torch.cat([self._match(self.up2(d3), e2), e2], 1))
        d1 = self.d1(torch.cat([self._match(self.up1(d2), e1), e1], 1))
        return self.out(d1).squeeze(1)         # [B, M, T]
