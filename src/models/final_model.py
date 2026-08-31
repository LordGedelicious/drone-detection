"""
FinalDetector -- the submitted "best model": an improvement of p2_v2.

p2_v2 = P2GranularDetector backbone/neck + decoupled DFL heads at P2..P5.
Three data-driven changes here:

1. **Drop the P5 head.** The largest drone in the corpus is 0.136 of the image
   (~87 px @640); the stride-32 P5 grid never receives a positive and only
   contributes false positives. FinalDetector detects at P2/P3/P4
   (strides 4/8/16) only. The neck still computes P5 internally for the
   top-down pathway.
2. **Squeeze-and-Excitation on each neck output** (Hu et al., 2018) -- cheap
   channel re-weighting before the head. One SE block per level (~0.02 M params);
   the heavy-attention AB2D reference showed diminishing returns from large
   attention blocks.
3. Trained with :class:`TaskAlignedLoss` (src/core/loss.py): a dynamic
   task-aligned assigner replaces the static centre+neighbour rule.

Head output per cell, channels-last: ``[obj(1), cls(C), box(4*(reg_max+1))]``
(same layout as the v2 head, so ``decode_predictions_v2`` decodes it).
"""

import torch
import torch.nn as nn

from src.models.base import BaseDetector
from src.models.v2 import DecoupledDFLHead


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))            # (B, C) global average pool
        s = self.fc(s)
        return x * s[:, :, None, None]


class FinalDetector(BaseDetector):
    # detection levels: P2, P3, P4  (P5 head dropped)
    strides = (4, 8, 16)

    def __init__(self, base, num_classes: int = 1, reg_max: int = 16, use_se: bool = True):
        super().__init__(num_classes=num_classes)
        for name in [n for n, _ in list(base.named_children()) if n.startswith("head")]:
            delattr(base, name)
        self.base = base
        self.reg_max = reg_max
        self.n_levels = 3
        chans = list(base.neck_channels[:3])          # P2, P3, P4
        self.neck_channels = chans
        self.se = nn.ModuleList(SEBlock(c) for c in chans) if use_se else None
        self.heads = nn.ModuleList(
            DecoupledDFLHead(c, num_classes, reg_max) for c in chans
        )

    def neck_forward(self, x: torch.Tensor) -> list:
        return self.base.neck_forward(x)[:3]          # P2, P3, P4 (P5 computed but unused)

    def forward(self, x: torch.Tensor) -> list:
        feats = self.neck_forward(x)
        if self.se is not None:
            feats = [se(f) for se, f in zip(self.se, feats)]
        return [head(f) for head, f in zip(self.heads, feats)]
