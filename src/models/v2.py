"""
V2 refinement heads for the from-scratch detectors.

Diagnosis from the bake-off: the from-scratch models match a pretrained YOLO on
detection (mAP@0.5) but trail on localisation (mAP@0.5:0.95). The regression head
is the weak link -- one conv, 4 raw offsets, an implicit Dirac-delta target.

V2 keeps each base model's backbone + neck and replaces the heads with:

  * a **decoupled head** -- separate 3x3 conv branches for objectness/class and
    for box regression (YOLOX, Ge et al. 2021, arXiv:2107.08430);
  * a **DFL box branch** -- each of the 4 box edges (l, t, r, b distances from the
    cell centre, in cell units) is predicted as a softmax distribution over
    `reg_max + 1` bins and decoded as its expectation (Generalized Focal Loss,
    Li et al. NeurIPS 2020).

Per-cell output, channels-last: ``[obj(1), cls(num_classes), box(4*(reg_max+1))]``.
"""

import math

import torch
import torch.nn as nn

from src.models.base import BaseDetector, ConvBNAct


class DecoupledDFLHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 1, reg_max: int = 16, hidden: int = None):
        super().__init__()
        h = hidden or max(in_ch // 2, 64)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.n_box = 4 * (reg_max + 1)

        self.stem = ConvBNAct(in_ch, h, kernel_size=1, stride=1, padding=0)
        self.cls_branch = nn.Sequential(ConvBNAct(h, h, 3, 1, 1), ConvBNAct(h, h, 3, 1, 1))
        self.reg_branch = nn.Sequential(ConvBNAct(h, h, 3, 1, 1), ConvBNAct(h, h, 3, 1, 1))
        self.obj_pred = nn.Conv2d(h, 1, 1)
        self.cls_pred = nn.Conv2d(h, num_classes, 1)
        self.reg_pred = nn.Conv2d(h, self.n_box, 1)

        prior = -math.log((1 - 0.01) / 0.01)  # focal-style prior, p=0.01
        nn.init.constant_(self.obj_pred.bias, prior)
        nn.init.constant_(self.cls_pred.bias, prior)
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        c = self.cls_branch(x)
        r = self.reg_branch(x)
        out = torch.cat([self.obj_pred(c), self.cls_pred(c), self.reg_pred(r)], dim=1)
        return out.permute(0, 2, 3, 1).contiguous()


class RefinedDetector(BaseDetector):
    """Wraps a base detector: reuse ``base.neck_forward``, drop its heads, attach
    ``DecoupledDFLHead`` per pyramid level."""

    def __init__(self, base: BaseDetector, num_classes: int = 1, reg_max: int = 16):
        super().__init__(num_classes=num_classes)
        # strip the base model's original heads — backbone + neck only
        for name in [n for n, _ in list(base.named_children()) if n.startswith("head")]:
            delattr(base, name)
        self.base = base
        self.reg_max = reg_max
        self.neck_channels = list(base.neck_channels)
        self.heads = nn.ModuleList(
            DecoupledDFLHead(c, num_classes, reg_max) for c in self.neck_channels
        )

    def neck_forward(self, x: torch.Tensor) -> list:
        return self.base.neck_forward(x)

    def forward(self, x: torch.Tensor) -> list:
        return [head(f) for head, f in zip(self.heads, self.neck_forward(x))]


def dfl_expectation(box_logits: torch.Tensor, reg_max: int) -> torch.Tensor:
    """(..., 4*(reg_max+1)) logits -> (..., 4) expected l/t/r/b distances (cell units)."""
    shape = box_logits.shape[:-1]
    probs = box_logits.reshape(*shape, 4, reg_max + 1).softmax(dim=-1)
    proj = torch.arange(reg_max + 1, device=box_logits.device, dtype=probs.dtype)
    return (probs * proj).sum(dim=-1)
