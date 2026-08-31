"""Exponential moving average of model weights (Polyak averaging).

A cheap, reliable +~0.5-1 mAP in modern detectors (YOLOX / YOLOv5). The EMA
copy is what gets evaluated and saved.
"""

import copy
import math

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995, warmup: int = 2000):
        self.ema = copy.deepcopy(self._unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = 0

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup))  # ramp in
        msd = self._unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1.0 - d)
