"""
Construction helpers shared by ``train.py``, ``finetune.py`` and ``eval.py`` so
model wiring, loss configuration and the data split live in exactly one place.

``SUBMISSION_MODELS`` (baseline/fpn/p2) are the from-scratch candidates for the
"best model" pick. ``BENCHMARK_MODELS`` (ab2d) is from-scratch too but heavier /
more derivative -- run through the same pipeline for comparison only. The
pretrained YOLO reference (``src/models/reference/yolov26.py``) has its own
trainer and is not registered here.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.core.dataset import DroneDataset, get_train_transforms, get_val_transforms, collate_fn
from src.core.loss import DetectionLoss
from src.core.split import make_splits
from src.models.single_scale import SingleScaleDetector
from src.models.multi_scale_fpn import MultiScaleFPNDetector
from src.models.p2_granular import P2GranularDetector
from src.models.reference.ab2d_yolo import AB2DYOLO

# name -> (class, per-scale max-edge ranges in normalized image coords, fine -> coarse)
_P2_RANGES = [(0.0, 0.05), (0.04, 0.12), (0.10, 0.28), (0.25, 1.00)]
MODEL_REGISTRY = {
    "baseline": (SingleScaleDetector, [(0.0, 1.0)]),
    "fpn": (MultiScaleFPNDetector, [(0.0, 0.10), (0.08, 0.25), (0.20, 1.00)]),
    "p2": (P2GranularDetector, _P2_RANGES),
    # AB2D-YOLO is from-scratch too, but heavier/more derivative (attention + BiFPN
    # + C2f-DWR) -- a comparison benchmark, NOT a "which is my best model" candidate.
    "ab2d": (AB2DYOLO, _P2_RANGES),
}
SUBMISSION_MODELS = ("baseline", "fpn", "p2")   # eligible to be "the best model"
BENCHMARK_MODELS = ("ab2d",)                     # comparison only
MODEL_NAMES = tuple(MODEL_REGISTRY)


def build_model(name: str, num_classes: int = 1) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}'. choices: {MODEL_NAMES}")
    cls, _ = MODEL_REGISTRY[name]
    return cls(num_classes=num_classes)


def build_criterion(name: str, loss_type: str = "iciou", neighbor_cells: bool = True) -> DetectionLoss:
    _, scale_ranges = MODEL_REGISTRY[name]
    return DetectionLoss(loss_type=loss_type, scale_ranges=scale_ranges, neighbor_cells=neighbor_cells)


def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 3):
    """Linear LR warmup then cosine anneal to ~0. Stepped once per epoch.

    From-scratch detection at AdamW lr=1e-3 is unstable for the first ~1 epoch
    (loss can spike into the hundreds); the warmup ramps LR from 1% -> 100% so
    all three architectures start cleanly.
    """
    warmup_epochs = max(0, min(warmup_epochs, max(epochs - 1, 0)))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


def _seed_worker(worker_id: int):
    """Seed NumPy / random inside each DataLoader worker so the Albumentations
    augmentation stream is reproducible across runs."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def freeze_backbone(model: torch.nn.Module) -> int:
    """Freeze the shared backbone stages so fine-tuning only adapts neck+heads.
    Returns the number of frozen parameters."""
    frozen = 0
    for attr in ("stem", "stage2", "stage3", "stage4", "stage5", "aifi"):
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen


def build_dataloaders(
    img_dir: str,
    lbl_dir: str,
    img_size: int = 640,
    batch_size: int = 16,
    which=("train", "val"),
    scene_counts=(48, 6, 6),
    seed: int = 42,
    manifest_path: str = None,
    train_workers: int = 8,
    eval_workers: int = 4,
) -> dict:
    """Returns {split_name: DataLoader} for the requested splits. The 'train'
    split gets the augmentation pipeline; 'val'/'test' get letterbox + normalize
    only. Shuffling is on for 'train' only."""
    split = make_splits(img_dir, scene_counts=scene_counts, seed=seed, manifest_path=manifest_path)
    loaders = {}
    for name in which:
        is_train = name == "train"
        tf = get_train_transforms(img_size) if is_train else get_val_transforms(img_size)
        ds = DroneDataset(img_dir, lbl_dir, transforms=tf, image_files=split[name])
        workers = train_workers if is_train else eval_workers
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            collate_fn=collate_fn,
            num_workers=workers,
            persistent_workers=is_train and workers > 0,
            prefetch_factor=2 if workers > 0 else None,
            pin_memory=True,
            drop_last=is_train,
            worker_init_fn=_seed_worker if is_train else None,
            generator=torch.Generator().manual_seed(seed) if is_train else None,
        )
    return loaders


def llrd_param_groups(model, base_lr: float, gamma: float, weight_decay: float = 1e-4):
    """Discriminative (layer-wise) learning-rate groups for a warm-started model
    (RefinedDetector / FinalDetector): backbone stages at ``base_lr * gamma**2``,
    the neck at ``base_lr * gamma``, the new heads (+ SE, if present) at
    ``base_lr``. Only ``requires_grad`` parameters are included, so it composes
    with ``freeze_backbone``. (Discriminative fine-tuning, Howard & Ruder 2018.)
    """
    base = model.base
    deep, neck = [], []
    for name, mod in base.named_children():
        ps = [q for q in mod.parameters() if q.requires_grad]
        if not ps:
            continue
        (deep if name in ("stem", "stage2", "stage3", "stage4", "stage5") else neck).extend(ps)
    head = [q for q in model.heads.parameters() if q.requires_grad]
    if getattr(model, "se", None) is not None:
        head += [q for q in model.se.parameters() if q.requires_grad]
    n_grouped = len(deep) + len(neck) + len(head)
    n_trainable = len([q for q in model.parameters() if q.requires_grad])
    assert n_grouped == n_trainable, f"llrd grouping missed {n_trainable - n_grouped} tensors"
    return [
        {"params": deep, "lr": base_lr * gamma ** 2, "weight_decay": weight_decay},
        {"params": neck, "lr": base_lr * gamma, "weight_decay": weight_decay},
        {"params": head, "lr": base_lr, "weight_decay": weight_decay},
    ]
