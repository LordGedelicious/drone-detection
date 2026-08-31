"""
V2 refinement: warm-start a from-scratch detector's backbone + neck from its
best 50-epoch checkpoint, replace the heads with a decoupled DFL head
(src/models/v2.py), and continue training under the Generalized Focal Loss
objective (QFL + DFL + CIoU) with:

  * layer-wise LR decay  -- backbone gets base_lr * gamma^2, neck * gamma,
    heads * base_lr  (discriminative fine-tuning, Howard & Ruder, ULMFiT 2018)
  * cosine schedule with warmup  (SGDR, Loshchilov & Hutter, ICLR 2017)
  * weight EMA  (Polyak averaging; YOLOX / YOLOv5)

References:
  Generalized Focal Loss, Li et al., NeurIPS 2020
  YOLOX, Ge et al., 2021 (arXiv:2107.08430)

Example:
  python refine.py --model fpn --base-weights checkpoints/full/fpn_neighbors/fpn_best.pth \
      --epochs 30 --peak-lr 2e-4 --llrd 0.85 --split test --profile
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.core.runtime import configure
from src.core.metrics import MetricEvaluator
from src.core.loss import DetectionLossV2
from src.core.ema import ModelEMA
from src.core.postprocess import decode_predictions_v2
from src.core import complexity
from src.engine.factory import build_model, build_dataloaders, build_scheduler, MODEL_REGISTRY, MODEL_NAMES
from src.engine.evaluation import run_evaluation
from src.engine.wnb_tracker import WandbTracker
from src.models.v2 import RefinedDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=[m for m in MODEL_NAMES if m != "ab2d"])
    p.add_argument("--base-weights", required=True, help="best 50-epoch checkpoint for --model")
    p.add_argument("--img-dir", default="data/images")
    p.add_argument("--lbl-dir", default="data/labels")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--peak-lr", type=float, default=2e-4)
    p.add_argument("--llrd", type=float, default=0.85, help="layer-wise LR decay factor gamma")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--reg-max", type=int, default=16)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--neighbor-cells", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--scene-counts", type=int, nargs=3, default=(48, 6, 6))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-manifest", default="splits/seed42_48-6-6.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ckpt-dir", default="checkpoints/v2")
    p.add_argument("--final-dir", default="models_final")
    p.add_argument("--json", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def llrd_param_groups(model: RefinedDetector, base_lr: float, gamma: float, wd: float):
    base = model.base
    deep, neck = [], []
    for name, mod in base.named_children():
        params = list(mod.parameters())
        if not params:
            continue
        if name in ("stem", "stage2", "stage3", "stage4", "stage5"):
            deep += params
        else:  # lateral_* / smooth_* / downsample_* / aifi / ...
            neck += params
    head = list(model.heads.parameters())
    n_grouped = len(deep) + len(neck) + len(head)
    n_total = len(list(model.parameters()))
    assert n_grouped == n_total, f"param grouping missed {n_total - n_grouped} tensors"
    return [
        {"params": deep, "lr": base_lr * gamma ** 2, "weight_decay": wd},
        {"params": neck, "lr": base_lr * gamma, "weight_decay": wd},
        {"params": head, "lr": base_lr, "weight_decay": wd},
    ]


def main():
    args = parse_args()
    configure(cpu_threads=args.workers, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.run_name or f"{args.model}_v2"

    # --- build V2 model: load base backbone+neck, attach fresh decoupled DFL heads ---
    base = build_model(args.model)
    ckpt = torch.load(args.base_weights, map_location="cpu")
    base.load_state_dict(ckpt["model_state_dict"])
    model = RefinedDetector(base, num_classes=1, reg_max=args.reg_max).to(device)
    _, scale_ranges = MODEL_REGISTRY[args.model]
    print(f"{tag}: warm-started backbone+neck from {args.base_weights} "
          f"(base epoch {ckpt.get('epoch', '?')}); {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    loaders = build_dataloaders(
        args.img_dir, args.lbl_dir, img_size=args.img_size, batch_size=args.batch_size,
        which=("train", args.split) if args.split != "train" else ("train",),
        scene_counts=tuple(args.scene_counts), seed=args.split_seed,
        manifest_path=args.split_manifest, train_workers=args.workers,
    )
    criterion = DetectionLossV2(reg_max=args.reg_max, scale_ranges=scale_ranges,
                                neighbor_cells=args.neighbor_cells)
    optimizer = optim.AdamW(llrd_param_groups(model, args.peak_lr, args.llrd, args.weight_decay))
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs=args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay else None
    evaluator = MetricEvaluator(img_size=args.img_size)

    def decode(preds):
        return decode_predictions_v2(preds, reg_max=args.reg_max, conf_thresh=0.01,
                                     iou_thresh=0.5, max_det=300)

    tracker = WandbTracker(project_name="drone-detection", run_name=tag,
                           config=vars(args), enabled=not args.no_wandb)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_map, best_state = 0.0, None

    for epoch in range(args.epochs):
        model.train()
        running, n_ok, skipped = 0.0, 0, 0
        for images, targets in tqdm(loaders["train"], desc=f"{tag} ep{epoch+1} [ref]", leave=False):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss, ld = criterion(model(images), targets)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)
            running += ld["loss/total"]
            n_ok += 1
        scheduler.step()

        eval_model = ema.ema if ema else model
        metrics = run_evaluation(eval_model, loaders[args.split], evaluator, device,
                                 criterion=criterion, decode=decode, progress=False,
                                 desc=f"{tag} ep{epoch+1}")
        tr_loss = running / max(n_ok, 1)
        print(f"[{tag}] ep {epoch+1}/{args.epochs} | train {tr_loss:.3f} | "
              f"{args.split} mAP@0.5 {metrics['mAP_50']:.4f} | mAP@0.5:0.95 {metrics['mAP_50_95']:.4f} | "
              f"F1 {metrics['f1_score']:.4f}" + (f" | skipped {skipped}" if skipped else ""))
        tracker.log({"train/total_loss": tr_loss, "epoch": epoch + 1, **metrics}, step=epoch + 1)

        if metrics["mAP_50"] > best_map:
            best_map = metrics["mAP_50"]
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            torch.save(
                {"epoch": epoch + 1, "model_name": f"{args.model}_v2", "arch": "RefinedDetector",
                 "base_model": args.model, "reg_max": args.reg_max,
                 "config": {"peak_lr": args.peak_lr, "llrd": args.llrd, "ema_decay": args.ema_decay},
                 "model_state_dict": best_state, "best_mAP": best_map},
                os.path.join(args.ckpt_dir, f"{args.model}_v2_best.pth"),
            )

    # --- final: reload best (EMA) weights, save self-contained model + report ---
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(args.final_dir, exist_ok=True)
    final_path = os.path.join(args.final_dir, f"{args.model}_v2.pt")
    torch.save({"model": model.cpu().eval(), "base_model": args.model, "reg_max": args.reg_max,
                "config": vars(args), "best_mAP": best_map}, final_path)
    model.to(device)
    print(f"[{tag}] saved self-contained model -> {final_path}  (best {args.split} mAP@0.5 {best_map:.4f})")

    if args.profile:
        prof = complexity.profile_model(model, str(device), img_size=args.img_size)
        metrics.update({f"profile/{k}": v for k, v in prof.items()})
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"model": f"{args.model}_v2", "run": tag, "split": args.split, **metrics}, fh, indent=2)
        print(f"[{tag}] wrote {args.json}")
    tracker.finish()


if __name__ == "__main__":
    main()
