"""Unified training entrypoint for every model family.

  # 1. from-scratch architecture (baseline / fpn / p2 / ab2d)
  python train.py --model fpn --epochs 50 --loss-type iciou

  # 2. head refinement: warm-start backbone+neck, swap in a decoupled DFL head
  python train.py --mode refine --model p2 \
      --base-weights checkpoints/p2_best.pth --epochs 50 --eval-split test --profile

  # 3. final model: FinalDetector + TaskAlignedLoss, warm-started from a refined ckpt
  python train.py --mode final --base-weights models_final/p2_v2.pt \
      --epochs 50 --iou-type mpdiou --eval-split test --profile --save-model models_final/final_model.pt

  # 3b. fine-tune the final model with the backbone frozen
  python train.py --mode final --fine-tune --freeze-backbone \
      --base-weights checkpoints/final/final_model_best.pth --epochs 20 --peak-lr 1e-4 --eval-split test
"""

import argparse
import json
import os
from functools import partial

import torch
import torch.optim as optim

from src.core.runtime import configure
from src.core import complexity
from src.core.loss import DetectionLoss, DetectionLossV2, TaskAlignedLoss
from src.core.metrics import time_inference
from src.core.postprocess import decode_predictions_v2
from src.engine.factory import (
    build_model, build_criterion, build_dataloaders, build_scheduler,
    llrd_param_groups, freeze_backbone, MODEL_REGISTRY, MODEL_NAMES,
)
from src.engine.singlegpu_trainer import SingleGPUTrainer
from src.engine.evaluation import run_evaluation
from src.engine.wnb_tracker import WandbTracker
from src.models.v2 import RefinedDetector
from src.models.p2_granular import P2GranularDetector
from src.models.final_model import FinalDetector

_SCRATCH_MODELS = tuple(MODEL_NAMES)                 # baseline / fpn / p2 / ab2d
_REFINE_MODELS = ("fpn", "p2")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["scratch", "refine", "final"], default="scratch")
    p.add_argument("--model", default="baseline", help="scratch/refine only")
    p.add_argument("--base-weights", default=None, help="warm-start checkpoint (refine/final)")
    p.add_argument("--fine-tune", action="store_true", help="final: resume + polish")
    p.add_argument("--freeze-backbone", action="store_true")

    p.add_argument("--img-dir", default="data/images")
    p.add_argument("--lbl-dir", default="data/labels")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # scratch schedule
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss-type", default="iciou", choices=["iou", "ciou", "eiou", "siou", "iciou"])
    p.add_argument("--neighbor-cells", action=argparse.BooleanOptionalAction, default=True)

    # refine / final schedule
    p.add_argument("--peak-lr", type=float, default=4e-4)
    p.add_argument("--llrd", type=float, default=0.9, help="layer-wise LR decay gamma")
    p.add_argument("--reg-max", type=int, default=16)
    p.add_argument("--ema-decay", type=float, default=0.9995)

    # final (TaskAlignedLoss + FinalDetector) knobs
    p.add_argument("--iou-type", default="mpdiou", choices=["ciou", "mpdiou"])
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--tal-alpha", type=float, default=0.5)
    p.add_argument("--tal-beta", type=float, default=6.0)
    p.add_argument("--w-box", type=float, default=2.5)
    p.add_argument("--w-dfl", type=float, default=0.5)
    p.add_argument("--no-se", action="store_true")

    # split / io
    p.add_argument("--scene-counts", type=int, nargs=3, default=(48, 6, 6))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--eval-split", choices=["val", "test"], default="val")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-model", default=None, help="also write a self-contained model .pt here")
    p.add_argument("--json", default=None)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_state(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck


def build_run(args, device):
    """Returns (model, criterion, optimizer, decode, select_metric, tag)."""
    if args.mode == "scratch":
        if args.model not in _SCRATCH_MODELS:
            raise SystemExit(f"--model must be one of {_SCRATCH_MODELS} for scratch mode")
        model = build_model(args.model)
        crit = build_criterion(args.model, args.loss_type, neighbor_cells=args.neighbor_cells)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return model, crit, opt, None, "mAP_50", args.run_name or f"{args.model}_{args.loss_type}"

    if args.mode == "refine":
        if args.model not in _REFINE_MODELS:
            raise SystemExit(f"--model must be one of {_REFINE_MODELS} for refine mode")
        if not args.base_weights:
            raise SystemExit("refine mode needs --base-weights (a scratch checkpoint)")
        base = build_model(args.model)
        base.load_state_dict(_load_state(args.base_weights)["model_state_dict"])
        model = RefinedDetector(base, num_classes=1, reg_max=args.reg_max)
        _, scale_ranges = MODEL_REGISTRY[args.model]
        crit = DetectionLossV2(reg_max=args.reg_max, scale_ranges=scale_ranges,
                               neighbor_cells=args.neighbor_cells)
        opt = optim.AdamW(llrd_param_groups(model, args.peak_lr, args.llrd, args.weight_decay))
        dec = partial(decode_predictions_v2, reg_max=args.reg_max, conf_thresh=0.01, iou_thresh=0.5, max_det=300)
        return model, crit, opt, dec, "mAP_50_95", args.run_name or f"{args.model}_v2"

    # --- final ---
    if not args.base_weights:
        raise SystemExit("final mode needs --base-weights (a p2_v2.pt or a final_*_best.pth to resume)")
    model = FinalDetector(P2GranularDetector(num_classes=1), num_classes=1,
                          reg_max=args.reg_max, use_se=not args.no_se)
    ck = _load_state(args.base_weights)
    if isinstance(ck, dict) and "model" in ck:          # p2_v2.pt -> take its backbone+neck
        model.base.load_state_dict(ck["model"].base.state_dict())
        print(f"warm-started FinalDetector backbone+neck from {args.base_weights}")
    else:                                               # resume a final_model checkpoint
        model.load_state_dict(ck["model_state_dict"])
        print(f"resumed FinalDetector from {args.base_weights} (epoch {ck.get('epoch', '?')})")
    if args.freeze_backbone:
        n = freeze_backbone(model.base)
        print(f"froze backbone: {n/1e6:.2f}M params fixed")
    crit = TaskAlignedLoss(reg_max=args.reg_max, iou_type=args.iou_type, topk=args.topk,
                           alpha=args.tal_alpha, beta=args.tal_beta, w_box=args.w_box, w_dfl=args.w_dfl)
    opt = optim.AdamW(llrd_param_groups(model, args.peak_lr, args.llrd, args.weight_decay))
    dec = partial(decode_predictions_v2, reg_max=args.reg_max, conf_thresh=0.01, iou_thresh=0.5, max_det=300)
    return model, crit, opt, dec, "mAP_50_95", args.run_name or ("final_model_ft" if args.fine_tune else "final_model")


def main():
    args = parse_args()
    configure(cpu_threads=args.workers, seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, criterion, optimizer, decode, select_metric, tag = build_run(args, device)
    prof = complexity.parameter_count(model)
    print(f"[{tag}] mode={args.mode}  {prof['params_millions']:.2f}M params "
          f"({sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable)")

    loaders = build_dataloaders(
        args.img_dir, args.lbl_dir, img_size=args.img_size, batch_size=args.batch_size,
        which=("train", args.eval_split), scene_counts=tuple(args.scene_counts),
        seed=args.split_seed, manifest_path=args.split_manifest, train_workers=args.workers,
    )
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs=args.warmup_epochs)
    tracker = WandbTracker(project_name="drone-detection", run_name=tag,
                           config={**vars(args), **prof}, enabled=not args.no_wandb)

    trainer = SingleGPUTrainer(
        model=model, train_loader=loaders["train"], val_loader=loaders[args.eval_split],
        criterion=criterion, optimizer=optimizer, lr_scheduler=scheduler, tracker=tracker,
        device=device, model_name=tag, img_size=args.img_size, checkpoint_dir=args.ckpt_dir,
        grad_clip=args.grad_clip,
        ema_decay=(args.ema_decay if args.mode != "scratch" else None),
        decode=decode, select_metric=select_metric,
    )
    trainer.fit(epochs=args.epochs)

    # --- reload best, optional self-contained save + profile + json ---
    best_path = os.path.join(args.ckpt_dir, f"{tag}_best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(_load_state(best_path)["model_state_dict"])
    model.to(device).eval()

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save({"model": model.cpu().eval(), "mode": args.mode, "reg_max": args.reg_max,
                    "config": vars(args), "best": trainer.best_mAP}, args.save_model)
        model.to(device)
        print(f"[{tag}] saved self-contained model -> {args.save_model}")

    if args.json:
        from src.core.metrics import MetricEvaluator
        ev = MetricEvaluator(img_size=args.img_size)
        metrics = run_evaluation(model, loaders[args.eval_split], ev, torch.device(device),
                                 criterion=criterion, decode=decode, progress=False)
        if args.profile:
            metrics.update({f"profile/{k}": v for k, v in
                            complexity.profile_model(model, str(device), img_size=args.img_size).items()})
            metrics["profile/infer_ms_p50_bs1"] = time_inference(model, str(device), args.img_size, 1)["latency_ms_p50"]
        with open(args.json, "w") as fh:
            json.dump({"model": tag, "run": tag, "split": args.eval_split, **metrics}, fh, indent=2)
        print(f"[{tag}] wrote {args.json}  (mAP@0.5 {metrics['mAP_50']:.4f} / "
              f"mAP@0.5:0.95 {metrics['mAP_50_95']:.4f})")


if __name__ == "__main__":
    main()
