"""Fine-tune a trained detector from an existing checkpoint: lower LR, shorter
cosine schedule, optional backbone freeze. Uses the same scene split as
training so val/test stay comparable.

Example:
    python finetune.py --weights checkpoints/fpn_best.pth --epochs 20 --lr 1e-4 --freeze-backbone
"""

import argparse
import torch
import torch.optim as optim

from src.core.runtime import configure
from src.engine.factory import (
    build_model, build_criterion, build_dataloaders, build_scheduler, freeze_backbone, MODEL_NAMES,
)
from src.engine.singlegpu_trainer import SingleGPUTrainer
from src.engine.wnb_tracker import WandbTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="checkpoint to start from")
    p.add_argument("--model", choices=MODEL_NAMES, default=None, help="override; else read from checkpoint")
    p.add_argument("--img-dir", default="data/images")
    p.add_argument("--lbl-dir", default="data/labels")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--loss-type", default="iciou", choices=["iou", "ciou", "eiou", "siou", "iciou"])
    p.add_argument("--neighbor-cells", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--scene-counts", type=int, nargs=3, default=(48, 6, 6))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    configure(cpu_threads=args.workers, seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.weights, map_location="cpu")
    model_name = args.model or ckpt.get("model_name")
    if model_name is None:
        raise SystemExit("checkpoint has no 'model_name'; pass --model explicitly")

    model = build_model(model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"resumed {model_name} from {args.weights} (epoch {ckpt.get('epoch', '?')}, "
          f"best mAP {ckpt.get('best_mAP', 0.0):.4f})")

    if args.freeze_backbone:
        n = freeze_backbone(model)
        print(f"froze backbone: {n/1e6:.2f}M params held fixed")

    loaders = build_dataloaders(
        args.img_dir, args.lbl_dir, img_size=args.img_size, batch_size=args.batch_size,
        which=("train", "val"), scene_counts=tuple(args.scene_counts), seed=args.split_seed,
        manifest_path=args.split_manifest, train_workers=args.workers,
    )
    criterion = build_criterion(model_name, args.loss_type, neighbor_cells=args.neighbor_cells)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs=min(1, args.epochs - 1))

    tracker = WandbTracker(
        project_name="drone-detection",
        run_name=args.run_name or f"{model_name}_finetune",
        config=vars(args),
        enabled=not args.no_wandb,
    )

    trainer = SingleGPUTrainer(
        model=model, train_loader=loaders["train"], val_loader=loaders["val"],
        criterion=criterion, optimizer=optimizer, lr_scheduler=scheduler, tracker=tracker,
        device=device, model_name=f"{model_name}_ft", img_size=args.img_size,
        checkpoint_dir=args.ckpt_dir,
    )
    trainer.best_mAP = ckpt.get("best_mAP", 0.0)
    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
