"""Train one from-scratch detector (baseline / fpn / p2) on the drone dataset.

Example:
    python train.py --model fpn --epochs 50 --loss-type iciou

The pretrained/library reference models (YOLOv26, AB2D-YOLO) are deliberately
not reachable from here -- see src/models/reference/.
"""

import argparse
import torch
import torch.optim as optim

from src.core.runtime import configure
from src.core import complexity
from src.engine.factory import build_model, build_criterion, build_dataloaders, build_scheduler, MODEL_NAMES
from src.engine.singlegpu_trainer import SingleGPUTrainer
from src.engine.wnb_tracker import WandbTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="baseline", choices=MODEL_NAMES)
    p.add_argument("--img-dir", type=str, default="data/images")
    p.add_argument("--lbl-dir", type=str, default="data/labels")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--loss-type", type=str, default="iciou",
                   choices=["iou", "ciou", "eiou", "siou", "iciou"])
    p.add_argument("--neighbor-cells", action=argparse.BooleanOptionalAction, default=True,
                   help="assign each GT to its centre cell + 2 nearest neighbours")
    p.add_argument("--scene-counts", type=int, nargs=3, default=(48, 6, 6),
                   metavar=("TRAIN", "VAL", "TEST"), help="scene counts per split")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-manifest", type=str, default=None,
                   help="freeze/reuse the split via a JSON manifest")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    configure(cpu_threads=args.workers, seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders = build_dataloaders(
        args.img_dir, args.lbl_dir, img_size=args.img_size, batch_size=args.batch_size,
        which=("train", "val"), scene_counts=tuple(args.scene_counts), seed=args.split_seed,
        manifest_path=args.split_manifest, train_workers=args.workers,
    )
    print(f"train batches: {len(loaders['train'])} | val batches: {len(loaders['val'])}")

    model = build_model(args.model)
    criterion = build_criterion(args.model, args.loss_type, neighbor_cells=args.neighbor_cells)

    prof = complexity.parameter_count(model)
    print(f"model '{args.model}': {prof['params_millions']:.2f}M params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs=args.warmup_epochs)

    tracker = WandbTracker(
        project_name="drone-detection",
        run_name=args.run_name or f"{args.model}_{args.loss_type}",
        config={**vars(args), **prof},
        enabled=not args.no_wandb,
    )

    trainer = SingleGPUTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        tracker=tracker,
        device=device,
        model_name=args.model,
        img_size=args.img_size,
        checkpoint_dir=args.ckpt_dir,
    )
    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
