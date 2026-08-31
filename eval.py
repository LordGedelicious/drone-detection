"""Evaluate a checkpoint on the val or test split and print a full detection
report (mAP, per-size AP, per-condition mAP, best-F1 P/R/F1, count error),
optionally with a model-complexity + speed profile.

Example:
    python eval.py --weights checkpoints/fpn_best.pth --split test --profile
"""

import argparse
import json
import torch

from src.core.runtime import configure
from src.core.metrics import MetricEvaluator, time_inference
from src.core import complexity
from src.engine.factory import build_model, build_criterion, build_dataloaders, MODEL_NAMES
from src.engine.evaluation import run_evaluation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to a *_best.pth / *_last.pth checkpoint")
    p.add_argument("--model", choices=MODEL_NAMES, default=None,
                   help="override; otherwise read from the checkpoint")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--img-dir", default="data/images")
    p.add_argument("--lbl-dir", default="data/labels")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.01, help="score floor for the PR curve")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--loss-type", default="iciou")
    p.add_argument("--scene-counts", type=int, nargs=3, default=(48, 6, 6))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--profile", action="store_true", help="also report params / FLOPs / speed")
    p.add_argument("--tag", default=None, help="label for this run in the comparison table")
    p.add_argument("--json", default=None, help="write the metrics dict to this path")
    return p.parse_args()


def main():
    args = parse_args()
    configure(cpu_threads=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location="cpu")
    model_name = args.model or ckpt.get("model_name")
    if model_name is None:
        raise SystemExit("checkpoint has no 'model_name'; pass --model explicitly")

    model = build_model(model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"loaded {model_name} from {args.weights} (epoch {ckpt.get('epoch', '?')})")

    loaders = build_dataloaders(
        args.img_dir, args.lbl_dir, img_size=args.img_size, batch_size=args.batch_size,
        which=(args.split,), scene_counts=tuple(args.scene_counts), seed=args.split_seed,
        manifest_path=args.split_manifest, eval_workers=args.workers,
    )

    evaluator = MetricEvaluator(img_size=args.img_size)
    criterion = build_criterion(model_name, args.loss_type)
    metrics = run_evaluation(
        model, loaders[args.split], evaluator, device, criterion=criterion,
        conf_thresh=args.conf, iou_thresh=args.iou, max_det=args.max_det,
        desc=f"[{args.split}]",
    )

    print(f"\n=== {model_name} @ {args.split} ===")
    print(evaluator.summary_table())
    print(f"  {'val loss':<28} : {metrics.get('val/total_loss', float('nan')):.4f}")

    if args.profile:
        prof = complexity.profile_model(model, str(device), img_size=args.img_size)
        lat1 = time_inference(model, str(device), args.img_size, batch_size=1)
        print("\n=== complexity / speed ===")
        print(f"  params            : {prof['params_millions']:.2f} M")
        print(f"  model size        : {prof['model_size_mb']:.1f} MB")
        print(f"  GFLOPs (1x640)     : {prof['gflops']:.2f}")
        print(f"  peak train mem     : {prof['peak_train_mem_mb']:.0f} MB")
        print(f"  train throughput   : {prof['train_imgs_per_s']:.0f} img/s")
        print(f"  infer latency p50  : {lat1['latency_ms_p50']:.2f} ms  ({lat1['fps']:.0f} FPS @bs1)")
        metrics.update({f"profile/{k}": v for k, v in prof.items()})
        metrics["profile/infer_ms_p50_bs1"] = lat1["latency_ms_p50"]

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"model": model_name, "run": args.tag or model_name,
                       "split": args.split, **metrics}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
