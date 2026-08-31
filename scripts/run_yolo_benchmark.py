"""YOLO reference benchmark on the frozen split. Trains a COCO-pretrained and a
from-scratch YOLOv8n and writes compare.py-compatible JSON.

Metrics come from Ultralytics' own validator (its NMS / conf / mAP implementation),
so they are only roughly comparable to the custom-model numbers — note that in
the paper.

Run with the isolated venv:
    PYTHONPATH=. /opt/ultra-venv/bin/python scripts/run_yolo_benchmark.py --epochs 50 --split test --out runs/full
"""
import argparse
import json
import os

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params


def run(weights, tag, data, epochs, split, out_dir):
    scratch = weights.endswith(".yaml")
    model = YOLO(weights)
    model.train(
        data=data, epochs=epochs, imgsz=640, batch=16, device=0, seed=42,
        deterministic=False, pretrained=not scratch, project="runs/yolo",
        name=tag, exist_ok=True, verbose=False, plots=False,
    )
    m = model.val(split=split, project="runs/yolo", name=f"{tag}_{split}",
                  exist_ok=True, verbose=False)

    n_params = get_num_params(model.model)
    try:
        flops = get_flops(model.model, 640)
    except Exception:
        flops = float("nan")
    p, r = float(m.box.mp), float(m.box.mr)
    rec = {
        "model": tag.rsplit("_", 1)[0],
        "run": tag,
        "split": split,
        "mAP_50": float(m.box.map50),
        "mAP_50_95": float(m.box.map),
        "precision": p,
        "recall": r,
        "f1_score": 2 * p * r / (p + r + 1e-9),
        "profile/params_millions": n_params / 1e6,
        "profile/gflops": float(flops),
        "_note": "Ultralytics val() metrics — methodology differs from the custom evaluator",
    }
    path = os.path.join(out_dir, f"{tag}_{split}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"wrote {path}: mAP50={rec['mAP_50']:.3f} mAP50-95={rec['mAP_50_95']:.3f} "
          f"P={p:.3f} R={r:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/drone.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="runs/bakeoff")
    ap.add_argument("--models", nargs="+",
                    default=["yolo26n:pretrained", "yolo26n:scratch"],
                    help="<base>:<pretrained|scratch> ...")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for spec in args.models:
        base, mode = spec.split(":")            # e.g. yolo26n:pretrained  or  yolo26n:scratch
        weights = f"{base}.pt" if mode == "pretrained" else f"{base}.yaml"
        run(weights, f"{base}_{mode}", args.data, args.epochs, args.split, args.out)
