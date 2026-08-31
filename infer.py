"""Run a trained detector on an image or a folder of images and write copies
with drawn boxes (and optionally YOLO-format .txt predictions).

Example:
    python infer.py --weights checkpoints/fpn_best.pth --source data/images/some.png --out runs/infer
"""

import argparse
import os
import glob
import cv2
import numpy as np
import torch

from src.core.runtime import configure
from src.core.dataset import get_val_transforms
from src.core.postprocess import decode_predictions, letterbox_boxes_to_original
from src.engine.factory import build_model, MODEL_NAMES

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--model", choices=MODEL_NAMES, default=None)
    p.add_argument("--source", required=True, help="image file or directory")
    p.add_argument("--out", default="runs/infer")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--save-txt", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def list_sources(source: str) -> list:
    if os.path.isdir(source):
        return sorted(f for f in glob.glob(os.path.join(source, "*")) if f.lower().endswith(IMG_EXTS))
    return [source]


@torch.no_grad()
def main():
    args = parse_args()
    configure(cpu_threads=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location="cpu")
    model_name = args.model or ckpt.get("model_name")
    model = build_model(model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    tf = get_val_transforms(args.img_size)
    os.makedirs(args.out, exist_ok=True)
    sources = list_sources(args.source)
    print(f"{model_name}: {len(sources)} image(s) -> {args.out}")

    for path in sources:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  skip (unreadable): {path}")
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = tf(image=rgb, bboxes=[], class_labels=[])["image"].unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            preds = model(x)
        det = decode_predictions([p.float() for p in preds], args.conf, args.iou, args.max_det)[0]
        boxes = letterbox_boxes_to_original(det["boxes"].cpu(), h, w, args.img_size).numpy()
        scores = det["scores"].cpu().numpy()

        for (x1, y1, x2, y2), s in zip(boxes, scores):
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(bgr, p1, p2, (0, 255, 0), 2)
            cv2.putText(bgr, f"{s:.2f}", (p1[0], max(0, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out, f"{stem}.png"), bgr)
        if args.save_txt:
            with open(os.path.join(args.out, f"{stem}.txt"), "w") as fh:
                for (x1, y1, x2, y2), s in zip(boxes, scores):
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {s:.4f}\n")
        print(f"  {stem}: {len(scores)} detection(s)")


if __name__ == "__main__":
    main()
