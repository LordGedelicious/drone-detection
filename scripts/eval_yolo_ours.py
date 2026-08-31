"""Score the trained YOLO checkpoints with OUR MetricEvaluator + timing, so the
comparison table has the same columns (AP-by-size, count MAE, bs=1 FPS,
per-condition) as the from-scratch models. Run in the ultralytics venv:

    PYTHONPATH=. /opt/ultra-venv/bin/python scripts/eval_yolo_ours.py --split test
"""
import argparse, json, os, glob
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from src.core.runtime import configure
from src.core.metrics import MetricEvaluator, time_inference
from src.core.split import make_splits, condition_of


def read_gt_xyxy(lbl_path):
    b = []
    if os.path.exists(lbl_path):
        for ln in open(lbl_path):
            p = ln.split()
            if len(p) == 5:
                cx, cy, w, h = map(float, p[1:])
                b.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return torch.tensor(b, dtype=torch.float32).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--lbl-dir", default="data/labels")
    ap.add_argument("--manifest", default="splits/seed42_48-6-6.json")
    ap.add_argument("--out", default="runs/full")
    args = ap.parse_args()
    configure(cpu_threads=8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split = make_splits(args.img_dir, manifest_path=args.manifest)[args.split]

    for w in sorted(glob.glob("runs/**/weights/best.pt", recursive=True)):
        tag = w.split("/")[-3]                         # yolo26_pretrained / yolo26_scratch
        model = YOLO(w)
        ev = MetricEvaluator(img_size=640)
        for fn in split:
            gt = read_gt_xyxy(os.path.join(args.lbl_dir, os.path.splitext(fn)[0] + ".txt")).to(dev)
            r = model.predict(os.path.join(args.img_dir, fn), imgsz=640, conf=0.001,
                              iou=0.5, max_det=300, verbose=False, device=0)[0]
            pb = r.boxes.xyxyn.to(dev) if len(r.boxes) else torch.zeros(0, 4, device=dev)
            ps = r.boxes.conf.to(dev) if len(r.boxes) else torch.zeros(0, device=dev)
            ev.update(pb, ps, gt, condition=condition_of(fn))
        m = ev.compute_metrics()
        lat = time_inference(model.model.to(dev).eval(), dev, 640, batch_size=1, runs=60)
        m["profile/infer_ms_p50_bs1"] = lat["latency_ms_p50"]
        m["profile/params_millions"] = get_num_params(model.model) / 1e6
        try:
            m["profile/gflops"] = float(get_flops(model.model, 640))
        except Exception:
            pass
        rec = {"model": tag.rsplit("_", 1)[0], "run": tag, "split": args.split, **m}
        path = os.path.join(args.out, f"{tag}_{args.split}.json")
        json.dump(rec, open(path, "w"), indent=2)
        print(f"{tag}: mAP50={m['mAP_50']:.3f} mAP50-95={m['mAP_50_95']:.3f} "
              f"F1={m['f1_score']:.3f} APsmall={m['AP_50/size_small']:.3f} "
              f"countMAE={m['count_mae']:.2f} {lat['fps']:.0f}FPS -> {path}")


if __name__ == "__main__":
    main()
