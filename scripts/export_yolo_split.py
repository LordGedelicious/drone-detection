"""Materialise the frozen scene split as an Ultralytics YOLO dataset (symlinks,
no copies) so the YOLO benchmark trains/evals on exactly the same train/val/test
frames as the from-scratch models.

    python scripts/export_yolo_split.py --manifest splits/seed42_48-6-6.json --out data/yolo
"""
import argparse
import json
import os

from src.core.split import make_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--lbl-dir", default="data/labels")
    ap.add_argument("--manifest", default="splits/seed42_48-6-6.json")
    ap.add_argument("--out", default="data/yolo")
    args = ap.parse_args()

    split = make_splits(args.img_dir, manifest_path=args.manifest)
    img_root = os.path.abspath(args.img_dir)
    lbl_root = os.path.abspath(args.lbl_dir)
    out = os.path.abspath(args.out)

    for part in ("train", "val", "test"):
        for kind in ("images", "labels"):
            d = os.path.join(out, kind, part)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.unlink(os.path.join(d, f))
        for fn in split[part]:
            stem = os.path.splitext(fn)[0]
            os.symlink(os.path.join(img_root, fn), os.path.join(out, "images", part, fn))
            os.symlink(os.path.join(lbl_root, stem + ".txt"), os.path.join(out, "labels", part, stem + ".txt"))
        print(f"{part}: {len(split[part])} frames")

    yaml_path = os.path.join(out, "drone.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: drone\n")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
