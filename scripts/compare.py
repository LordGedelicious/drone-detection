"""Render eval.py JSON dumps as one markdown comparison table.

    python scripts/compare.py runs/bakeoff/*_val.json
"""

import json
import sys

# (json key, column header, format)
ROWS = [
    ("mAP_50", "mAP@0.5", "{:.3f}"),
    ("mAP_50_95", "mAP@0.5:0.95", "{:.3f}"),
    ("f1_score", "F1 (max)", "{:.3f}"),
    ("precision", "P @F1*", "{:.3f}"),
    ("recall", "R @F1*", "{:.3f}"),
    ("count_mae", "count MAE", "{:.2f}"),
    ("AP_50/size_tiny", "AP tiny(<16px)", "{:.3f}"),
    ("AP_50/size_small", "AP small(16-32)", "{:.3f}"),
    ("AP_50/size_medium", "AP medium(>32)", "{:.3f}"),
    ("profile/params_millions", "params (M)", "{:.2f}"),
    ("profile/gflops", "GFLOPs", "{:.1f}"),
    ("profile/model_size_mb", "size (MB)", "{:.1f}"),
    ("profile/peak_train_mem_mb", "train mem (MB)", "{:.0f}"),
    ("profile/train_imgs_per_s", "train img/s", "{:.0f}"),
    ("profile/infer_ms_p50_bs1", "infer ms @bs1", "{:.2f}"),
]


def load(paths):
    runs = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        runs.append((d.get("run") or d.get("model", p), d))
    return runs


def fmt(val, spec):
    try:
        return spec.format(val)
    except (TypeError, ValueError):
        return "-"


def main(paths):
    runs = load(paths)
    names = [n for n, _ in runs]
    header = "| metric | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    lines = [header, sep]
    for key, label, spec in ROWS:
        if not any(key in d for _, d in runs):
            continue
        cells = [fmt(d.get(key), spec) for _, d in runs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # per-condition mAP block
    cond_keys = sorted({k for _, d in runs for k in d if k.startswith("mAP_50/cond_")})
    for key in cond_keys:
        label = "mAP@0.5 " + key.split("cond_")[1]
        cells = [fmt(d.get(key), "{:.3f}") for _, d in runs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
