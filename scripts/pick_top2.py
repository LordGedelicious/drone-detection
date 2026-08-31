"""Print the top-N non-reference models from runs/full/*_test.json, one per line
as  <model> <best-variant-checkpoint-path> <test_mAP5095> , ranked by mAP@0.5:0.95.
"""
import glob
import json
import os
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SUBMISSION = {"baseline", "fpn", "p2"}

best = {}  # model -> (score, run, ckpt)
for f in sorted(glob.glob("runs/full/*_test.json")):
    d = json.load(open(f))
    m = d.get("model")
    if m not in SUBMISSION:
        continue
    run = d.get("run", m)                       # e.g. fpn_neighbors
    score = d.get("mAP_50_95", 0.0)
    ckpt = f"checkpoints/full/{run}/{m}_best.pth"
    if not os.path.exists(ckpt):
        continue
    if m not in best or score > best[m][0]:
        best[m] = (score, run, ckpt)

ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:N]
for model, (score, run, ckpt) in ranked:
    print(f"{model} {ckpt} {score:.4f} {run}")
