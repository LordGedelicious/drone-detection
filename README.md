# Aerial Drone Detection — a vanilla detector built from scratch

Locate and count small drones in synthetic aerial imagery (city / forest / lake,
sunny / foggy), under the constraint that **no pretrained weights and no
third‑party detection framework** may appear in the submitted model. Everything
in [src/models/](src/models/) is a plain PyTorch `nn.Module` written for this
project; every loss in [src/core/loss.py](src/core/loss.py) is hand‑rolled.

The detector is built in **four stages**, each pairing an architecture change
with a matching change to the training objective:

| Stage | Architecture | Objective | test mAP@0.5 / mAP@0.5:0.95 |
|------:|--------------|-----------|:--------------------------:|
| 1 | single‑scale head (P5) | Focal + ICIoU | 0.972 / 0.545 |
| 2 | + top‑down FPN (P3–P5) | Focal + ICIoU | 0.978 / 0.589 |
| 3 | + P2 high‑resolution head | Focal + ICIoU | 0.985 / 0.594 |
| 3b | + decoupled Distribution‑Focal‑Loss head | QFL + DFL + CIoU | 0.990 / 0.646 |
| **4b** | **− P5 head, + SE gate, task‑aligned + frozen‑backbone fine‑tune** | **VFL + DFL + MPDIoU** | **0.988 / 0.687** |

**Best model: Stage 4b** — the task‑aligned final model (`models_final/final_model.pt`).
It beats a from‑scratch YOLOv8‑n (0.649) and YOLO26‑n (0.613) trained under the
identical protocol and approaches COCO‑pretrained YOLO (~0.70–0.72) at strict IoU.
Full numbers, ablations and cost analysis are in the paper:
[paper/paper.pdf](paper/paper.pdf).

---

## Repository layout

```
src/
  core/      data, augmentation, scene split, losses, metrics, post‑processing, EMA, runtime
  models/    one architecture per file over a shared BaseDetector
             single_scale.py · multi_scale_fpn.py · p2_granular.py
             v2.py (decoupled DFL head) · final_model.py (Stage 4)
             reference/  ab2d_yolo.py (from‑scratch attention/BiFPN benchmark), yolov26.py (external, context only)
  engine/    SingleGPUTrainer, MultiGPUTrainer (DDP, bonus), model/loss/data factory,
             one shared evaluation routine, W&B tracker
train.py     one entrypoint for every stage (scratch / refine / final / fine‑tune)
eval.py      full detection report for a checkpoint (mAP, per‑size AP, per‑condition, P/R/F1, count MAE) + cost profile
infer.py     draw boxes on an image or a folder
finetune.py  low‑LR fine‑tune from a checkpoint
scripts/     reproduce the whole comparison end to end (bake‑off, full matrix, refinement, YOLO benchmark)
splits/      the frozen scene split manifest (tracked — every experiment reads this)
paper/       IEEE‑format write‑up (paper.pdf)
Makefile · Dockerfile · docker-compose.yml
```

---

## Installation

Requires an NVIDIA GPU (CUDA 12.8) for training; CPU works for inference on a few
images. Two options:

### A. Local virtualenv

```bash
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate
pip install -r requirements-final.txt        # torch 2.8.0+cu128, numpy, albumentations, opencv, tqdm, wandb
export PYTHONPATH=.                           # so `import src...` resolves
```

### B. Docker (bonus: containerization)

```bash
docker compose build                                     # image: drone-detection:latest
docker compose run --rm trainer train.py --help
docker compose run --rm --entrypoint bash trainer        # interactive shell
```

`data/`, `checkpoints/`, `runs/` and `wandb/` are bind‑mounted (see
[docker-compose.yml](docker-compose.yml)); all GPUs are exposed to the container.

---

## Dataset setup

The dataset (2400 frames, 2560×1440, YOLO‑format labels, exactly two drones per
frame) is **not** committed due to upload limitations. Place `dataset.zip` in `data/` and materialise it:

```bash
cd data
unzip dataset.zip -d extracted_raw          # -> data/extracted_raw/curated_datasets/obj_det_base/...
python split_dataset.py                     # flattens into data/images/ and data/labels/
cd ..
```

You should end with `data/images/*.png` and `data/labels/*.txt` (2400 each).

> Dataset download URL: [_Link_](https://drive.google.com/file/d/19L9yUP62xMESJMw6srf5HGcL8s5b0gv8/view)

### Freeze the train/val/test split

The corpus is really ~60 scenes, not 2400 independent images (same‑scene frames
are near‑duplicates; every `augmented_*` frame is a variant of a `raw_*` frame).
Splitting by image leaks. We split **whole scenes**, stratified by condition:

```bash
make split                                  # writes splits/seed42_48-6-6.json (48/6/6 scenes = 1929/243/228 frames)
python -m src.core.split data/images         # just inspect the split
```

The manifest is committed, so results are reproducible without regenerating it.

Otherwise, you can download the scenes split and use them directly from this link:
> Split, model weights, and model runs download URL: [_Link_](https://drive.google.com/file/d/1EnpRFnNKuk4s5kOV7qHstHw0chJt4q-V/view?usp=sharing)

---

## How to run

`train.py` is the single entrypoint. Checkpoints land in
`checkpoints/<run-name>_best.pth` (run‑name defaults to `<model>_<loss>`).

### 1. Train the from‑scratch stages

```bash
# Stage 1 — single-scale baseline
python train.py --model baseline --epochs 50 --run-name baseline --split-manifest splits/seed42_48-6-6.json

# Stage 2 — multi-scale FPN
python train.py --model fpn --epochs 50 --run-name fpn --split-manifest splits/seed42_48-6-6.json

# Stage 3 — + P2 high-resolution head
python train.py --model p2  --epochs 50 --run-name p2  --split-manifest splits/seed42_48-6-6.json
```

Models: `baseline | fpn | p2` are the submission candidates; `ab2d` is a heavier
from‑scratch benchmark (attention + BiFPN), context only.

### 2. Stage 3b — swap in the decoupled DFL head (warm‑start backbone + neck)

```bash
python train.py --mode refine --model p2 \
    --base-weights checkpoints/p2_best.pth \
    --epochs 50 --peak-lr 4e-4 --llrd 0.9 --eval-split test --profile \
    --save-model models_final/p2_v2.pt          # self-contained; feeds Stage 4
```

### 3. Stage 4 — task‑aligned final model, then the frozen‑backbone fine‑tune

```bash
# task-aligned training (dynamic assignment + Varifocal + MPDIoU), warm-started from a refined checkpoint
python train.py --mode final --base-weights models_final/p2_v2.pt \
    --epochs 50 --iou-type mpdiou --topk 10 --eval-split test --profile \
    --run-name final_model --save-model models_final/final_model_stage4.pt

# short EMA polish with the backbone frozen  ->  the SUBMITTED model
python train.py --mode final --fine-tune --freeze-backbone \
    --base-weights checkpoints/final_model_best.pth \
    --epochs 20 --peak-lr 1e-4 --eval-split test \
    --save-model models_final/final_model.pt          # <- the submitted best model
```

### 4. Evaluate any checkpoint

```bash
python eval.py --weights checkpoints/fpn_best.pth --split test --profile
# prints mAP@0.5, mAP@0.5:0.95, AP by drone size, per-condition mAP, best-F1 P/R/F1,
# per-image count MAE, and (with --profile) params / FLOPs / model size / peak train mem / latency
```

### 5. Run the trained detector on images

```bash
python infer.py --weights models_final/final_model.pt --source data/images/some_frame.png --out runs/infer --save-txt
```

### 6. Reproduce the whole comparison

```bash
make split
bash scripts/run_full_matrix.sh        # 50-epoch runs: {baseline,fpn,p2,ab2d} x {centre-only, centre+neighbours} + YOLO refs
bash scripts/run_refine.sh             # pick top-2, LR/LLRD search, 50-epoch refined heads (needs run_full_matrix first)
python scripts/compare.py runs/full/*_test.json runs/refine/final/*_test.json   # one markdown comparison table
```

Common Makefile shortcuts: `make train MODEL=fpn`, `make eval WEIGHTS=... SPLIT=test`,
`make infer WEIGHTS=... SOURCE=...`, `make help`.

---

## The pipeline, intuitively

1. **Split by scene, not by image.** A difference‑hash study shows same‑scene
   frames have a median Hamming distance of 0. [src/core/split.py](src/core/split.py)
   partitions whole scenes so train/test never share a rendered view.
2. **Letterbox + augment.** Frames go to 640×640 (aspect preserved). Train‑only
   augmentation: flip, shift‑scale‑rotate, CLAHE, brightness/contrast, hue/sat.
3. **Backbone → neck → heads.** A stride‑2 stem + four residual stages (base
   width 32) feed a top‑down FPN; detection heads sit on strides 4–32.
4. **Grow the pyramid and the objective together.** Single head → FPN → add the
   stride‑4 head for tiny drones → replace point‑target regression with a
   **decoupled head that predicts a distribution over each box edge** (DFL).
5. **Task‑aligned finish.** Drop the stride‑32 head (no drone is that big),
   add a squeeze‑and‑excitation gate, and switch to a **dynamic assigner**
   (`t = s^0.5 · IoU^6`, top‑10) with Varifocal objectness and an MPDIoU box
   term. A 20‑epoch frozen‑backbone fine‑tune polishes it.
6. **Evaluate one way for everyone.** [src/core/metrics.py](src/core/metrics.py)
   scores every model — including the YOLO references — through the same
   all‑points‑interpolation evaluator.

See [paper/paper.pdf](paper/paper.pdf) for the figures and the
stage‑by‑stage ablations.

---

## Models produced

| Name | File | Stage | Notes |
|------|------|:-----:|-------|
| `baseline` | [src/models/single_scale.py](src/models/single_scale.py) | 1 | single P5 head |
| `fpn` | [src/models/multi_scale_fpn.py](src/models/multi_scale_fpn.py) | 2 | P3–P5 pyramid |
| `p2` | [src/models/p2_granular.py](src/models/p2_granular.py) | 3 | + stride‑4 head |
| refined head | [src/models/v2.py](src/models/v2.py) | 3b | decoupled DFL head on `fpn`/`p2` |
| **`final_model`** | [src/models/final_model.py](src/models/final_model.py) | **4b** | **submitted best model** |
| `ab2d` | [src/models/reference/ab2d_yolo.py](src/models/reference/ab2d_yolo.py) | – | from‑scratch attention/BiFPN benchmark |

Trained weights are git‑ignored. Ship them via **Git LFS** (`git lfs track "*.pt" "*.pth"`)
or upload separately:

> Model weights download URL: [_Link_](https://drive.google.com/file/d/1EnpRFnNKuk4s5kOV7qHstHw0chJt4q-V/view?usp=sharing)

---

## Bonus deliverables

- **Multi‑GPU** — [src/engine/multigpu_trainer.py](src/engine/multigpu_trainer.py)
  is a DistributedDataParallel trainer class (per‑rank `DistributedSampler`,
  gradient sync via DDP, rank‑0 evaluation through the shared evaluator). It was
  developed on a single‑GPU pod and is not wired to a CLI entrypoint yet — drive
  it from a small `torchrun` launch script and give it a validation pass before
  relying on it.
- **Containerization & orchestration** — [Dockerfile](Dockerfile) (CUDA 12.8
  runtime, pinned deps) + [docker-compose.yml](docker-compose.yml) (GPU
  reservation, shared‑memory sizing for DataLoader workers, volume mounts,
  `WANDB_API_KEY` passthrough).
- **Clean code / OOP** — single‑responsibility `src/` layout, one architecture
  per file over a shared `BaseDetector`, a `build_model / build_criterion /
  build_dataloaders` factory, and one shared evaluator used by every trainer and
  by `eval.py`.

---

## Paper

`paper/paper.pdf`

---

## Experiment tracking (Weights & Biases)

Every run logs to the `drone-detection` W&B project (config + per‑epoch metrics +
best checkpoint). Set `WANDB_API_KEY` to enable it, or pass `--no-wandb` to
disable.

```bash
export WANDB_API_KEY=...
python train.py --model fpn --epochs 50            # logs to W&B
python train.py --model fpn --epochs 50 --no-wandb # offline
```

> W&B project (shareable): [_Link_](https://wandb.ai/gedeprasidha-insitut-teknologi-bandung/drone-detection)
