"""
Leak-free, deterministic train/val/test splitting.

Why this module exists
----------------------
``data/images`` holds 2400 frames, but they are not 2400 independent samples:

* Frames that share the ``<env>_<weather>_<a>_<b>`` token ("capture segment", or
  *scene*) are near-identical -- a perceptual-hash probe found the median
  Hamming distance between same-scene raw frames is 0. A scene is one rendered
  view, sampled ~20 times with the drone nudged a few pixels.
* Every ``augmented_*`` frame is a photometric/geometric variant of a ``raw_*``
  frame from the *same* scene (99% within Hamming distance 8).

So the dataset is effectively **60 independent scenes** (6 env/weather
conditions x 10 scenes). Splitting on individual frames -- even stratified --
scatters near-duplicates and augmented twins across train/val/test and inflates
every metric. This module partitions **whole scenes**, stratified by condition,
and keeps a scene's raw and augmented frames together on the same side.
"""

import os
import re
import json
import random
from collections import defaultdict

IMG_EXTENSIONS = (".png",)

# Scene id = everything between the dataset prefix and "_sequence."
_SCENE_RE = re.compile(r"^(?:augmented_)?raw_dataset_(.+?)_sequence\.")


def list_images(img_dir: str) -> list:
    """Sorted list of image basenames (deterministic ordering)."""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    return sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTENSIONS))


def is_augmented(filename: str) -> bool:
    return os.path.basename(filename).startswith("augmented_")


def scene_of(filename: str) -> str:
    """Capture-segment id, e.g. 'city_foggy_city_foggy_0_1'. Unmatched names
    become their own singleton scene (still leak-safe)."""
    m = _SCENE_RE.match(os.path.basename(filename))
    return m.group(1) if m else os.path.basename(filename)


def condition_of(name: str) -> str:
    """Environment/weather bucket for stratification and per-condition reporting,
    e.g. 'city_foggy'. Accepts a filename or a scene id."""
    scene = scene_of(name) if name.endswith(IMG_EXTENSIONS) else name
    parts = scene.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else scene


def _balanced_counts(total: int, n_buckets: int) -> list:
    """n_buckets ints summing to total, as even as possible (larger buckets first)."""
    base, rem = divmod(total, n_buckets)
    return [base + 1] * rem + [base] * (n_buckets - rem)


def make_splits(
    img_dir: str,
    scene_counts=(48, 6, 6),
    seed: int = 42,
    manifest_path: str = None,
) -> dict:
    """
    Deterministic scene-grouped, condition-stratified train/val/test split.

    Parameters
    ----------
    img_dir       : directory of image files.
    scene_counts  : (n_train, n_val, n_test) as a count of *scenes* (not frames).
                    Must sum to the number of scenes on disk (60). Each count is
                    spread as evenly as possible across the conditions, so e.g.
                    (48, 6, 6) puts 8/1/1 scenes per condition in train/val/test.
    seed          : controls which scenes land where. Fixed default -> the split
                    is reproducible across machines and runs.
    manifest_path : if given and the file exists, the split is loaded from it
                    verbatim (frozen split). If given and absent, the computed
                    split is written there.

    Returns
    -------
    {"train": [...], "val": [...], "test": [...]}  -- sorted image basenames,
    no scene shared between splits.
    """
    files = list_images(img_dir)
    if not files:
        raise RuntimeError(f"No {'/'.join(IMG_EXTENSIONS)} images found in {img_dir}")

    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            split = json.load(fh)
        return {k: sorted(split[k]) for k in ("train", "val", "test")}

    frames_by_scene = defaultdict(list)
    for f in files:
        frames_by_scene[scene_of(f)].append(f)

    scenes_by_cond = defaultdict(list)
    for s in frames_by_scene:
        scenes_by_cond[condition_of(s)].append(s)

    n_scenes = len(frames_by_scene)
    if sum(scene_counts) != n_scenes:
        raise ValueError(
            f"scene_counts {scene_counts} sum to {sum(scene_counts)}, "
            f"but {n_scenes} scenes exist in {img_dir}"
        )

    conditions = sorted(scenes_by_cond)
    per_cond = {
        split_name: _balanced_counts(count, len(conditions))
        for split_name, count in zip(("train", "val", "test"), scene_counts)
    }

    rng = random.Random(seed)
    assign = {"train": [], "val": [], "test": []}
    for ci, cond in enumerate(conditions):
        scenes = sorted(scenes_by_cond[cond])
        rng.shuffle(scenes)
        n_val = per_cond["val"][ci]
        n_test = per_cond["test"][ci]
        need = n_val + n_test
        if need > len(scenes):
            raise ValueError(
                f"condition '{cond}' has {len(scenes)} scenes but the split needs "
                f"{need} for val+test; adjust scene_counts"
            )
        assign["val"] += scenes[:n_val]
        assign["test"] += scenes[n_val:n_val + n_test]
        assign["train"] += scenes[n_val + n_test:]

    split = {
        name: sorted(f for s in scene_list for f in frames_by_scene[s])
        for name, scene_list in assign.items()
    }

    # Hard guarantee: no scene appears in more than one split.
    seen = {}
    for name, scene_list in assign.items():
        for s in scene_list:
            assert s not in seen, f"scene {s} in both {seen[s]} and {name}"
            seen[s] = name

    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w") as fh:
            json.dump({**split, "_meta": {"seed": seed, "scene_counts": list(scene_counts)}}, fh, indent=1)

    return split


def describe_splits(img_dir: str, **kwargs) -> str:
    split = make_splits(img_dir, **kwargs)
    lines = []
    for name in ("train", "val", "test"):
        files = split[name]
        scenes = sorted({scene_of(f) for f in files})
        raw = sum(not is_augmented(f) for f in files)
        by_cond = defaultdict(int)
        for s in scenes:
            by_cond[condition_of(s)] += 1
        cond_str = " ".join(f"{c.split('_')[0][0]}{c.split('_')[1][0]}:{n}" for c, n in sorted(by_cond.items()))
        lines.append(
            f"{name:<5} {len(scenes):>2} scenes | {len(files):>4} frames "
            f"({raw} raw + {len(files) - raw} aug) | per-cond {cond_str}"
        )
    # cross-split scene overlap
    scenesets = {n: {scene_of(f) for f in split[n]} for n in split}
    overlap = (scenesets["train"] & scenesets["val"]) | (scenesets["train"] & scenesets["test"]) | (scenesets["val"] & scenesets["test"])
    lines.append(f"scene overlap between splits: {len(overlap)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect / freeze the train/val/test split.")
    ap.add_argument("img_dir", nargs="?", default="data/images")
    ap.add_argument("--scenes", type=int, nargs=3, default=(48, 6, 6), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", metavar="PATH", help="write the split to a JSON manifest")
    args = ap.parse_args()

    print(describe_splits(args.img_dir, scene_counts=tuple(args.scenes), seed=args.seed))
    if args.write:
        make_splits(args.img_dir, scene_counts=tuple(args.scenes), seed=args.seed, manifest_path=args.write)
        print(f"\nwrote manifest -> {args.write}")
