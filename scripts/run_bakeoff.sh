#!/usr/bin/env bash
# Architecture bake-off: baseline vs fpn vs p2, under an identical config.
#
#   scripts/run_bakeoff.sh screen [manifest]          # 3 models x 2 label-assignment modes, ~20ep, eval on val
#   scripts/run_bakeoff.sh full   [manifest] <model> [neighbors|single]   # ~50ep, eval on test + profile
#
# All runs share one frozen scene split so the comparison is fair. The screening
# pass also ablates the label-assignment rule (centre+neighbours vs centre-only).
set -euo pipefail

MODE="${1:-screen}"
MANIFEST="${2:-splits/seed42_48-6-6.json}"
read -r -a MODELS <<< "${BAKEOFF_MODELS:-baseline fpn p2}"
EPOCHS_SCREEN="${EPOCHS_SCREEN:-20}"
EPOCHS_FULL="${EPOCHS_FULL:-50}"
RESULTS_DIR="runs/bakeoff"
PY="${PY:-python}"

mkdir -p "$RESULTS_DIR" "$(dirname "$MANIFEST")"
[ -f "$MANIFEST" ] || $PY -m src.core.split --write "$MANIFEST"

assign_flag() { [ "$1" = "neighbors" ] && echo "--neighbor-cells" || echo "--no-neighbor-cells"; }

train_and_eval() {  # <model> <assign> <epochs> <split>
  local m="$1" assign="$2" epochs="$3" split="$4"
  local tag="${m}_${assign}" ckpt="checkpoints/bakeoff/${assign}"
  echo "=== ${tag}: ${epochs}ep -> eval on ${split} ==="
  $PY train.py --model "$m" $(assign_flag "$assign") --epochs "$epochs" \
       --split-manifest "$MANIFEST" --ckpt-dir "$ckpt" --run-name "bakeoff_${tag}"
  $PY eval.py --weights "${ckpt}/${m}_best.pth" --split "$split" --profile \
       --split-manifest "$MANIFEST" --tag "$tag" --json "${RESULTS_DIR}/${tag}_${split}.json"
}

case "$MODE" in
  screen)
    for assign in neighbors single; do
      for m in "${MODELS[@]}"; do
        train_and_eval "$m" "$assign" "$EPOCHS_SCREEN" val
      done
    done
    echo; echo "########## SCREENING COMPARISON ##########"
    $PY scripts/compare.py "${RESULTS_DIR}"/*_val.json | tee "${RESULTS_DIR}/screen_table.md"
    ;;
  full)
    MODEL="${3:?usage: run_bakeoff.sh full <manifest> <model> [neighbors|single]}"
    ASSIGN="${4:-neighbors}"
    train_and_eval "$MODEL" "$ASSIGN" "$EPOCHS_FULL" test
    echo; echo "########## FULL-RUN RESULT ##########"
    $PY scripts/compare.py "${RESULTS_DIR}/${MODEL}_${ASSIGN}_test.json" | tee "${RESULTS_DIR}/full_${MODEL}_${ASSIGN}.md"
    ;;
  *)
    echo "unknown mode '$MODE' (expected: screen | full)" >&2
    exit 2
    ;;
esac
