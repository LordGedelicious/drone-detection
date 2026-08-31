#!/usr/bin/env bash
# Full 50-epoch runs on the frozen split, evaluated on the held-out TEST split:
#   {baseline, fpn, p2, ab2d}  x  {single-cell, centre+neighbours}   = 8 runs
# then yolov8n {pretrained, from-scratch} as reference benchmarks.
#
# Greedy scheduler: at most MAXJOBS concurrent, and a second job only starts if
# the combined GPU-memory cost stays under BUDGET (measured peaks from the
# screening pass). p2 / ab2d therefore always run solo; the light models pair up.
set -o pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export YOLO_CONFIG_DIR=/tmp/Ultralytics
MANIFEST="${MANIFEST:-splits/seed42_48-6-6.json}"
EPOCHS="${EPOCHS:-50}"
RES=runs/full
CKPT=checkpoints/full
mkdir -p "$RES" "$CKPT"

BUDGET=11000          # MiB — combined cost ceiling for concurrent jobs
MAXJOBS=2
declare -A COST=([baseline]=4300 [fpn]=5500 [p2]=13500 [ab2d]=20500)

JOBS=(baseline:single baseline:neighbors fpn:single fpn:neighbors
      p2:single p2:neighbors ab2d:single ab2d:neighbors)

declare -A PID_COST
used=0

reap() {
  (( ${#PID_COST[@]} )) || return 0
  local pid
  for pid in "${!PID_COST[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      used=$(( used - ${PID_COST[$pid]} ))
      unset "PID_COST[$pid]"
    fi
  done
}

run_one() {
  local m=$1 a=$2 tag="${m}_${a}"
  local flag; [ "$a" = neighbors ] && flag=--neighbor-cells || flag=--no-neighbor-cells
  {
    echo "[$(date +%H:%M)] START $tag ($EPOCHS ep)"
    python train.py --model "$m" $flag --epochs "$EPOCHS" --split-manifest "$MANIFEST" \
        --ckpt-dir "$CKPT/$tag" --workers 6 --run-name "full_${tag}"
    if [ -f "$CKPT/$tag/${m}_best.pth" ]; then
      python eval.py --weights "$CKPT/$tag/${m}_best.pth" --split test --profile \
          --split-manifest "$MANIFEST" --tag "$tag" --json "$RES/${tag}_test.json"
    else
      echo "[$(date +%H:%M)] !! no checkpoint for $tag — training failed, skipping eval"
    fi
    echo "[$(date +%H:%M)] DONE  $tag"
  } > "$RES/${tag}.log" 2>&1
}

echo "[$(date +%H:%M)] full matrix start — $EPOCHS epochs, ${#JOBS[@]} custom runs"
for job in "${JOBS[@]}"; do
  m="${job%%:*}"; a="${job##*:}"; c="${COST[$m]}"
  while :; do
    reap
    n=${#PID_COST[@]}
    if [ "$n" -lt "$MAXJOBS" ] && { [ "$n" -eq 0 ] || [ $(( used + c )) -le "$BUDGET" ]; }; then
      break
    fi
    sleep 30
  done
  run_one "$m" "$a" &
  PID_COST[$!]=$c
  used=$(( used + c ))
  echo "[$(date +%H:%M)] launched ${m}_${a}  (cost ${c}MiB, in-flight ${n_now:-$(( n + 1 ))}, budget-used ${used})"
  sleep 45
done
wait
echo "[$(date +%H:%M)] all ${#JOBS[@]} custom runs done"

# --- YOLO reference benchmarks (isolated venv), same epoch budget ---
if [ -x /opt/ultra-venv/bin/python ]; then
  echo "[$(date +%H:%M)] YOLO benchmarks ($EPOCHS ep)"
  PYTHONPATH=. /opt/ultra-venv/bin/python scripts/run_yolo_benchmark.py \
      --epochs "$EPOCHS" --split test --out "$RES" > "$RES/yolo.log" 2>&1 \
      || echo "[$(date +%H:%M)] YOLO benchmark failed (see $RES/yolo.log)"
fi

echo; echo "########## FULL ${EPOCHS}-EPOCH TEST-SET RESULTS ##########"
python scripts/compare.py "$RES"/*_test.json | tee "$RES/full_table.md"
echo "[$(date +%H:%M)] FULL MATRIX COMPLETE"
