#!/usr/bin/env bash
# V2 refinement pipeline (runs after the full matrix):
#   1. pick the top-2 non-reference models by test mAP@0.5:0.95
#   2. Stage 1 — small LR / layer-wise-decay search, 15-epoch val runs
#   3. Stage 2 — best config, 50-epoch run on the test split + profile
#   4. rebuild the comparison table including <model>_v2
set -o pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="${MANIFEST:-splits/seed42_48-6-6.json}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-15}"
FINAL_EPOCHS="${FINAL_EPOCHS:-50}"
RES=runs/refine
mkdir -p "$RES/search" "$RES/final" checkpoints/v2 models_final

# LR x layer-wise-decay(gamma) configs to try
CONFIGS=("1e-4 0.9" "2e-4 0.85" "2e-4 0.9" "4e-4 0.8" "4e-4 0.9")

mapfile -t TOP < <(python scripts/pick_top2.py 2)
echo "[$(date +%H:%M)] top-2 to refine:"; printf '  %s\n' "${TOP[@]}"

declare -A BEST_CFG
for line in "${TOP[@]}"; do
  read -r model ckpt score run <<< "$line"
  echo "[$(date +%H:%M)] === Stage 1 search: $model (base $run, mAP50-95 $score) ==="
  for cfg in "${CONFIGS[@]}"; do
    read -r lr gamma <<< "$cfg"
    j="$RES/search/${model}_lr${lr}_g${gamma}.json"
    [ -f "$j" ] && continue
    python train.py --mode refine --model "$model" --base-weights "$ckpt" --epochs "$SEARCH_EPOCHS" \
        --peak-lr "$lr" --llrd "$gamma" --eval-split val --no-wandb \
        --split-manifest "$MANIFEST" --ckpt-dir "checkpoints/v2/search_${model}" \
        --json "$j" --run-name "${model}_v2_search_lr${lr}_g${gamma}" \
        > "$RES/search/${model}_lr${lr}_g${gamma}.log" 2>&1
  done
  BEST_CFG[$model]=$(python - "$model" <<'PY'
import glob, json, sys
m = sys.argv[1]; best = (-1, None)
for f in glob.glob(f"runs/refine/search/{m}_lr*_g*.json"):
    d = json.load(open(f)); s = d.get("mAP_50_95", 0)
    if s > best[0]:
        import re
        mm = re.search(r"_lr([0-9e.-]+)_g([0-9.]+)\.json$", f)
        best = (s, (mm.group(1), mm.group(2)))
print(f"{best[1][0]} {best[1][1]} {best[0]:.4f}" if best[1] else "2e-4 0.85 0")
PY
)
  echo "[$(date +%H:%M)] $model best config: ${BEST_CFG[$model]}"
done

for line in "${TOP[@]}"; do
  read -r model ckpt score run <<< "$line"
  read -r lr gamma sc <<< "${BEST_CFG[$model]}"
  echo "[$(date +%H:%M)] === Stage 2 final: ${model}_v2  (lr=$lr gamma=$gamma, search mAP50-95 $sc) ==="
  python train.py --mode refine --model "$model" --base-weights "$ckpt" --epochs "$FINAL_EPOCHS" \
      --peak-lr "$lr" --llrd "$gamma" --eval-split test --profile \
      --split-manifest "$MANIFEST" --ckpt-dir "checkpoints/v2/${model}_v2" \
      --final-dir models_final --json "$RES/final/${model}_v2_test.json" \
      --run-name "${model}_v2" > "$RES/final/${model}_v2.log" 2>&1
done

echo; echo "########## V2 vs BASELINE — TEST SET ##########"
python scripts/compare.py runs/full/*_test.json "$RES"/final/*_v2_test.json | tee "$RES/v2_table.md"
echo "[$(date +%H:%M)] REFINEMENT COMPLETE"
