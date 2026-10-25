#!/usr/bin/env bash
# Real H&E only. No synthetic cohort anywhere in this script.
#
#   ROOT=data/real/pannuke bash scripts/run_real.sh
#   ROOT=data/real/monuseg PRESET=4b_bf16_1024 bash scripts/run_real.sh
#
# It runs the two arms that decide the project:
#   A  LoRA-only baseline on the diagnostic QA      -> macro-F1
#   B  evidence tokens + observation + answer       -> macro-F1
# Same base model, same data, same steps; only the architecture differs.
#
# Prerequisites (produced by the data importers, not by this script):
#   $ROOT/manifest_train.csv  $ROOT/manifest_val.csv
#   $ROOT/cells/{roi_id}.csv  (CellViT output, converted with `pilot cells`)
#   $ROOT/growth_train.jsonl  $ROOT/growth_val.jsonl  (pilot build)
#   $ROOT/cf_train_bal.jsonl                          (balanced counterfactuals)

set -euo pipefail

ROOT="${ROOT:-data/real/pannuke}"
CELLS="${CELLS:-$ROOT/cells}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/manifest_train.csv}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT/manifest_val.csv}"
TRAIN_QA="${TRAIN_QA:-$ROOT/growth_train.jsonl}"
VAL_QA="${VAL_QA:-$ROOT/growth_val.jsonl}"
CF_TRAIN="${CF_TRAIN:-$ROOT/cf_train_bal.jsonl}"
PRESET="${PRESET:-4b_bf16_1024}"
STEPS="${STEPS:-600}"
OUT="${OUT:-out/real}"

for f in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$TRAIN_QA" "$VAL_QA"; do
  if [[ ! -f "$f" ]]; then echo "missing $f -- run the importers and 'pilot build' first"; exit 2; fi
done
if [[ ! -d "$CELLS" ]]; then echo "missing cell tables at $CELLS (CellViT -> pilot cells)"; exit 2; fi

mkdir -p "$OUT"

echo "== doctor =="
pilot doctor --manifest "$TRAIN_MANIFEST" --cells "$CELLS"

echo "== arm A: LoRA-only baseline =="
pilot train --growth "$TRAIN_QA" --steps "$STEPS" --preset "$PRESET" --no-aux --out "$OUT/A_lora"
pilot eval --data "$VAL_QA" --adapter "$OUT/A_lora" --preset "$PRESET" \
  --out "$OUT/A_pred.json" | tee "$OUT/A_metrics.txt"

echo "== building the evidence corpus (real images only) =="
python scripts/build/build_structure_qa.py \
  --manifest "$TRAIN_MANIFEST" --cells "$CELLS" \
  --out "$OUT/struct_observe_train.jsonl" \
  --wrap "$TRAIN_QA" --out-wrapped "$OUT/struct_growth_train.jsonl"

# training mix: observation rows + the same diagnostic rows as arm A
cat "$OUT/struct_observe_train.jsonl" "$OUT/struct_growth_train.jsonl" > "$OUT/struct_all_train.jsonl"

python scripts/build/build_structure_qa.py \
  --manifest "$VAL_MANIFEST" --cells "$CELLS" \
  --out "$OUT/struct_observe_val.jsonl" \
  --wrap "$VAL_QA" --out-wrapped "$OUT/struct_growth_val.jsonl"

echo "== arm B: evidence tokens =="
python scripts/train/train_structure_vlm.py --data "$OUT/struct_all_train.jsonl" \
  --stage perceive --preset "$PRESET" --steps 100 --out "$OUT/B_perceive"
python scripts/train/train_structure_vlm.py --data "$OUT/struct_all_train.jsonl" \
  --stage reason --preset "$PRESET" --steps "$STEPS" --out "$OUT/B_reason"

echo "== arm B: evaluation on the same items as arm A =="
python scripts/train/train_structure_vlm.py --eval-only \
  --data "$OUT/struct_growth_val.jsonl" --adapter "$OUT/B_reason" --preset "$PRESET" \
  | tee "$OUT/B_metrics.txt"

echo
echo "compare macro-F1:"
echo "  A: grep -i macro $OUT/A_metrics.txt"
echo "  B: grep -i answer_accuracy $OUT/B_metrics.txt   (plus observation accuracy)"
echo "decision: B - A >= 2 points -> continue; otherwise stop and rethink"
