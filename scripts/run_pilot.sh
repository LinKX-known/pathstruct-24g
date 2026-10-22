#!/usr/bin/env bash
# One-shot pilot on a 24 GB card.
#
#   bash scripts/run_pilot.sh                 # synthetic cohort
#   DATA=real bash scripts/run_pilot.sh       # uses data/real/manifest.csv + cells/
#
# Every step is idempotent: rerunning skips what already exists unless FORCE=1.
# P0/P1/P2 all evaluate a patient-disjoint holdout (built by `pilot build`).

set -euo pipefail

PRESET="${PRESET:-4b_bf16}"
STEPS_P1="${STEPS_P1:-150}"
STEPS_P2="${STEPS_P2:-300}"
PER_CLASS="${PER_CLASS:-12}"
VAL_FRAC="${VAL_FRAC:-0.3}"
LIMIT="${LIMIT:-0}"
FORCE="${FORCE:-0}"

LIMIT_ARG=()
if [[ "$LIMIT" -gt 0 ]]; then LIMIT_ARG=(--limit "$LIMIT"); fi

echo "== hardware =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

echo "== environment =="
pilot check

if [[ "${DATA:-synthetic}" == "synthetic" ]]; then
  ROOT="data/synth"
  CELLS="$ROOT/cells_seg"
  if [[ ! -f "$ROOT/manifest.csv" || "$FORCE" == "1" ]]; then
    echo "== [1/6] synthetic cohort =="
    pilot synth --out "$ROOT" --per-class "$PER_CLASS" --size 384
  fi
  if [[ ! -d "$CELLS" || "$FORCE" == "1" ]]; then
    echo "== [2/6] segment (classical, no weights) =="
    # Segment into a separate directory: the ground-truth tables written by
    # `pilot synth` live in $ROOT/cells and must never mask a segmentation
    # failure.
    pilot segment --images "$ROOT/images" --out "$CELLS" --backend classical
  fi
else
  ROOT="data/real"
  CELLS="$ROOT/cells"
  echo "== [1/6] using $ROOT (manifest + cells must already exist) =="
fi

echo "== [3/6] doctor =="
pilot doctor --manifest "$ROOT/manifest.csv" --cells "$CELLS" || {
  echo "doctor failed: fix the problems above before training"; exit 1;
}

echo "== [4/6] build datasets + patient split =="
if [[ ! -f "$ROOT/cf_train.jsonl" || "$FORCE" == "1" ]]; then
  pilot build --manifest "$ROOT/manifest.csv" --cells "$CELLS" --out "$ROOT" \
    --per-roi 6 --val-frac "$VAL_FRAC"
fi

echo "== [5/6] P0 + P1: counterfactual task =="
pilot p0 --data "$ROOT/cf_val.jsonl" --preset "$PRESET" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}
if [[ ! -d out/p1 || "$FORCE" == "1" ]]; then
  pilot train --cf "$ROOT/cf_train.jsonl" --steps "$STEPS_P1" --preset "$PRESET" --out out/p1
fi
pilot eval --data "$ROOT/cf_val.jsonl" --adapter out/p1 --preset "$PRESET" \
  --out out/pred_p1.json ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}

echo "== [6/6] P2: aux vs LoRA-only (same data mix, only the loss differs) =="
if [[ ! -d out/p2_aux || "$FORCE" == "1" ]]; then
  pilot train --growth "$ROOT/growth_train.jsonl" --cf "$ROOT/cf_train.jsonl" \
    --steps "$STEPS_P2" --preset "$PRESET" --out out/p2_aux
fi
if [[ ! -d out/p2_lora || "$FORCE" == "1" ]]; then
  pilot train --growth "$ROOT/growth_train.jsonl" --cf "$ROOT/cf_train.jsonl" \
    --steps "$STEPS_P2" --preset "$PRESET" --out out/p2_lora --no-aux
fi

pilot eval --data "$ROOT/growth_val.jsonl" --adapter out/p2_aux  --out out/pred_aux.json  \
  --preset "$PRESET" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}
pilot eval --data "$ROOT/growth_val.jsonl" --adapter out/p2_lora --out out/pred_lora.json \
  --preset "$PRESET" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} --compare out/pred_aux.json

echo
echo "done. decide:"
echo "  P1 balanced acc >= 0.65 (patient-disjoint val) -> the task is learnable"
echo "  P2 delta > +2 points                            -> the aux signal transfers, scale up"
