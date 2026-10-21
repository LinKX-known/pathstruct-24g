#!/usr/bin/env bash
# One-shot pilot on a 24 GB card.
#
#   bash scripts/run_pilot.sh                 # synthetic cohort, ~10 min on a 4090
#   DATA=real bash scripts/run_pilot.sh       # uses data/real/manifest.csv + cells/
#
# Every step is idempotent: rerunning skips what already exists unless FORCE=1.

set -euo pipefail

PRESET="${PRESET:-4b_bf16}"
STEPS_P1="${STEPS_P1:-150}"
STEPS_P2="${STEPS_P2:-300}"
LIMIT="${LIMIT:-24}"
FORCE="${FORCE:-0}"

echo "== hardware =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

echo "== environment =="
pilot check

if [[ "${DATA:-synthetic}" == "synthetic" ]]; then
  ROOT="data/synth"
  if [[ ! -f "$ROOT/manifest.csv" || "$FORCE" == "1" ]]; then
    echo "== [1/6] synthetic cohort =="
    pilot synth --out "$ROOT" --per-class 12 --size 384
  fi
  if [[ ! -d "$ROOT/cells" || "$FORCE" == "1" ]]; then
    echo "== [2/6] segment (classical, no weights) =="
    pilot segment --images "$ROOT/images" --out "$ROOT/cells" --backend classical
  fi
else
  ROOT="data/real"
  echo "== [1/6] using $ROOT (manifest + cells must already exist) =="
fi

echo "== [3/6] doctor =="
pilot doctor --manifest "$ROOT/manifest.csv" --cells "$ROOT/cells" || {
  echo "doctor failed: fix the problems above before training"; exit 1;
}

echo "== [4/6] build datasets =="
if [[ ! -f "$ROOT/growth.jsonl" || "$FORCE" == "1" ]]; then
  pilot build --manifest "$ROOT/manifest.csv" --cells "$ROOT/cells" --out "$ROOT" --per-roi 6
fi

echo "== [5/6] P1: counterfactual task =="
if [[ ! -d out/p1 || "$FORCE" == "1" ]]; then
  pilot train --cf "$ROOT/cf.jsonl" --steps "$STEPS_P1" --preset "$PRESET" --out out/p1
fi
pilot eval --data "$ROOT/cf.jsonl" --adapter out/p1 --preset "$PRESET" --limit "$LIMIT"

echo "== [6/6] P2: aux vs LoRA-only =="
if [[ ! -d out/p2_aux || "$FORCE" == "1" ]]; then
  pilot train --growth "$ROOT/growth.jsonl" --cf "$ROOT/cf.jsonl" --steps "$STEPS_P2" --preset "$PRESET" --out out/p2_aux
fi
if [[ ! -d out/p2_lora || "$FORCE" == "1" ]]; then
  pilot train --growth "$ROOT/growth.jsonl" --cf "$ROOT/cf.jsonl" --steps "$STEPS_P2" --preset "$PRESET" --out out/p2_lora --no-aux
fi

pilot eval --data "$ROOT/growth.jsonl" --adapter out/p2_aux  --out out/pred_aux.json  --preset "$PRESET" --limit "$LIMIT"
pilot eval --data "$ROOT/growth.jsonl" --adapter out/p2_lora --out out/pred_lora.json --preset "$PRESET" --limit "$LIMIT" --compare out/pred_aux.json

echo
echo "done. decide:"
echo "  P1 balanced acc >= 0.65 -> the task is learnable"
echo "  P2 delta > +2 points   -> the aux signal transfers, scale up"
