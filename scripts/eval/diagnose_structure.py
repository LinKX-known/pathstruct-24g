#!/usr/bin/env python
"""CPU-only diagnosis of the structure head's held-out predictions.

This is the ten-minute check that should have preceded the "structure head
failed" conclusion.  It consumes the JSON arrays written by
``scripts/eval/eval_heads.py --out`` and the training JSONL used for that run,
then reports, per descriptor dimension:

* the model's R^2 *and* the R^2 of a constant train-mean predictor;
* skill relative to that predictor;
* log-space MAE, MAE ratio to the mean predictor, Spearman rho, calibration;
* presence AUC / positive-only metrics for zero-inflated dimensions.

It needs no GPU and no model.  Example::

    python scripts/eval/diagnose_structure.py \
        --pred out/heads_topo_real_s0.json \
        --train-data data/real/monuseg_train/cf_train_bal.jsonl \
        --dims 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.structure_eval import (  # noqa: E402
    diagnostics_payload,
    format_table,
    log1p_target,
    per_dim_diagnostics,
)

TOPO_KEYS = ["n_comp", "hole_count", "hole_area_frac", "h0_max", "h0_sum"]


def read_rows(path: str | Path):
    text = Path(path).read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dim_names(d: int):
    return TOPO_KEYS[:d] + [f"dim{i}" for i in range(len(TOPO_KEYS), d)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="heads_*.json (repeatable)")
    ap.add_argument(
        "--train-data",
        action="append",
        required=True,
        help="training JSONL with `descriptor` (repeat once per --pred, or once to broadcast)",
    )
    ap.add_argument("--dims", type=int, default=5, help="number of structure dims predicted")
    ap.add_argument("--out", default=None, help="optional JSON with per-dim diagnostics")
    args = ap.parse_args()

    if len(args.train_data) == 1:
        train_paths = args.train_data * len(args.pred)
    elif len(args.train_data) == len(args.pred):
        train_paths = args.train_data
    else:
        raise SystemExit(
            f"got {len(args.train_data)} --train-data for {len(args.pred)} --pred; "
            "pass one or one per prediction file"
        )
    names = dim_names(args.dims)
    payload = {"dims": args.dims, "runs": {}}
    for path, train_path in zip(args.pred, train_paths):
        train = read_rows(train_path)
        if not train:
            raise SystemExit(f"empty training data: {train_path}")
        train_desc = log1p_target([r["descriptor"] for r in train], args.dims)
        train_mean = np.nanmean(train_desc, axis=0)
        rows = read_rows(path)
        if not rows:
            print(f"\n== {path}: empty ==")
            continue
        true = log1p_target([r["descriptor"] for r in rows], args.dims)
        pred = np.asarray([r["s_pred"] for r in rows], dtype=np.float64)[:, : args.dims]
        diags = per_dim_diagnostics(true, pred, names=names, train_mean=train_mean)
        print(f"\n== {path} | n={len(rows)} | train n={len(train)} | dims={args.dims} ==")
        print(format_table(diags))
        summary = diagnostics_payload(diags)["summary"]
        print("\nsummary:", json.dumps(summary, indent=2))
        payload["runs"][str(path)] = {
            "train_data": str(train_path),
            **diagnostics_payload(diags),
        }

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
