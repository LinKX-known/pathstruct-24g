#!/usr/bin/env python
"""Down-sample counterfactual rows to an exact label balance within each kind.

`pilot build --min-affected` can leave an imbalance: if a slot cannot find its
target label, it is skipped, and the remaining kind may be mostly "yes". This
script takes min(#yes, #no) per kind independently for the train and val
splits. Run it after `pilot build`, before training.

    python scripts/balance_cf.py \
        --train data/synth_hard/cf_train.jsonl \
        --val   data/synth_hard/cf_val.jsonl \
        --out-train data/synth_hard/cf_train_bal.jsonl \
        --out-val   data/synth_hard/cf_val_bal.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.data import read_jsonl, write_jsonl  # noqa: E402


def balance(rows, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for kind in sorted({r["kind"] for r in rows}):
        yes = [r for r in rows if r["kind"] == kind and r["changed"] == 1]
        no = [r for r in rows if r["kind"] == kind and r["changed"] == 0]
        n = min(len(yes), len(no))
        for pool in (yes, no):
            idx = rng.choice(len(pool), size=n, replace=False)
            out.extend(pool[i] for i in idx)
        print(f"  {kind:9s}: yes {len(yes):3d} no {len(no):3d} -> {n} each")
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for src, dst in ((args.train, args.out_train), (args.val, args.out_val)):
        print(f"balancing {src}")
        rows = balance(read_jsonl(src), args.seed)
        write_jsonl(rows, dst)
        print(f"  -> {dst}: n={len(rows)} yes={sum(r['changed'] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
