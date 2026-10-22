#!/usr/bin/env python
"""Linear probe: does auxiliary structure training change the backbone?

P2's downstream growth task saturates on the synthetic cohort, so accuracy
cannot separate the two arms. This probe freezes each adapter, extracts the
image-token grid from the last hidden state, and fits a fresh ridge regression
from those features to the vector of structure descriptors. The split is the
same patient-disjoint train/val split used everywhere else.

    HF_HUB_OFFLINE=1 python scripts/probe_structure.py \
        --aux out/p2_aux --lora out/p2_lora \
        --train data/synth/cf_train.jsonl --val data/synth/cf_val.jsonl

Metrics are normalised MSE / R^2 per descriptor block; lower MSE is better.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.data import read_jsonl  # noqa: E402
from pilot.heads import grid_from_hidden  # noqa: E402
from pilot.model import ModelConfig, image_token_id, load_model  # noqa: E402
from pilot.train import build_batch  # noqa: E402

TOPO = 5


def extract_many(adapter: str | None, row_sets, preset: str = "4b_bf16", image_tokens: int = 512):
    """Load the adapter once, then extract features for each row set."""
    from peft import PeftModel

    cfg = ModelConfig.from_preset(preset)
    model, processor = load_model(cfg)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    iid = image_token_id(base)
    merge = int(getattr(base.config.vision_config, "spatial_merge_size", 2) or 2)

    outputs = []
    for rows in row_sets:
        feats_mean, feats_grid, desc, patient, ex_id = [], [], [], [], []
        for i, ex in enumerate(rows, start=1):
            batch = build_batch(processor, ex, "cuda", image_tokens)
            if batch is None:
                continue
            with torch.no_grad():
                out = model(**batch, output_hidden_states=True)
            grid = grid_from_hidden(
                out.hidden_states[-1],
                batch["input_ids"] == iid,
                batch.get("image_grid_thw"),
                merge_size=merge,
            )
            if grid is None:
                continue
            feats_mean.append(grid.mean((0, 1)).float().cpu().numpy())
            feats_grid.append(
                F.adaptive_avg_pool2d(grid.float().permute(2, 0, 1).unsqueeze(0), (3, 3))
                .flatten()
                .cpu()
                .numpy()
            )
            desc.append(np.asarray(ex["descriptor"], dtype=np.float64))
            patient.append(ex["patient_id"])
            ex_id.append(ex["ex_id"])
            if i % 50 == 0:
                print(f"  extracted {i}/{len(rows)}", flush=True)
        if not feats_mean:
            raise RuntimeError("no features extracted; check the image paths")
        outputs.append(
            {
                "mean": np.stack(feats_mean),
                "grid": np.stack(feats_grid),
                "desc": np.stack(desc),
                "patient": np.asarray(patient),
                "ex_id": np.asarray(ex_id),
            }
        )
    return outputs


def fit_probe(train: Dict, val: Dict, key: str, alpha: float = 100.0) -> Dict[str, float]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    sx = StandardScaler().fit(train[key])
    sy = StandardScaler().fit(train["desc"])
    x_tr = sx.transform(train[key])
    x_va = sx.transform(val[key])
    y_tr = sy.transform(train["desc"])
    y_va = sy.transform(val["desc"])
    reg = Ridge(alpha=alpha).fit(x_tr, y_tr)
    pred = sy.inverse_transform(reg.predict(x_va))
    true = val["desc"]
    mse_all = float(np.mean((pred - true) ** 2))
    mae_all = float(np.mean(np.abs(pred - true)))
    mse_topo = float(np.mean((pred[:, :TOPO] - true[:, :TOPO]) ** 2))
    mse_reg = float(np.mean((pred[:, TOPO:] - true[:, TOPO:]) ** 2))
    r2 = [r2_score(true[:, j], pred[:, j]) for j in range(true.shape[1])]
    return {
        "n": int(len(true)),
        "mse_all": mse_all,
        "mae_all": mae_all,
        "mse_topology": mse_topo,
        "mse_regional": mse_reg,
        "r2_mean": float(np.mean(r2)),
        "r2_topology_mean": float(np.mean(r2[:TOPO])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux", default="out/p2_aux")
    ap.add_argument("--lora", default="out/p2_lora")
    ap.add_argument("--train", default="data/synth/cf_train.jsonl")
    ap.add_argument("--val", default="data/synth/cf_val.jsonl")
    ap.add_argument("--alpha", type=float, default=100.0)
    args = ap.parse_args()

    train = read_jsonl(args.train)
    val = read_jsonl(args.val)
    print("extracting aux features ...", flush=True)
    tr, va = extract_many(args.aux, [train, val])
    print("extracting baseline features ...", flush=True)
    tr_b, va_b = extract_many(args.lora, [train, val])

    for key in ("mean", "grid"):
        print(f"\n== features: {key} ==")
        for name, a, b in (("aux", tr, va), ("lora", tr_b, va_b)):
            print(name, fit_probe(a, b, key, alpha=args.alpha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
