#!/usr/bin/env python
"""Evaluate the *trained* structure and counterfactual heads on held-out rows.

The P1/P2 numbers in the main report evaluate the language model's yes/no
generation. This is a harder and different test than the auxiliary modules
themselves. This script loads ``aux_heads.pt`` from an adapter and runs the two
heads directly on the held-out rows:

* counterfactual head: balanced accuracy / AUC for ``changed``;
* structure head: per-dimension R^2, R^2 of a constant train-mean predictor,
  skill, log-space MAE, Spearman rho, calibration, and presence metrics for
  zero-inflated descriptors (see ``pilot.structure_eval``).

Pass ``--train-data`` so the train-mean baseline is computed on the training
split. Without it the ``R2mean`` / ``skill`` columns are ``nan`` and the
diagnosis is incomplete.

    HF_HUB_OFFLINE=1 python scripts/eval/eval_heads.py \
        --adapter out/p1_hard --data data/synth_hard/cf_val_bal.jsonl \
        --train-data data/synth_hard/cf_train_bal.jsonl --preset 4b_bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.data import read_jsonl  # noqa: E402
from pilot.heads import build_heads, grid_or_global_mean  # noqa: E402
from pilot.model import ModelConfig, image_token_id, load_model  # noqa: E402
from pilot.structure_eval import (  # noqa: E402
    format_table,
    log1p_target,
    per_dim_diagnostics,
    summary,
)
from pilot.structures import TOPO_KEYS  # noqa: E402
from pilot.train import build_batch, collate_aux  # noqa: E402


def dim_names(d: int):
    return TOPO_KEYS[:d] + [f"dim{i}" for i in range(len(TOPO_KEYS), d)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--train-data",
        default=None,
        help="training JSONL used for this adapter (enables the train-mean baseline)",
    )
    ap.add_argument("--preset", default="4b_bf16")
    ap.add_argument("--image-tokens", dest="image_tokens", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out", default=None, help="optional JSON array of per-row results")
    ap.add_argument("--metrics-out", dest="metrics_out", default=None, help="optional JSON of diagnostics")
    args = ap.parse_args()

    cfg = ModelConfig.from_preset(args.preset)
    image_tokens = args.image_tokens or cfg.image_tokens
    rows = read_jsonl(args.data)
    if not rows:
        raise SystemExit("empty data")

    from peft import PeftModel

    model, processor = load_model(cfg)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = next(model.parameters()).device
    iid = image_token_id(base)
    merge = int(getattr(base.config.vision_config, "spatial_merge_size", 2) or 2)
    d_model = int(base.config.text_config.hidden_size)
    n_desc = len(rows[0]["descriptor"])
    state = torch.load(
        Path(args.adapter) / "aux_heads.pt", map_location="cpu", weights_only=True
    )
    # Infer the trained structure-head output width from the checkpoint: the
    # head may have been trained on the topology block only (5 dims).
    weight_keys = [
        k
        for k, v in state.items()
        if k.startswith("structure.net.") and k.endswith(".weight") and v.dim() == 2
    ]
    s_dim = int(state[weight_keys[-1]].shape[0]) if weight_keys else n_desc
    heads = build_heads(d_model, s_dim, hidden=args.hidden).to(device)
    heads.load_state_dict(state)
    heads.eval()

    train_mean = None
    if args.train_data:
        train_rows = read_jsonl(args.train_data)
        if not train_rows:
            raise SystemExit(f"empty training data: {args.train_data}")
        train_target = log1p_target([r["descriptor"] for r in train_rows], s_dim)
        train_mean = np.nanmean(train_target, axis=0)

    logits, labels, kinds, s_pred, desc_true = [], [], [], [], []
    records = []
    skipped_batch = skipped_grid = grid_fallbacks = 0
    for i, ex in enumerate(rows, start=1):
        batch = build_batch(processor, ex, device, image_tokens)
        if batch is None:
            skipped_batch += 1
            continue
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True)
            grid, grid_status = grid_or_global_mean(
                out.hidden_states[-1],
                batch["input_ids"] == iid,
                batch.get("image_grid_thw"),
                merge_size=merge,
            )
            if grid is None:
                skipped_grid += 1
                continue
            if grid_status == "fallback":
                grid_fallbacks += 1
            desc, changed, feats = (t.to(device) for t in collate_aux(ex))
            logit = float(heads["cf"](grid, feats).reshape(-1)[0])
            s = heads["structure"](grid).reshape(-1).cpu()
            logits.append(logit)
            s_pred.append(s)
            records.append(
                {
                    "ex_id": ex.get("ex_id", ""),
                    "roi_id": ex.get("roi_id", ""),
                    "patient_id": ex.get("patient_id", ""),
                    "wsi_id": ex.get("wsi_id", ""),
                    "kind": ex.get("kind", ""),
                    "changed": int(ex["changed"]),
                    "logit": logit,
                    "pred": int(logit > 0),
                    "correct": int((logit > 0) == bool(ex["changed"])),
                    "grid_fallback": grid_status == "fallback",
                    "s_pred": s.tolist(),
                    "descriptor": ex["descriptor"],
                }
            )
        labels.append(int(ex["changed"]))
        kinds.append(ex["kind"])
        desc_true.append(np.asarray(ex["descriptor"], dtype=np.float64))
        if i % 50 == 0:
            print(f"  eval {i}/{len(rows)}", flush=True)

    n_ok = len(records)
    print(
        f"\nrows requested {len(rows)} | evaluated {n_ok} | "
        f"skipped batch={skipped_batch} no_image_tokens={skipped_grid} | "
        f"grid fallback={grid_fallbacks} | adapter {args.adapter}"
    )
    if n_ok == 0:
        raise SystemExit("no rows evaluated; check the image paths and processor")
    if skipped_batch or skipped_grid or grid_fallbacks:
        print(
            "[warn] some rows were not evaluated on a proper token grid; "
            "the structure/cf metrics below are not directly comparable to a "
            "run with zero skips/fallbacks",
            flush=True,
        )

    y = np.asarray(labels)
    logit_arr = np.asarray(logits)
    pred = (logit_arr > 0).astype(int)
    print(
        "cf head: "
        f"acc {np.mean(pred == y):.3f} | balanced {balanced_accuracy_score(y, pred):.3f} | "
        f"AUC {roc_auc_score(y, logit_arr) if len(set(y)) > 1 else float('nan'):.3f} | "
        f"pred yes {np.mean(pred):.3f} | gold yes {np.mean(y):.3f}"
    )
    for kind in sorted(set(kinds)):
        m = np.asarray([k == kind for k in kinds])
        print(
            f"  {kind:9s}: n={int(m.sum()):3d} "
            f"balanced {balanced_accuracy_score(y[m], pred[m]):.3f} "
            f"pred yes {np.mean(pred[m]):.3f} gold yes {np.mean(y[m]):.3f}"
        )

    true = np.stack(desc_true)
    pred_desc = np.stack([p.numpy() for p in s_pred])
    target = log1p_target(true, s_dim)
    keep = min(pred_desc.shape[1], target.shape[1])
    target, pred_desc = target[:, :keep], pred_desc[:, :keep]
    if train_mean is not None:
        train_mean = np.asarray(train_mean, dtype=np.float64)[:keep]
        if len(train_mean) < keep:
            train_mean = np.pad(train_mean, (0, keep - len(train_mean)), constant_values=np.nan)
    diags = per_dim_diagnostics(
        target,
        pred_desc,
        names=dim_names(keep),
        train_mean=train_mean,
    )
    print("\nstructure head (log1p space):")
    print(format_table(diags))
    if train_mean is None:
        print(
            "[note] no --train-data: the train-mean baseline (R2mean/skill) "
            "was not computed; pass the training JSONL for a complete diagnosis"
        )
    metric_summary = summary(diags)
    print("structure summary:", json.dumps(metric_summary, indent=2))

    if args.metrics_out:
        payload = {
            "adapter": args.adapter,
            "data": args.data,
            "train_data": args.train_data,
            "rows_requested": len(rows),
            "rows_evaluated": n_ok,
            "skipped_batch": skipped_batch,
            "skipped_no_image_tokens": skipped_grid,
            "grid_fallbacks": grid_fallbacks,
            "per_dim": diags,
            "summary": metric_summary,
        }
        Path(args.metrics_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"diagnostics written to {args.metrics_out}")
    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
