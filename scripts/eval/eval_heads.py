#!/usr/bin/env python
"""Evaluate the *trained* structure and counterfactual heads on held-out rows.

The P1/P2 numbers in the main report evaluate the language model's yes/no
generation. That is a harder and different test than the auxiliary modules
themselves. This script loads ``aux_heads.pt`` from an adapter and runs the two
heads directly on the held-out rows:

* counterfactual head: balanced accuracy / AUC for ``changed``;
* structure head: MSE / R^2 for the descriptor target (in log1p space, the
  space the head was trained in).

    HF_HUB_OFFLINE=1 python scripts/eval_heads.py \
        --adapter out/p1_hard --data data/synth_hard/cf_val_bal.jsonl \
        --preset 4b_bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.data import read_jsonl  # noqa: E402
from pilot.heads import build_heads, grid_from_hidden  # noqa: E402
from pilot.model import ModelConfig, image_token_id, load_model  # noqa: E402
from pilot.train import build_batch, collate_aux  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--preset", default="4b_bf16")
    ap.add_argument("--image-tokens", dest="image_tokens", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out", default=None, help="optional JSON array of per-row results")
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

    logits, labels, kinds, s_pred, desc_true = [], [], [], [], []
    records = []
    for i, ex in enumerate(rows, start=1):
        batch = build_batch(processor, ex, device, image_tokens)
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
            desc, changed, feats = (t.to(device) for t in collate_aux(ex))
            logit = float(heads["cf"](grid, feats).reshape(-1)[0])
            s = heads["structure"](grid).reshape(-1).cpu()
            logits.append(logit)
            s_pred.append(s)
            records.append(
                {
                    "ex_id": ex.get("ex_id", ""),
                    "patient_id": ex.get("patient_id", ""),
                    "kind": ex.get("kind", ""),
                    "changed": int(ex["changed"]),
                    "logit": logit,
                    "pred": int(logit > 0),
                    "correct": int((logit > 0) == bool(ex["changed"])),
                    "s_pred": s.tolist(),
                    "descriptor": ex["descriptor"],
                }
            )
        labels.append(int(ex["changed"]))
        kinds.append(ex["kind"])
        desc_true.append(np.asarray(ex["descriptor"], dtype=np.float64))
        if i % 50 == 0:
            print(f"  eval {i}/{len(rows)}", flush=True)

    y = np.asarray(labels)
    logit = np.asarray(logits)
    pred = (logit > 0).astype(int)
    print(f"\nrows {len(y)} | adapter {args.adapter}")
    print(
        "cf head: "
        f"acc {np.mean(pred == y):.3f} | balanced {balanced_accuracy_score(y, pred):.3f} | "
        f"AUC {roc_auc_score(y, logit) if len(set(y)) > 1 else float('nan'):.3f} | "
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
    target = np.log1p(np.clip(true[:, :s_dim], 0, None))
    mse = float(np.mean((pred_desc - target) ** 2))
    mae = float(np.mean(np.abs(pred_desc - target)))
    r2 = [r2_score(target[:, j], pred_desc[:, j]) for j in range(target.shape[1])]
    print(
        f"structure head: dims {s_dim} | log1p MSE {mse:.4f} | MAE {mae:.4f} | "
        f"R2 mean {np.mean(r2):.3f} | R2 topology mean {np.mean(r2[:5]):.3f}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
