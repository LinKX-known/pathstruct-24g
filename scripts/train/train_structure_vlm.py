#!/usr/bin/env python
"""Train the structure-evidence VLM (stage 1/2/3) or evaluate a trained one.

    # stage 1: evidence tokens in the prompt, model writes observation + answer
    python scripts/train/train_structure_vlm.py \
        --data data/real/monuseg/struct_qa.jsonl \
        --stage perceive --preset 4b_bf16 --steps 300 --out out/sft_perceive

    # stage 2/3 + evaluation
    python scripts/train/train_structure_vlm.py \
        --data data/real/monuseg/struct_qa.jsonl \
        --stage reason --preset 4b_bf16 --steps 600 --out out/sft_reason
    python scripts/train/train_structure_vlm.py --eval-only --data ... \
        --adapter out/sft_reason --preset 4b_bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pilot.evidence import read_jsonl  # noqa: E402
from pilot.model import ModelConfig, attach_lora, load_model, prepare_for_training  # noqa: E402
from pilot.sft import (  # noqa: E402
    SFTConfig,
    evaluate_rows,
    resize_for_evidence,
    run,
    unfreeze_embeddings,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stage", default="reason", choices=("perceive", "generate", "reason"))
    ap.add_argument("--preset", default="4b_bf16")
    ap.add_argument("--image-tokens", dest="image_tokens", type=int, default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--grad-accum", dest="grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-struct", dest="lam_struct", type=float, default=0.5)
    ap.add_argument("--lam-cell", dest="lam_cell", type=float, default=0.1)
    ap.add_argument("--out", default="out/sft")
    ap.add_argument("--adapter", default=None, help="adapter to evaluate (with --eval-only)")
    ap.add_argument("--eval-only", dest="eval_only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = ModelConfig.from_preset(args.preset)
    image_tokens = args.image_tokens or cfg.image_tokens

    model, processor = load_model(cfg)
    tokenizer = processor.tokenizer
    resize_for_evidence(model, tokenizer)          # before LoRA: new rows must stay trainable

    rows = read_jsonl(args.data)
    if not rows:
        raise SystemExit("empty data")

    if args.eval_only:
        from peft import PeftModel

        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
        metrics = evaluate_rows(model, processor, tokenizer, rows, image_tokens, args.limit)
        print(json.dumps(metrics, indent=2))
        return 0

    model = attach_lora(model, cfg)
    model = prepare_for_training(model, cfg)
    unfreeze_embeddings(model)
    model.print_trainable_parameters()

    sft = SFTConfig(
        stage=args.stage,
        steps=args.steps,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lam_struct=args.lam_struct,
        lam_cell=args.lam_cell,
        image_tokens=image_tokens,
        save_dir=args.out,
    )
    out = run(model, processor, tokenizer, rows, sft)
    print(f"saved to {out['save_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
