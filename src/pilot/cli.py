"""Pilot CLI for a 24 GB card.

    pilot check                                  # what is installed, what fits
    pilot synth  --out data/synth                # synthetic cohort, no real data
    pilot build  --manifest m.csv --cells cells/ --out data/
    pilot p0     --data data/cf.jsonl --limit 30 # zero-shot: is the task non-trivial?
    pilot train  --growth data/growth.jsonl --cf data/cf.jsonl --steps 200
    pilot eval   --data data/growth.jsonl --adapter out/pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .data import (
    GROWTH_QUESTION,
    build_cf_examples,
    build_growth_examples,
    make_synthetic,
    patient_splits,
    read_jsonl,
    read_manifest,
    write_jsonl,
)
from .eval import evaluate, gate, paired_bootstrap, score
from .model import PRESETS, ModelConfig


def cmd_check(args) -> int:
    info = {"presets": PRESETS}
    for mod in ("torch", "transformers", "peft", "bitsandbytes", "qwen_vl_utils"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except ImportError:
            info[mod] = None
    try:
        import skimage

        info["scikit-image"] = getattr(skimage, "__version__", "installed")
    except ImportError:
        info["scikit-image"] = None
    try:
        from .model import gpu_report

        info["gpu"] = gpu_report()
    except Exception as exc:  # noqa: BLE001
        info["gpu"] = f"unavailable ({exc})"
    info["ready"] = bool(info.get("torch") and info.get("transformers"))
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if not info.get("scikit-image"):
        print("[warn] scikit-image missing: `pilot segment --backend classical` will fail")
    return 0 if info["ready"] else 1


def cmd_synth(args) -> int:
    df = make_synthetic(
        args.out,
        per_class=args.per_class,
        n_cells=args.n_cells,
        size=args.size,
        seed=args.seed,
        dot_radius=args.dot_radius,
    )
    print(f"synthetic cohort: {len(df)} ROIs -> {args.out}")
    return 0


def cmd_build(args) -> int:
    manifest = read_manifest(args.manifest)
    labels = sorted(
        {
            str(r.get("label", "")).strip()
            for r in manifest.to_dict(orient="records")
            if str(r.get("label", "")).strip()
        }
    )
    question = args.growth_question
    if not question:
        if set(labels) <= {"ring", "solid", "multifocal"}:
            question = None  # build_growth_examples' default
        else:
            question = (
                "What tissue type is this? Choose one of: "
                + ", ".join(labels)
                + ". Answer with one word."
            )
    growth = build_growth_examples(manifest, question=question or GROWTH_QUESTION)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.presentation == "before_after" and args.no_annotate:
        print("--no-annotate is incompatible with --presentation before_after")
        return 2
    annotate_dir = None if args.no_annotate else out / "annotated"
    cf = build_cf_examples(
        manifest,
        args.cells,
        per_roi=args.per_roi,
        seed=args.seed,
        annotate=annotate_dir is not None,
        annotate_dir=annotate_dir,
        balance_by_kind=not args.no_balance_by_kind,
        min_affected=args.min_affected,
        min_cells=args.min_cells,
        presentation=args.presentation,
        render_dot_radius=args.render_dot_radius,
    )
    write_jsonl(growth, out / "growth.jsonl")
    write_jsonl(cf, out / "cf.jsonl")
    rate = sum(r["changed"] for r in cf) / max(len(cf), 1)
    print(f"growth: {len(growth)} | counterfactual: {len(cf)} | changed rate {rate:.3f}")
    if cf:
        kinds = sorted({r["kind"] for r in cf})
        for k in kinds:
            subset = [r for r in cf if r["kind"] == k]
            print(
                f"  {k:9s}: n={len(subset):3d} changed={np.mean([r['changed'] for r in subset]):.2f}"
            )
        print(f"  annotated images: {annotate_dir if annotate_dir is not None else 'off'}")

    if args.val_frac > 0:
        g_train, g_val, c_train, c_val = patient_splits(
            growth, cf, val_frac=args.val_frac, seed=args.seed
        )
        write_jsonl(g_train, out / "growth_train.jsonl")
        write_jsonl(g_val, out / "growth_val.jsonl")
        write_jsonl(c_train, out / "cf_train.jsonl")
        write_jsonl(c_val, out / "cf_val.jsonl")
        val_rate = sum(r["changed"] for r in c_val) / max(len(c_val), 1)
        print(
            f"split by patient: growth {len(g_train)}/{len(g_val)} train/val | "
            f"cf {len(c_train)}/{len(c_val)} train/val | val changed rate {val_rate:.3f}"
        )
        if not g_val or not c_val:
            print("[warn] holdout is empty; lower --val-frac or add patients")
    return 0


def _load(preset: str, lora: bool = True):
    from .model import attach_lora, load_model, prepare_for_training

    cfg = ModelConfig.from_preset(preset)
    model, processor = load_model(cfg)
    if lora:
        model = attach_lora(model, cfg)
        model = prepare_for_training(model, cfg)
        model.print_trainable_parameters()
    return model, processor, cfg


def cmd_p0(args) -> int:
    data = read_jsonl(args.data)
    model, processor, cfg = _load(args.preset, lora=False)
    choices = ["yes", "no"] if data and data[0].get("task") == "cf" else sorted({r["answer"] for r in data})
    rows = evaluate(model, processor, data, choices, limit=args.limit, image_tokens=cfg.image_tokens)
    s = score(rows)
    print(json.dumps({k: v for k, v in s.items() if k != "per_class"}, indent=2))
    print("per class:", s.get("per_class"))
    if s.get("per_kind"):
        print("per kind:", json.dumps(s["per_kind"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("verdict:", gate(s, threshold=args.gate))
    return 0


def cmd_train(args) -> int:
    from .train import TrainConfig, run

    growth = read_jsonl(args.growth) if args.growth and Path(args.growth).exists() else []
    cf = read_jsonl(args.cf) if args.cf and Path(args.cf).exists() else []
    if not growth and not cf:
        print("no data: run `pilot build` first" )
        return 2
    if not cf and not args.no_aux:
        print("[warn] --cf is empty: training without the counterfactual head")

    model, processor, cfg = _load(args.preset)
    tcfg = TrainConfig(
        steps=args.steps,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lam_structure=args.lam_structure,
        lam_cf=args.lam_cf,
        save_dir=args.out,
        log_every=args.log_every,
        seed=args.seed,
        structure_dims=args.structure_dims,
    )
    run(
        model,
        processor,
        growth,
        cf,
        tcfg,
        image_tokens=cfg.image_tokens,
        use_aux=not args.no_aux,
    )
    print(f"saved adapter to {args.out}")
    return 0


def cmd_eval(args) -> int:
    model, processor, cfg = _load(args.preset, lora=False)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    data = read_jsonl(args.data)
    choices = ["yes", "no"] if data and data[0].get("task") == "cf" else sorted({r["answer"] for r in data})
    rows = evaluate(model, processor, data, choices, limit=args.limit, image_tokens=cfg.image_tokens)
    s = score(rows)
    print(json.dumps({k: v for k, v in s.items() if k != "per_class"}, indent=2))
    if s.get("per_kind"):
        print("per kind:", json.dumps(s["per_kind"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.compare:
        base = read_jsonl(args.compare)
        key = lambda r: r.get("ex_id") or (r["roi_id"] + "|" + r["gold"])
        b = {key(r): r["correct"] for r in base}
        pairs = [(r, b[key(r)]) for r in rows if key(r) in b]
        if pairs:
            print(
                "paired:",
                paired_bootstrap(
                    [p[0]["correct"] for p in pairs],
                    [p[1] for p in pairs],
                    groups=[p[0].get("patient_id") or p[0]["roi_id"] for p in pairs],
                ),
            )
    print("verdict:", gate(s, threshold=args.gate))
    return 0


def cmd_segment(args) -> int:
    from .segment import segment_dir

    df = segment_dir(args.images, args.out, backend=args.backend, pattern=args.pattern)
    if len(df):
        print(
            f"segmented {len(df)} images | "
            f"cells: median {df['n_cells'].median():.0f}, min {df['n_cells'].min()}, max {df['n_cells'].max()}"
        )
        bad = df[df["error"] != ""]
        if len(bad):
            print(f"{len(bad)} failed, e.g.:")
            print(bad.head(3).to_string(index=False))
    print(f"cells written to {args.out}")
    return 0


def cmd_cells(args) -> int:
    from .cells import cells_from_cellvit_dir

    df = cells_from_cellvit_dir(args.json_dir, args.out, suffix=args.suffix)
    if len(df):
        print(
            f"converted {len(df)} CellViT exports | cells median {df['n_cells'].median():.0f}, "
            f"min {df['n_cells'].min()}, max {df['n_cells'].max()}"
        )
        bad = df[df["error"] != ""]
        if len(bad):
            print(f"{len(bad)} failed, e.g.:")
            print(bad.head(3).to_string(index=False))
    print(f"cells written to {args.out}")
    return 0


def cmd_doctor(args) -> int:
    from .diagnostics import doctor

    df = read_manifest(args.manifest)
    out = doctor(df, args.cells, min_cells=args.min_cells)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if out["problems"]:
        print("\nfix these before training:")
        for p in out["problems"]:
            print(f"  - {p}")
    return 0 if out["ok"] else 1


def cmd_viz(args) -> int:
    from .cells import load_cells_dir
    from .diagnostics import render_roi

    roi_id = args.roi_id or Path(args.image).stem
    item = load_cells_dir(args.cells, roi_id)
    if item is None:
        print(f"no cells for {roi_id} in {args.cells}")
        return 2
    pts, _ = item
    info = render_roi(args.image, pts, args.out)
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pilot", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check")
    c.set_defaults(func=cmd_check)

    cl = sub.add_parser("cells")
    cl.add_argument("--json-dir", dest="json_dir", required=True)
    cl.add_argument("--out", required=True)
    cl.add_argument("--suffix", default="_cells")
    cl.set_defaults(func=cmd_cells)

    dc = sub.add_parser("doctor")
    dc.add_argument("--manifest", required=True)
    dc.add_argument("--cells", required=True)
    dc.add_argument("--min-cells", dest="min_cells", type=int, default=60)
    dc.set_defaults(func=cmd_doctor)

    vz = sub.add_parser("viz")
    vz.add_argument("--image", required=True)
    vz.add_argument("--cells", required=True)
    vz.add_argument("--out", required=True)
    vz.add_argument("--roi-id", dest="roi_id", default=None)
    vz.set_defaults(func=cmd_viz)

    sg = sub.add_parser("segment")
    sg.add_argument("--images", required=True)
    sg.add_argument("--out", required=True)
    sg.add_argument("--backend", default="classical", choices=("classical", "mask"))
    sg.add_argument("--pattern", default="*.png")
    sg.set_defaults(func=cmd_segment)

    s = sub.add_parser("synth")
    s.add_argument("--out", default="data/synth")
    s.add_argument("--per-class", dest="per_class", type=int, default=12)
    s.add_argument("--n-cells", dest="n_cells", type=int, default=220)
    s.add_argument("--size", type=int, default=384)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dot-radius", dest="dot_radius", type=float, default=3.0)
    s.set_defaults(func=cmd_synth)

    b = sub.add_parser("build")
    b.add_argument("--manifest", required=True)
    b.add_argument("--cells", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--per-roi", dest="per_roi", type=int, default=6)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--val-frac", dest="val_frac", type=float, default=0.3)
    b.add_argument("--no-annotate", dest="no_annotate", action="store_true")
    b.add_argument("--no-balance-by-kind", dest="no_balance_by_kind", action="store_true")
    b.add_argument(
        "--min-affected",
        dest="min_affected",
        type=int,
        default=0,
        help="reject edits whose box contains fewer than N cells (try 5-8)",
    )
    b.add_argument(
        "--min-cells",
        dest="min_cells",
        type=int,
        default=60,
        help="skip ROIs with fewer cells (60 synthetic; 16-20 for PanNuke patches)",
    )
    b.add_argument(
        "--growth-question",
        dest="growth_question",
        default=None,
        help="override the downstream question; inferred for tissue labels",
    )
    b.add_argument(
        "--presentation",
        choices=("box", "before_after"),
        default="box",
        help="box: full ROI + box/arrow; before_after: rendered cell diagram pair",
    )
    b.add_argument(
        "--render-dot-radius",
        dest="render_dot_radius",
        type=float,
        default=8.0,
        help="dot radius for the before/after renderer",
    )
    b.set_defaults(func=cmd_build)

    for name, fn in (("p0", cmd_p0), ("train", cmd_train), ("eval", cmd_eval)):
        sp = sub.add_parser(name)
        sp.add_argument("--preset", default="4b_bf16", choices=sorted(PRESETS))
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--gate", type=float, default=0.65)
        sp.set_defaults(func=fn)

    p0 = sub.choices["p0"]
    p0.add_argument("--data", required=True)
    p0.add_argument("--out", default=None)

    tr = sub.choices["train"]
    tr.add_argument("--growth", default=None)
    tr.add_argument("--cf", default=None)
    tr.add_argument("--steps", type=int, default=200)
    tr.add_argument("--grad-accum", dest="grad_accum", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--lam-structure", dest="lam_structure", type=float, default=0.5)
    tr.add_argument("--lam-cf", dest="lam_cf", type=float, default=0.5)
    tr.add_argument("--no-aux", dest="no_aux", action="store_true")
    tr.add_argument("--out", default="out/pilot")
    tr.add_argument("--log-every", dest="log_every", type=int, default=10)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument(
        "--structure-dims",
        dest="structure_dims",
        type=int,
        default=0,
        help="0=full descriptor; 5=topology block only (drop 3x3 regional)",
    )

    ev = sub.choices["eval"]
    ev.add_argument("--data", required=True)
    ev.add_argument("--adapter", default=None)
    ev.add_argument("--out", default=None)
    ev.add_argument("--compare", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


