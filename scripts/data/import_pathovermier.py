#!/usr/bin/env python
"""PathoVernier: inspect the release, then convert it to our QA format.

The schema of the gated release is not assumed here. ``--inspect`` prints the files,
keys, one sample row and the answer distribution; ``--convert`` takes the field names
explicitly, so a change on their side costs a flag, not a code change.

    python scripts/data/import_pathovermier.py --inspect --src data/raw/pathovermier
    python scripts/data/import_pathovermier.py --convert --src data/raw/pathovermier \
        --out data/real/pathovermier --question-key question --answer-key answer \
        --image-key image --counts-key counts
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd


def load_table(src: Path) -> pd.DataFrame:
    parquet = sorted(src.rglob("*.parquet"))
    if parquet:
        return pd.concat([pd.read_parquet(p) for p in parquet], ignore_index=True)
    jsonl = sorted(src.rglob("*.jsonl"))
    if jsonl:
        rows = []
        for p in jsonl:
            with open(p, "r", encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
        return pd.DataFrame(rows)
    raise SystemExit(f"no parquet or jsonl under {src}")


def inspect(df: pd.DataFrame) -> None:
    print(f"rows {len(df)} | columns {list(df.columns)}")
    row = df.iloc[0].to_dict()
    print("\nfirst row:")
    for key, value in row.items():
        text = repr(value)
        print(f"  {key:24s} {type(value).__name__:10s} {text[:110]}")
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda v: isinstance(v, str)).all():
            uniq = df[col].nunique()
            if uniq <= 12:
                print(f"\n{col} values: {dict(df[col].value_counts())}")


def to_image_bytes(value):
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return value["bytes"]
        if "path" in value and value["path"]:
            return Path(value["path"]).read_bytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str) and Path(value).exists():
        return Path(value).read_bytes()
    return None


def convert(args) -> int:
    df = load_table(Path(args.src))
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, rec in enumerate(df.to_dict(orient="records")):
        question = str(rec.get(args.question_key, "")).strip()
        answer = str(rec.get(args.answer_key, "")).strip()
        if not question or not answer:
            continue
        blob = to_image_bytes(rec.get(args.image_key)) if args.image_key else None
        if blob is None:
            continue
        roi_id = str(rec.get("id", rec.get("roi_id", f"pv_{i:05d}")))
        path = img_dir / f"{roi_id}.png"
        if not path.exists():
            path.write_bytes(blob)
        row = {
            "roi_id": roi_id,
            "patient_id": str(rec.get("patient_id", roi_id)),
            "image": str(path),
            "question": question,
            "answer": answer,
            "task": "pathovermier",
        }
        if args.counts_key and args.counts_key in rec:
            counts = rec[args.counts_key]
            row["counts"] = counts if isinstance(counts, (dict, list)) else str(counts)
        rows.append(row)

    if not rows:
        print("nothing converted: check --image-key and --question-key", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "qa.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {out / 'qa.jsonl'}")
    print("answers:", dict(pd.Series([r['answer'] for r in rows]).value_counts().head(10)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/real/pathovermier")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--question-key", dest="question_key", default="question")
    ap.add_argument("--answer-key", dest="answer_key", default="answer")
    ap.add_argument("--image-key", dest="image_key", default="image")
    ap.add_argument("--counts-key", dest="counts_key", default=None)
    args = ap.parse_args()

    if args.inspect:
        inspect(load_table(Path(args.src)))
        return 0
    if args.convert:
        return convert(args)
    print("choose --inspect or --convert", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
