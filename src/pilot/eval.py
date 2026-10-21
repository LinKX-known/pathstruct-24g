"""Zero-shot and post-training evaluation, with paired comparison."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

_PUNCT = re.compile(r"[^a-z0-9 ]")
_LEAD = re.compile(
    r"^(the answer is|the answer|answer is|answer|it is|i think( that)?|option)\s*[:\-]?\s*"
)


def normalise(text: str) -> str:
    return _PUNCT.sub("", (text or "").strip().lower())


def parse(text: str, choices: Sequence[str]) -> Optional[str]:
    """Strict-ish parsing of a one-word answer.

    Binary answers (yes/no) must match exactly, because "no idea" would
    otherwise be scored as a confident "no". Word answers tolerate a trailing
    clause ("acinar pattern" -> "acinar").
    """
    t = _LEAD.sub("", normalise(text)).split(",")[0].strip()
    if not t:
        return None
    if t in choices:
        return t
    toks = t.split()
    if toks and toks[0] in choices and max(len(c) for c in choices) > 3:
        return toks[0]
    return None


def generate_answer(model, processor, example: Dict[str, Any], image_tokens: int = 512, max_new_tokens: int = 8) -> str:
    import torch
    from PIL import Image

    from .vision import fit_image

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["question"]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(example["image"]) as im:
        image = fit_image(im.convert("RGB"), image_tokens)
        image.load()
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0][inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(new, skip_special_tokens=True).strip()


def evaluate(
    model,
    processor,
    examples: Sequence[Dict[str, Any]],
    choices: Sequence[str],
    limit: Optional[int] = None,
    image_tokens: int = 512,
    progress_every: int = 20,
) -> List[Dict[str, Any]]:
    model.eval()
    rows = []
    subset = list(examples)[:limit] if limit else list(examples)
    for i, ex in enumerate(subset, start=1):
        raw = ""
        try:
            raw = generate_answer(model, processor, ex, image_tokens=image_tokens)
        except Exception as exc:  # noqa: BLE001
            raw = f"ERROR {type(exc).__name__}"
        pred = parse(raw, choices)
        rows.append(
            {
                "task": ex.get("task", ""),
                "roi_id": ex.get("roi_id", ""),
                "patient_id": ex.get("patient_id", ""),
                "gold": normalise(ex["answer"]),
                "pred": pred or "",
                "raw": raw[:200],
                "correct": int(pred == normalise(ex["answer"])),
            }
        )
        if progress_every and i % progress_every == 0:
            print(f"  eval {i}/{len(subset)}", flush=True)
    return rows


def score(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"n": 0}
    y = np.asarray([r["correct"] for r in rows], dtype=float)
    golds = np.asarray([r["gold"] for r in rows])
    per_class = {}
    for g in sorted(set(golds)):
        mask = golds == g
        per_class[g] = float(y[mask].mean())
    return {
        "n": int(len(y)),
        "accuracy": float(y.mean()),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "per_class": per_class,
        "parse_failures": int(sum(1 for r in rows if not r["pred"])),
    }


def paired_bootstrap(
    a_correct: Sequence[int],
    b_correct: Sequence[int],
    n_boot: int = 1000,
    seed: int = 0,
    groups: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Mean difference (a - b) with a 95% percentile interval.

    Pass ``groups`` (patient or slide ids) to resample *clusters* instead of
    items. ROIs from the same patient are correlated, and ignoring that is the
    fastest way to manufacture a significant result. (Same idea as the grouped
    bootstrap in CPathOGen's honesty statistics.)
    """
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    if groups is None:
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        deltas = a[idx].mean(1) - b[idx].mean(1)
    else:
        groups = np.asarray(groups)
        if len(groups) != len(a):
            raise ValueError("groups must align with the correctness arrays")
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        for i in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([members[g] for g in pick])
            deltas[i] = a[idx].mean() - b[idx].mean()
    return {
        "delta": float(a.mean() - b.mean()),
        "lo": float(np.quantile(deltas, 0.025)),
        "hi": float(np.quantile(deltas, 0.975)),
        "n": int(len(a)),
        "n_groups": int(len(np.unique(groups))) if groups is not None else int(len(a)),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """FDR adjustment for the per-class / per-edit-kind comparisons."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted.tolist()
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted.tolist()


def gate(score_dict: Dict[str, float], threshold: float = 0.65) -> str:
    """Plain-language verdict for the pilot decision."""
    acc = score_dict.get("balanced_accuracy", float("nan"))
    if acc != acc:
        return "no data"
    if acc >= threshold:
        return f"PASS (balanced acc {acc:.3f} >= {threshold})"
    return f"NOT YET (balanced acc {acc:.3f} < {threshold})"
