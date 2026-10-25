"""Evidence architecture, model-agnostic parts.

Everything here works without a GPU: where the evidence slots are, how to mask
the prompt, how to score a generated response. The torch loop lives in
:mod:`pilot.sft`; keeping the two apart is what makes this testable on CI.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .tokens import (
    CELL,
    CLOSE_ANS,
    CLOSE_OBS,
    CLOSE_THINK,
    OPEN_ANS,
    OPEN_OBS,
    OPEN_THINK,
    STRUCT,
    UNI,
    observe_sentence,
    observe_text,
)

STAGES = ("perceive", "generate", "reason")

_ANSWER_RE = re.compile(r"<\|answer\|>(.*?)<\|/answer\|>", re.S)
_OBS_RE = re.compile(r"<\|observe\|>(.*?)<\|/observe\|>", re.S)


def think_block(n_uni: int = 8, n_cell: int = 6, n_struct: int = 8) -> str:
    return f"{OPEN_THINK}{UNI * n_uni}{CELL * n_cell}{STRUCT * n_struct}{CLOSE_THINK}"


def target_for_stage(row: Dict[str, Any], stage: str) -> tuple[str, str]:
    """Split a row into (prompt_prefix, target) for the three ASPECT stages.

    perceive  the evidence tokens are given in the prompt; only observation and
              answer are generated
    generate  the model produces the token block itself
    reason    the whole response is generated
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    think = think_block()
    obs = f"{OPEN_OBS}{observe_text(row['observe'])}{CLOSE_OBS}" if row.get("observe") else ""
    ans = f"{OPEN_ANS}{row['answer']}{CLOSE_ANS}"

    if stage == "perceive":
        return think, obs + ans
    if stage == "generate":
        return "", think + obs + ans
    return "", think + obs + ans


def mask_prompt(input_ids: Sequence[int], n_prompt: int, ignore_index: int = -100) -> List[int]:
    """Labels with the prompt positions set to ``ignore_index``."""
    labels = list(input_ids)
    for i in range(min(n_prompt, len(labels))):
        labels[i] = ignore_index
    return labels


def find_positions(input_ids: Sequence[int], token_id: int) -> List[int]:
    return [i for i, t in enumerate(input_ids) if t == token_id]


def evidence_index(input_ids: Sequence[int], ids: Dict[str, int]) -> Dict[str, List[int]]:
    """Slot positions for each evidence kind, in one tokenised sequence."""
    return {
        "uni": find_positions(input_ids, ids[UNI]),
        "cell": find_positions(input_ids, ids[CELL]),
        "struct": find_positions(input_ids, ids[STRUCT]),
    }


def parse_generated(text: str) -> Dict[str, Any]:
    """Observation dict and answer text from a generated response."""
    out: Dict[str, Any] = {"answer": "", "observe": None}
    m = _ANSWER_RE.search(text or "")
    if m:
        out["answer"] = m.group(1).strip()
    m = _OBS_RE.search(text or "")
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
            if isinstance(parsed, dict):
                out["observe"] = parsed
        except json.JSONDecodeError:
            out["observe"] = None
    return out


def _norm(value: Any) -> str:
    return str(value).strip().lower().rstrip(".")


def score_evidence(rows: Sequence[Dict[str, Any]], choices: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Answer accuracy, observation accuracy, coherence and RAWR-struct.

    ``rows`` need ``gold_answer``, ``gold_observe`` (dict or None) and the parsed
    ``answer`` / ``observe`` from the generation.
    """
    if not rows:
        return {"n": 0}
    answer_ok = []
    field_ok = []
    obs_exact = []
    coherent = []
    for r in rows:
        gold = _norm(r["gold_answer"])
        pred = _norm(r.get("answer", ""))
        ok = pred == gold if choices is None else (pred in {_norm(c) for c in choices} and pred == gold)
        answer_ok.append(bool(ok))

        gold_obs = r.get("gold_observe") or {}
        pred_obs = r.get("observe") or {}
        fields = list(gold_obs.keys())
        if fields and isinstance(pred_obs, dict):
            hits = [str(pred_obs.get(f, "")) == str(gold_obs[f]) for f in fields]
            field_ok.extend(hits)
            obs_exact.append(all(hits))
            # coherence: does the answer agree with the observation the model itself reported?
            if r.get("coherence_rule") is not None and pred_obs:
                coherent.append(bool(r["coherence_rule"](pred_obs, pred)))
        else:
            obs_exact.append(False)

    n = len(rows)
    out = {
        "n": n,
        "answer_accuracy": float(sum(answer_ok) / n),
        "observation_field_accuracy": float(sum(field_ok) / len(field_ok)) if field_ok else float("nan"),
        "observation_exact_match": float(sum(obs_exact) / n),
        "coherent_accuracy": float(sum(coherent) / len(coherent)) if coherent else float("nan"),
    }
    correct_with_obs = [i for i, ok in enumerate(answer_ok) if ok and obs_exact[i] is not None]
    if correct_with_obs:
        wrong = [0 if obs_exact[i] else 1 for i in correct_with_obs]
        out["rawr_struct"] = float(sum(wrong) / len(wrong))
    else:
        out["rawr_struct"] = float("nan")
    return out


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
