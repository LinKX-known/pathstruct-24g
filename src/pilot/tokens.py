"""The response format: generated visual evidence, then observation, then answer.

ASPECT teaches a VLM to emit ``<uni>`` and ``<cell>`` tokens, write measurements,
and answer. This module adds the third kind of evidence -- structure -- so that
topology is *in the generation path* rather than in a side head.

    <|think|> <|uni|>x8 <|cell|>x6 <|struct|>x8 <|/think|>
    <|observe|> {"components": "1", "lumina": "3+", "lumen_size": "large"} <|/observe|>
    <|answer|> One component, three or more lumina, large lumen area. <|/answer|>

The observation is deliberately *binned* rather than raw floats: a token-level
regression head can keep the precision, while the generated text stays short and
learnable. The bins are also what the consistency reward checks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

N_UNI, N_CELL, N_STRUCT = 8, 6, 8

OPEN_THINK, CLOSE_THINK = "<|think|>", "<|/think|>"
OPEN_OBS, CLOSE_OBS = "<|observe|>", "<|/observe|>"
OPEN_ANS, CLOSE_ANS = "<|answer|>", "<|/answer|>"
UNI, CELL, STRUCT = "<|uni|>", "<|cell|>", "<|struct|>"

SPECIAL_TOKENS = [
    OPEN_THINK, CLOSE_THINK, OPEN_OBS, CLOSE_OBS, OPEN_ANS, CLOSE_ANS,
    UNI, CELL, STRUCT,
]

OBSERVE_QUESTION = (
    "Report the structural organisation of this region: how many connected "
    "components, how many lumina, and how large the lumina are."
)


def add_special_tokens(tokenizer) -> int:
    """Register the nine markers as single tokens. Returns how many were new."""
    existing = set(tokenizer.get_vocab())
    new = [t for t in SPECIAL_TOKENS if t not in existing]
    if new:
        tokenizer.add_special_tokens({"additional_special_tokens": new})
    return len(new)


def bin_descriptor(values: Sequence[float], names: Sequence[str]) -> Dict[str, str]:
    """Raw descriptor -> the three bins the model is asked to report."""
    d = {n: float(v) for n, v in zip(names, values)}
    n_comp = d.get("n_comp", float("nan"))
    holes = d.get("hole_count", float("nan"))
    frac = d.get("hole_area_frac", 0.0)

    comp_bin = "1" if n_comp <= 1 else ("2" if n_comp <= 2 else "3+")
    hole_bin = "0" if holes <= 0 else ("1" if holes <= 1 else ("2" if holes <= 2 else "3+"))
    if frac < 0.02:
        size_bin = "none"
    elif frac < 0.10:
        size_bin = "small"
    elif frac < 0.25:
        size_bin = "medium"
    else:
        size_bin = "large"
    return {"components": comp_bin, "lumina": hole_bin, "lumen_size": size_bin}


def observe_text(bins: Dict[str, str]) -> str:
    return json.dumps(bins, separators=(",", ":"), sort_keys=False)


def observe_sentence(bins: Dict[str, str]) -> str:
    """Surface form for the answer, so it is not a verbatim copy of the JSON."""
    comp = {"1": "one component", "2": "two components", "3+": "three or more components"}[bins["components"]]
    lum = {"0": "no lumina", "1": "one lumen", "2": "two lumina", "3+": "three or more lumina"}[bins["lumina"]]
    size = {
        "none": "no measurable lumen area",
        "small": "small lumen area",
        "medium": "moderate lumen area",
        "large": "large lumen area",
    }[bins["lumen_size"]]
    return f"{comp[0].upper()}{comp[1:]}, {lum}, {size}."


def build_response(
    answer: str,
    bins: Optional[Dict[str, str]] = None,
    n_uni: int = N_UNI,
    n_cell: int = N_CELL,
    n_struct: int = N_STRUCT,
) -> str:
    think = OPEN_THINK + UNI * n_uni + CELL * n_cell + STRUCT * n_struct + CLOSE_THINK
    obs = f"{OPEN_OBS}{observe_text(bins)}{CLOSE_OBS}" if bins else ""
    return f"{think}{obs}{OPEN_ANS}{answer}{CLOSE_ANS}"


_ANSWER_RE = re.compile(r"<\|answer\|>(.*?)<\|/answer\|>", re.S)
_OBS_RE = re.compile(r"<\|observe\|>(.*?)<\|/observe\|>", re.S)


def parse_response(text: str) -> Dict[str, Any]:
    """Pull the observation JSON and the answer out of a generated response."""
    out: Dict[str, Any] = {"answer": "", "observe": None}
    m = _ANSWER_RE.search(text or "")
    if m:
        out["answer"] = m.group(1).strip()
    m = _OBS_RE.search(text or "")
    if m:
        try:
            out["observe"] = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            out["observe"] = None
    return out


def token_spans(input_ids: Sequence[int], tokenizer) -> Dict[str, List[int]]:
    """Positions of each marker in a tokenised response, for state supervision."""
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    spans: Dict[str, List[int]] = {t: [] for t in SPECIAL_TOKENS}
    for i, tok in enumerate(input_ids):
        for name, tid in ids.items():
            if tok == tid:
                spans[name].append(i)
    return spans


def evidence_positions(input_ids: Sequence[int], tokenizer, marker: str) -> List[int]:
    """Indices of the evidence slots (the ``n`` positions after a marker)."""
    spans = token_spans(input_ids, tokenizer)
    return spans.get(marker, [])
