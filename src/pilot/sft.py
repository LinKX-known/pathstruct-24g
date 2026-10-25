"""Stage-1/2/3 supervised fine-tuning for the evidence architecture.

One loop, three stages (see :mod:`pilot.evidence`):

* **perceive** -- the evidence tokens are given in the prompt, the model learns to
  read them and to write the observation and the answer;
* **generate** -- the model emits the token block itself;
* **reason** -- the whole response, tokens included, is generated and scored.

Losses: language CE over the response, plus a structure head on the ``<|struct|>``
states and (when the corpus carries counts) a cell head on the ``<|cell|>``
states. The structure head is what ties the *generated* evidence to the actual
tissue: without it the tokens are free-floating text.

The `<uni>` reconstruction loss needs encoder features that the corpus does not
carry yet, so it is defined but off by default -- deliberately, rather than
pretending to supervise something we do not have.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .evidence import evidence_index, parse_generated, score_evidence, target_for_stage
from .tokens import CELL, CLOSE_THINK, OPEN_THINK, SPECIAL_TOKENS, STRUCT, UNI


@dataclass
class SFTConfig:
    stage: str = "reason"
    steps: int = 300
    grad_accum: int = 8
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lam_struct: float = 0.5
    lam_cell: float = 0.1
    max_length: int = 2048
    image_tokens: int = 512
    save_dir: str = "out/sft"
    log_every: int = 10
    checkpoint_every: int = 50
    seed: int = 0


def resize_for_evidence(model, tokenizer) -> int:
    """Add the nine markers as single tokens and resize the embedding matrix."""
    vocab = tokenizer.get_vocab()
    missing = [t for t in SPECIAL_TOKENS if t not in vocab]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
    # new rows start trainable, but PEFT freezes the base model afterwards
    emb = model.get_input_embeddings()
    if emb is not None:
        emb.weight.requires_grad_(True)
    return len(tokenizer)


def marker_ids(tokenizer) -> Dict[str, int]:
    return {t: int(tokenizer.convert_tokens_to_ids(t)) for t in SPECIAL_TOKENS}


def _messages(row: Dict[str, Any], stage: str) -> tuple[List[Dict[str, Any]], str]:
    prefix, target = target_for_stage(row, stage)
    question = row["question"] + (("\n" + prefix) if prefix else "")
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": row["image"]},
            {"type": "text", "text": question},
        ],
    }
    return [user], target


def build_batch(processor, tokenizer, row: Dict[str, Any], stage: str, device, image_tokens: int):
    """Tokenise one row into model inputs plus evidence slot positions."""
    from PIL import Image

    from .vision import fit_image

    try:
        with Image.open(row["image"]) as im:
            image = fit_image(im.convert("RGB"), image_tokens)
            image.load()
    except Exception:
        return None

    user, target = _messages(row, stage)
    full_messages = user + [{"role": "assistant", "content": target}]
    full_text = processor.apply_chat_template(full_messages, tokenize=False)
    prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)

    full = processor(text=[full_text], images=[image], return_tensors="pt")
    prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")
    n_prompt = int(prompt["input_ids"].shape[1])
    if n_prompt >= full["input_ids"].shape[1]:
        return None

    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    if int((labels != -100).sum()) == 0:
        return None

    keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    batch = {k: v.to(device) for k, v in full.items() if k in keep}
    batch["labels"] = labels.to(device)
    batch["labels"] = batch["labels"][:, : batch["input_ids"].shape[1]]
    batch["positions"] = evidence_index(full["input_ids"][0].tolist(), marker_ids(tokenizer))
    return batch


if True:  # torch-dependent classes are defined lazily to keep import cheap

    def _heads(d_model: int, n_desc: int, n_cell_types: int = 5, hidden: int = 512):
        import torch
        from torch import nn

        class StructHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.slot = nn.Linear(d_model, 128)
                self.net = nn.Sequential(
                    nn.LayerNorm(128 * 8),
                    nn.Linear(128 * 8, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, n_desc),
                )

            def forward(self, states):  # (n_struct, d)
                h = self.slot(states.float()).reshape(1, -1)
                return self.net(h)

        class CellHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, n_cell_types),
                )

            def forward(self, states):  # (n_cell, d)
                return self.net(states.float().mean(0))

        return nn.ModuleDict({"struct": StructHead(), "cell": CellHead()})

    def smooth_l1(pred, target, beta: float = 1.0):
        import torch

        d = (pred - target).abs()
        return torch.where(d < beta, 0.5 * d ** 2 / beta, d - 0.5 * beta).mean()

    def run(
        model,
        processor,
        tokenizer,
        rows: Sequence[Dict[str, Any]],
        cfg: SFTConfig,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        import torch
        from transformers import get_cosine_schedule_with_warmup

        if not rows:
            raise ValueError("no training rows")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        d_model = int(getattr(model.config, "hidden_size", 0) or model.config.text_config.hidden_size)
        n_desc = len(rows[0]["descriptor"])
        heads = _heads(d_model, n_desc).to(device)

        params = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
        sched = get_cosine_schedule_with_warmup(
            opt, int(cfg.warmup_ratio * cfg.steps), cfg.steps
        )

        ids = marker_ids(tokenizer)
        rng = np.random.default_rng(cfg.seed)
        model.train()
        history: List[Dict[str, float]] = []
        t0 = time.time()

        for step in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            logs: Dict[str, float] = {}
            done = 0
            for _ in range(cfg.grad_accum):
                row = rows[int(rng.integers(0, len(rows)))]
                batch = build_batch(processor, tokenizer, row, cfg.stage, device, cfg.image_tokens)
                if batch is None:
                    continue
                positions = batch.pop("positions")
                out = model(**batch, output_hidden_states=True)
                loss = out.loss
                logs["lm"] = logs.get("lm", 0.0) + float(out.loss.detach()) / cfg.grad_accum

                hidden = out.hidden_states[-1][0]
                struct_pos = positions["struct"][:8]
                if len(struct_pos) == 8 and cfg.lam_struct > 0:
                    states = hidden[struct_pos]
                    pred = heads["struct"](states).reshape(-1)
                    target = torch.log1p(
                        torch.tensor(row["descriptor"], device=device, dtype=torch.float32).clamp(min=0)
                    )[:n_desc]
                    l_s = smooth_l1(pred, target)
                    loss = loss + cfg.lam_struct * l_s / cfg.grad_accum
                    logs["struct"] = logs.get("struct", 0.0) + float(l_s.detach()) / cfg.grad_accum

                cell_pos = positions["cell"][:6]
                counts = row.get("cell_counts")
                if counts and len(cell_pos) == 6 and cfg.lam_cell > 0:
                    states = hidden[cell_pos]
                    pred = heads["cell"](states).reshape(-1)
                    target = torch.tensor(counts, device=device, dtype=torch.float32)[: pred.numel()]
                    pad = pred.numel() - target.numel()
                    if pad > 0:
                        target = torch.cat([target, torch.zeros(pad, device=device)])
                    l_c = smooth_l1(pred, torch.log1p(target.clamp(min=0)))
                    loss = loss + cfg.lam_cell * l_c / cfg.grad_accum
                    logs["cell"] = logs.get("cell", 0.0) + float(l_c.detach()) / cfg.grad_accum

                loss.backward()
                done += 1

            if done == 0:
                raise RuntimeError("every batch was skipped: check images and chat template")
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            sched.step()

            if (step + 1) % cfg.log_every == 0:
                rec = {
                    "step": step + 1,
                    "lr": float(sched.get_last_lr()[0]),
                    **logs,
                    "elapsed_min": round((time.time() - t0) / 60, 1),
                }
                history.append(rec)
                print(
                    f"step {rec['step']:4d} | lm {rec.get('lm', float('nan')):.4f}"
                    f" | struct {rec.get('struct', float('nan')):.3f}"
                    f" | cell {rec.get('cell', float('nan')):.3f}"
                    f" | {rec['elapsed_min']} min | lr {rec['lr']:.2e}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    print(f"           vram {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB", flush=True)

            if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
                ckpt = Path(cfg.save_dir) / f"checkpoint-{step + 1}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt)
                torch.save(heads.state_dict(), ckpt / "evidence_heads.pt")

        save = Path(cfg.save_dir)
        save.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save)
        torch.save(heads.state_dict(), save / "evidence_heads.pt")
        tokenizer.save_pretrained(save)
        return {"history": history, "save_dir": str(save)}


def generate_evidence(model, processor, tokenizer, row: Dict[str, Any], image_tokens: int = 512, max_new_tokens: int = 96) -> Dict[str, Any]:
    """Generate the full response for one row and parse it."""
    import torch
    from PIL import Image

    from .vision import fit_image

    user, _ = _messages(row, "reason")
    text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    with Image.open(row["image"]) as im:
        image = fit_image(im.convert("RGB"), image_tokens)
        image.load()
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new, skip_special_tokens=False)
    parsed = parse_generated(raw)
    parsed["raw"] = raw
    return parsed


def evaluate_rows(model, processor, tokenizer, rows: Sequence[Dict[str, Any]], image_tokens: int = 512, limit: Optional[int] = None) -> Dict[str, Any]:
    model.eval()
    subset = list(rows)[:limit] if limit else list(rows)
    scored = []
    for i, row in enumerate(subset, start=1):
        gen = generate_evidence(model, processor, tokenizer, row, image_tokens=image_tokens)
        scored.append(
            {
                "gold_answer": row["answer"],
                "gold_observe": row.get("observe"),
                "answer": gen.get("answer", ""),
                "observe": gen.get("observe"),
                "raw": gen.get("raw", ""),
            }
        )
        if i % 20 == 0:
            print(f"  eval {i}/{len(subset)}", flush=True)
    return score_evidence(scored)

def unfreeze_embeddings(model) -> None:
    """PEFT freezes the base model; the resized embedding rows must stay trainable."""
    emb = model.get_input_embeddings()
    if emb is not None:
        emb.weight.requires_grad_(True)
