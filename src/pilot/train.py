"""A small, readable training loop for a 24 GB card.

Two batches per optimiser step: one from the downstream task (growth pattern)
and one from the counterfactual task (does the architecture change). The second
is optional, which gives the LoRA-only baseline for free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class TrainConfig:
    steps: int = 300
    grad_accum: int = 8
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lam_structure: float = 0.5
    lam_cf: float = 0.5
    seed: int = 0
    log_every: int = 10
    save_dir: str = "out/pilot"
    max_new_tokens: int = 8


def _messages(example: Dict[str, Any], with_answer: bool) -> List[Dict[str, Any]]:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["question"]},
            ],
        }
    ]
    if with_answer:
        msgs.append({"role": "assistant", "content": example["answer"]})
    return msgs


def build_batch(processor, example: Dict[str, Any], device, max_pixels: int) -> Optional[Dict[str, Any]]:
    """Processor outputs + labels with the prompt masked out.

    The image is processed twice (full conversation and prompt-only) to find the
    answer boundary. It costs ~2x preprocessing but avoids fragile string
    matching on the chat template; for a pilot that is the right trade.
    """
    from PIL import Image

    try:
        with Image.open(example["image"]) as im:
            image = im.convert("RGB")
            image.load()
    except Exception:
        return None

    full_text = processor.apply_chat_template(_messages(example, True), tokenize=False)
    prompt_text = processor.apply_chat_template(_messages(example, False), tokenize=False, add_generation_prompt=True)

    full = processor(text=[full_text], images=[image], return_tensors="pt", max_pixels=max_pixels)
    prompt = processor(text=[prompt_text], images=[image], return_tensors="pt", max_pixels=max_pixels)

    labels = full["input_ids"].clone()
    n_prompt = int(prompt["input_ids"].shape[1])
    labels[:, :n_prompt] = -100
    if (labels != -100).sum() == 0:
        return None

    batch = {k: v.to(device) for k, v in full.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    batch["labels"] = labels.to(device)
    return batch


def collate_aux(example: Dict[str, Any]):
    import torch

    desc = torch.tensor([example["descriptor"]], dtype=torch.float32)
    changed = torch.tensor([float(example["changed"])], dtype=torch.float32)
    kind = example.get("kind", "collapse")
    feats = torch.tensor(
        [[0.5, 0.5, 0.2] + [1.0 if kind == k else 0.0 for k in ("collapse", "expand", "translate")]],
        dtype=torch.float32,
    )
    return desc, changed, feats


def smooth_l1(pred, target, beta: float = 1.0):
    import torch

    d = (pred - target).abs()
    return torch.where(d < beta, 0.5 * d**2 / beta, d - 0.5 * beta).mean()


def run(
    model,
    processor,
    growth: Sequence[Dict[str, Any]],
    cf: Sequence[Dict[str, Any]],
    cfg: TrainConfig,
    image_tokens: int = 512,
    use_aux: bool = True,
    device: str = "cuda",
) -> Dict[str, Any]:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    from .heads import build_heads
    from .model import image_token_id

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = cfg.steps
    sched = get_cosine_schedule_with_warmup(
        opt, int(cfg.warmup_ratio * total_steps), total_steps
    )

    d_model = int(getattr(model.config, "hidden_size", 0) or model.config.text_config.hidden_size)
    n_desc = len(cf[0]["descriptor"]) if cf else 0
    heads = build_heads(d_model, max(n_desc, 1), hidden=1024).to(device) if (use_aux and cf) else None
    if heads is not None:
        opt = torch.optim.AdamW(params + list(heads.parameters()), lr=cfg.lr, betas=(0.9, 0.95))

    iid = image_token_id(model)
    max_pixels = int(28 * 28 * image_tokens)
    rng = np.random.default_rng(cfg.seed)
    model.train()

    history: List[Dict[str, float]] = []
    t0 = time.time()
    micro = 0
    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        logs: Dict[str, float] = {}

        for sub in range(cfg.grad_accum):
            use_growth = bool(growth) and (micro % 4 != 3)   # 3 growth : 1 counterfactual
            pool = growth if use_growth else cf
            if not pool:
                pool = cf or growth
            ex = pool[int(rng.integers(0, len(pool)))]
            batch = build_batch(processor, ex, device, max_pixels)
            micro += 1
            if batch is None:
                continue

            need_aux = heads is not None and ex.get("task") == "cf"
            out = model(**batch, output_hidden_states=need_aux)
            loss = out.loss / cfg.grad_accum
            logs["lm"] = logs.get("lm", 0.0) + float(out.loss.detach()) / cfg.grad_accum

            if need_aux:
                hidden = out.hidden_states[-1]
                vis = batch["input_ids"] == iid
                w = vis.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * w).sum(1) / w.sum(1).clamp(min=1.0)
                desc, changed, feats = (t.to(device) for t in collate_aux(ex))
                s_pred = heads["structure"](hidden, attention_mask=vis)
                l_s = smooth_l1(s_pred, torch.log1p(desc.clamp(min=0.0)))
                logit = heads["cf"](pooled, feats)
                l_c = torch.nn.functional.binary_cross_entropy_with_logits(logit, changed)
                loss = loss + (cfg.lam_structure * l_s + cfg.lam_cf * l_c) / cfg.grad_accum
                logs["struct"] = logs.get("struct", 0.0) + float(l_s.detach()) / cfg.grad_accum
                logs["cf"] = logs.get("cf", 0.0) + float(l_c.detach()) / cfg.grad_accum

            loss.backward()

        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        sched.step()

        if (step + 1) % cfg.log_every == 0:
            rec = {"step": step + 1, "lr": sched.get_last_lr()[0], **logs,
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            history.append(rec)
            print(
                f"step {rec['step']:4d} | lm {rec.get('lm', float('nan')):.4f}"
                f" | struct {rec.get('struct', float('nan')):.3f}"
                f" | cf {rec.get('cf', float('nan')):.3f}"
                f" | {rec['elapsed_min']} min | lr {rec['lr']:.2e}",
                flush=True,
            )
            if torch.cuda.is_available():
                print(f"           vram {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB", flush=True)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    if heads is not None:
        torch.save(heads.state_dict(), save_dir / "aux_heads.pt")
    return {"history": history, "save_dir": str(save_dir)}
