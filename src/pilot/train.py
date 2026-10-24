"""A small, readable training loop for a 24 GB card.

Two batches per optimiser step: one from the downstream task (growth pattern)
and one from the counterfactual task (does the architecture change). The second
is optional, which gives the LoRA-only baseline for free.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .hurdle import positive_class_weights, zero_inflated_dims


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
    # 0 = train on the full descriptor; 5 = topology block only (n_comp,
    # hole_count, hole_area_frac, h0_max, h0_sum). The 3x3 regional block was
    # consistently unlearned on real data and diluted the topology loss.
    structure_dims: int = 0
    # Hurdle decomposition for zero-inflated dims (hole_count, hole_area_frac):
    # a presence BCE plus a positive-only smooth-L1, instead of one regression
    # that is dominated by the zero spike.
    structure_hurdle: bool = False
    hurdle_zero_threshold: float = 0.5
    # P2: supervise the structure head on downstream (growth/tissue) examples
    # too, not only on counterfactual examples. This is what makes topology a
    # shared representation for the downstream task instead of a side task.
    structure_on_growth: bool = True
    lam_structure_growth: float = 0.5
    log_every: int = 10
    # Save an intermediate adapter every N steps (0 = only the final one).
    # The real-data runs are short and the balance between under- and
    # overfitting is the main open question, so keep the trajectory to
    # evaluate/early-stop afterwards.
    checkpoint_every: int = 50
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


def build_batch(processor, example: Dict[str, Any], device, image_tokens: int = 512) -> Optional[Dict[str, Any]]:
    """Processor outputs + labels with the prompt masked out.

    The image is resized before it reaches the processor (see ``vision.py``)
    instead of passing ``max_pixels``, which not every processor version
    accepts. The image is processed twice (full conversation and prompt-only) to
    find the answer boundary: ~2x preprocessing, but no fragile string matching
    on the chat template.
    """
    import torch
    from PIL import Image

    from .vision import fit_image

    try:
        with Image.open(example["image"]) as im:
            image = fit_image(im.convert("RGB"), image_tokens)
            image.load()
    except Exception:
        return None

    full_text = processor.apply_chat_template(_messages(example, True), tokenize=False)
    prompt_text = processor.apply_chat_template(_messages(example, False), tokenize=False, add_generation_prompt=True)

    full = processor(text=[full_text], images=[image], return_tensors="pt")
    prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")

    labels = full["input_ids"].clone()
    n_prompt = int(prompt["input_ids"].shape[1])
    if n_prompt >= labels.shape[1]:
        return None
    # The two processor calls must produce the same prefix. If the tokenizer
    # merged the answer with the assistant header (or the image expansion
    # differs), masking the first n_prompt tokens would mask part of the
    # answer, so refuse the example instead of training on a wrong boundary.
    if not torch.equal(full["input_ids"][0, :n_prompt], prompt["input_ids"][0]):
        return None
    labels[:, :n_prompt] = -100
    if (labels != -100).sum() == 0:
        return None

    batch = {k: v.to(device) for k, v in full.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    batch["labels"] = labels.to(device)
    return batch


def collate_aux(example: Dict[str, Any]):
    """Tensors for the two auxiliary heads.

    The mask features carry the *actual* edit centre and radius, so the
    counterfactual head knows where the intervention happened. Downstream
    (growth/tissue) rows have a descriptor but no ``changed`` label; they only
    feed the structure head, so ``changed`` defaults to 0 and is ignored.
    """
    import torch

    if "descriptor" not in example:
        raise KeyError("auxiliary supervision needs a `descriptor` field")
    desc = torch.tensor([example["descriptor"]], dtype=torch.float32)
    changed = torch.tensor([float(example.get("changed", 0.0))], dtype=torch.float32)
    center = example.get("center") or [0.5, 0.5]
    radius = float(example.get("radius") or 0.2)
    kind = example.get("kind", "collapse")
    vector = example.get("vector") or [0.0, 0.0]
    feats = torch.tensor(
        [
            [
                float(center[0]),
                float(center[1]),
                radius,
                float(vector[0]),
                float(vector[1]),
            ]
            + [1.0 if kind == k else 0.0 for k in ("collapse", "expand", "translate")]
        ],
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

    from .heads import build_heads, grid_or_global_mean
    from .model import image_token_id

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # heads first: the optimiser and the scheduler must be built once, over the
    # final parameter set. Creating the scheduler before the heads and then
    # replacing the optimiser leaves the schedule attached to an unused object.
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    base_cfg = base_model.config
    d_model = int(
        getattr(base_cfg, "hidden_size", 0)
        or getattr(getattr(base_cfg, "text_config", None), "hidden_size", 0)
        or 0
    )
    if d_model <= 0:
        raise ValueError("cannot find the text hidden size on the model config")
    vision_cfg = getattr(base_cfg, "vision_config", None)
    merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2) or 2)
    n_desc = len(cf[0]["descriptor"]) if cf else 0
    has_growth_desc = any(bool(g.get("descriptor")) for g in growth)
    if n_desc == 0 and has_growth_desc:
        n_desc = len(next(g["descriptor"] for g in growth if g.get("descriptor")))
    s_dims = int(cfg.structure_dims or n_desc)
    if s_dims > n_desc:
        raise ValueError(
            f"structure_dims={s_dims} exceeds descriptor length {n_desc}"
        )
    # Hurdle dims are chosen on the training descriptors only; the eval side
    # reads the same list back from structure_hurdle.json.
    desc_pool = [
        r["descriptor"]
        for r in list(cf) + list(growth)
        if r.get("descriptor") and len(r["descriptor"]) >= s_dims
    ]
    hurdle_dims: List[int] = []
    hurdle_zero_frac = [0.0] * s_dims
    hurdle_pos_weight: List[float] = []
    if cfg.structure_hurdle:
        hurdle_dims, hurdle_zero_frac = zero_inflated_dims(
            desc_pool, s_dims, cfg.hurdle_zero_threshold
        )
        hurdle_pos_weight = positive_class_weights(desc_pool, hurdle_dims)
    heads = (
        build_heads(
            d_model,
            max(s_dims, 1),
            hidden=1024,
            presence_dims=len(hurdle_dims),
        ).to(device)
        if (use_aux and (cf or has_growth_desc))
        else None
    )
    if cfg.structure_hurdle:
        hurdle_cfg = {
            "dims": s_dims,
            "hurdle_dims": hurdle_dims,
            "zero_fraction": hurdle_zero_frac,
            "pos_weight": hurdle_pos_weight,
            "zero_threshold": cfg.hurdle_zero_threshold,
        }
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.save_dir) / "structure_hurdle.json").write_text(
            json.dumps(hurdle_cfg, indent=2), encoding="utf-8"
        )

    params = [p for p in model.parameters() if p.requires_grad]
    if heads is not None:
        params = params + list(heads.parameters())
    total_steps = cfg.steps
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(
        opt, int(cfg.warmup_ratio * total_steps), total_steps
    )

    iid = image_token_id(base_model)
    rng = np.random.default_rng(cfg.seed)
    model.train()
    warned_grid = False
    warned_grid_failed = False
    grid_fallbacks = 0
    grid_failures = 0

    history: List[Dict[str, float]] = []
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    micro = 0
    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        logs: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        used = 0
        did_backward = False

        for sub in range(cfg.grad_accum):
            use_growth = bool(growth) and (micro % 4 != 3)   # 3 growth : 1 counterfactual
            pool = growth if use_growth else cf
            if not pool:
                pool = cf or growth
            ex = pool[int(rng.integers(0, len(pool)))]
            batch = build_batch(processor, ex, device, image_tokens)
            micro += 1
            if batch is None:
                continue

            used += 1
            # The before/after image is a two-panel composite: the edit centre
            # in the left panel does not address the pooled token grid of the
            # whole composite, so do not train the geometry-conditioned heads on
            # it. The language loss is the experiment there.
            is_cf = ex.get("task") == "cf" and ex.get("presentation", "box") == "box"
            has_descriptor = bool(ex.get("descriptor"))
            # P2 supervision: train the structure head on downstream examples
            # that carry a descriptor (computed from the same patch's cell
            # table), not only on the counterfactual examples. Without this the
            # topology loss is a side task on a different data distribution.
            need_structure = heads is not None and (
                is_cf or (cfg.structure_on_growth and has_descriptor)
            )
            need_cf = heads is not None and is_cf
            need_aux = need_structure or need_cf
            out = model(**batch, output_hidden_states=need_aux)
            loss = out.loss / cfg.grad_accum
            logs["lm"] = logs.get("lm", 0.0) + float(out.loss.detach())

            if need_aux:
                hidden = out.hidden_states[-1]
                vis = batch["input_ids"] == iid
                grid, grid_status = grid_or_global_mean(
                    hidden, vis, batch.get("image_grid_thw"), merge_size=merge_size
                )
                if grid_status == "fallback":
                    grid_fallbacks += 1
                    if not warned_grid:
                        print(
                            "[warn] image tokens do not form the expected grid; "
                            "auxiliary heads fall back to global pooling",
                            flush=True,
                        )
                        warned_grid = True
                elif grid_status == "failed":
                    grid_failures += 1
                    if not warned_grid_failed:
                        print(
                            "[warn] no image tokens found; auxiliary losses skipped "
                            "for these batches",
                            flush=True,
                        )
                        warned_grid_failed = True

                if grid is not None:
                    desc, changed, feats = (t.to(device) for t in collate_aux(ex))
                    if need_structure and desc.shape[1] >= s_dims:
                        s_out = heads["structure"](grid)
                        reg_pred = s_out[:, :s_dims]
                        # Topology-only mode drops the 3x3 regional block, which
                        # had negative R^2 on both real cohorts.
                        raw = desc[:, :s_dims].clamp(min=0.0)
                        target = torch.log1p(raw)
                        if hurdle_dims:
                            pres_pred = s_out[:, s_dims:]
                            non_h = [i for i in range(s_dims) if i not in hurdle_dims]
                            parts = []
                            if non_h:
                                parts.append(smooth_l1(reg_pred[:, non_h], target[:, non_h]))
                            h = torch.as_tensor(
                                hurdle_dims, device=desc.device, dtype=torch.long
                            )
                            pos = raw.index_select(1, h) > 0
                            pw = torch.as_tensor(
                                hurdle_pos_weight,
                                device=desc.device,
                                dtype=pres_pred.dtype,
                            )
                            l_pres = torch.nn.functional.binary_cross_entropy_with_logits(
                                pres_pred, pos.to(pres_pred.dtype), pos_weight=pw
                            )
                            if bool(pos.any()):
                                l_pos = smooth_l1(
                                    reg_pred.index_select(1, h)[pos],
                                    target.index_select(1, h)[pos],
                                )
                            else:
                                l_pos = pres_pred.sum() * 0.0
                            l_s = sum(parts) + l_pres + l_pos
                        else:
                            l_s = smooth_l1(reg_pred, target)
                        lam_s = cfg.lam_structure if is_cf else cfg.lam_structure_growth
                        loss = loss + lam_s * l_s / cfg.grad_accum
                        logs["struct"] = logs.get("struct", 0.0) + float(l_s.detach())
                        counts["struct"] = counts.get("struct", 0) + 1
                    if need_cf:
                        logit = heads["cf"](grid, feats)
                        l_c = torch.nn.functional.binary_cross_entropy_with_logits(logit, changed)
                        loss = loss + cfg.lam_cf * l_c / cfg.grad_accum
                        logs["cf"] = logs.get("cf", 0.0) + float(l_c.detach())
                        counts["cf"] = counts.get("cf", 0) + 1

            loss.backward()
            did_backward = True

        if not did_backward:
            # Every example in this step failed preprocessing. Do not reuse the
            # previous step's graph (and do not advance the schedule).
            print(f"step {step + 1:4d} | no trainable example, skipping", flush=True)
            continue

        n_used = max(used, 1)
        # ``lm`` is present for every micro-batch; the auxiliary losses are
        # present only for batches where the head was actually applied. Divide
        # each by its own count instead of diluting it by all batches.
        logs = {
            k: v / (n_used if k == "lm" else max(counts.get(k, 0), 1))
            for k, v in logs.items()
        }
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

        if (
            cfg.checkpoint_every
            and (step + 1) % cfg.checkpoint_every == 0
            and (step + 1) < total_steps
        ):
            ckpt = save_dir / f"checkpoint-{step + 1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            if heads is not None:
                torch.save(heads.state_dict(), ckpt / "aux_heads.pt")
            if cfg.structure_hurdle:
                (ckpt / "structure_hurdle.json").write_text(
                    (save_dir / "structure_hurdle.json").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            print(f"           checkpoint {ckpt}", flush=True)

    model.save_pretrained(save_dir)
    if heads is not None:
        torch.save(heads.state_dict(), save_dir / "aux_heads.pt")
    if grid_fallbacks or grid_failures:
        print(
            f"[warn] auxiliary geometry: {grid_fallbacks} fallback batches, "
            f"{grid_failures} batches without image tokens; "
            "see the per-run diagnostics before trusting localisation",
            flush=True,
        )
    return {
        "history": history,
        "save_dir": str(save_dir),
        "grid_fallbacks": grid_fallbacks,
        "grid_failures": grid_failures,
    }
