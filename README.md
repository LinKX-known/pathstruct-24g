# pathstruct-24g

A **24 GB feasibility pilot** for structure-supervised pathology VLMs. It is a separate,
self-contained repo: it does **not** touch [pathstruct](https://github.com/LinKX-known/pathstruct),
which stays the full method. Here we only ask one question:

> On a single RTX 4090 (24 GB), does training a model to judge **structural changes**
> move the needle, or is the idea dead?

## What the pilot can and cannot tell you

| | |
|---|---|
| ✅ can | whether the data pipeline and the auxiliary task work at all |
| ✅ can | whether a 4B/8B model can learn the counterfactual task (>chance) |
| ✅ can | whether adding the auxiliary loss beats **LoRA-only** on the same data |
| ⚠️ weakly | the *direction* of the effect on downstream accuracy |
| ❌ cannot | the final numbers — a 24 GB run uses a smaller model and fewer image tokens |
| ❌ cannot | any comparison against ASPECT, which trains on a different scale |

Treat a 24 GB result as a **lower bound**: if it works at 512 image tokens, it will very
likely work better at 1024; if it fails, the resolution may simply be too low.

## Three pilots, three gates

| step | what | time on 4090 | pass if |
|---|---|---|---|
| **P0** | zero-shot the counterfactual task | 10 min | accuracy ≈ 0.5–0.65 (near chance → the task is non-trivial) |
| **P1** | train on the counterfactual task only | 1–2 h | balanced accuracy climbs above **0.65** |
| **P2** | growth pattern + aux vs LoRA-only | 2–3 h | downstream accuracy **+2 points** over LoRA-only |

P0 above 0.8 means the task is too easy — shrink the edit radius before training.
P1 stuck at 0.5 means the model cannot learn it from this much data — add data or simplify edits.
P2 flat means the auxiliary signal does not transfer — check `λ` and whether gradients reach the backbone.

## Install

```bash
pip install -e ".[qwen]"        # torch, transformers>=4.57, peft, bitsandbytes
pilot check                     # prints versions + GPU + the three presets
```

`pilot check` must print `"ready": true` and a GPU line before anything else.

## Presets for 24 GB

| preset | model | quant | image tokens | fits? |
|---|---|---|---|---|
| `4b_bf16` | Qwen3-VL-4B | none | 512 | ✅ comfortable (~16 GB) |
| `8b_4bit` | Qwen3-VL-8B | 4-bit | 512 | ✅ (~14 GB) |
| `8b_4bit_768` | Qwen3-VL-8B | 4-bit | 768 | ⚠️ tight, try it |

Start with `4b_bf16`. It has no quantisation bugs, so any failure is your code.

## Run

```bash
# 0. synthetic cohort: proves the pipeline without any real data
pilot synth --out data/synth --per-class 12
pilot build --manifest data/synth/manifest.csv --cells data/synth/cells --out data/synth --per-roi 4

# 1. P0 — zero shot
pilot p0 --data data/synth/cf.jsonl --limit 24 --preset 4b_bf16

# 2. P1 — learn the counterfactual task
pilot train --cf data/synth/cf.jsonl --steps 150 --preset 4b_bf16 --out out/p1
pilot eval  --data data/synth/cf.jsonl --adapter out/p1 --preset 4b_bf16

# 3. P2 — downstream + aux vs LoRA-only (same data, same steps, only the loss differs)
pilot train --growth data/synth/growth.jsonl --cf data/synth/cf.jsonl --steps 300 --preset 4b_bf16 --out out/p2_aux
pilot train --growth data/synth/growth.jsonl --cf data/synth/cf.jsonl --steps 300 --preset 4b_bf16 --out out/p2_lora --no-aux
pilot eval  --data data/synth/growth.jsonl --adapter out/p2_aux  --out pred_aux.json
pilot eval  --data data/synth/growth.jsonl --adapter out/p2_lora --out pred_lora.json --compare pred_aux.json
```

The synthetic cohort has only 12 ROIs — it validates plumbing, not learning. Replace it with
real data as soon as the commands run end to end:

```bash
pilot build --manifest your/manifest.csv --cells your/cells/ --out data/real --per-roi 6
```

`manifest.csv` needs `roi_id, patient_id, image_path, label`; `cells/{roi_id}.csv` needs `x, y, type`.

## What to watch while training

```
step   10 | lm 2.8140 | struct 0.693 | cf 0.693 | 0.8 min | lr 9.8e-05
           vram 17.2 GB
```

- `lm` must fall. On 12 samples it should collapse towards 0 — that is overfitting, and the
  point of the smoke test.
- `struct` and `cf` must fall. `cf` stuck at 0.693 = the auxiliary head is not learning.
- `vram` must stay under ~21 GB. If it creeps up, you have a leak from keeping graphs alive.

## If it OOMs

In order, stop at the first that fixes it:

1. `--preset 4b_bf16` instead of 8B.
2. Lower image tokens: edit `PRESETS` to 384.
3. `--grad-accum 16` and keep batch 1 (it is already 1).
4. `--steps` down; the eval pass is often what OOMs, not training.
5. 8-bit Adam (`bnb.optim.Adam8bit`) instead of AdamW.

Do **not** disable gradient checkpointing — it is what makes 24 GB possible.

## Cost

A 4090 is roughly ¥1.5–2.5/h. The full P0+P1+P2 sequence is about 5–6 GPU-hours,
so **under ¥15**. Do the data work on CPU, free.

## What to do with the result

- **P1 passes and P2 passes**: scale to 8B + 80 GB and 5k ROIs. The idea is alive.
- **P1 passes, P2 flat**: the auxiliary task is learnable but does not transfer. Try a larger
  `--lam-cf`, or check that the auxiliary loss is not being applied to a detached tensor.
- **P1 flat**: the task is too hard for 4B at this resolution. Try `8b_4bit_768` before giving up.
- **P0 already high**: the edits are too easy or too visible. Shrink `radius_frac` in
  `interventions.sample_edit`.

## Honest notes

- The counterfactual labels are exact (computed from the cell point set), but the *images* are
  not edited — the model sees the original ROI plus a box. It must reason about what is inside
  the box, which is the ability we want, but it is not the same as seeing an edited image.
- `build_cf_examples` computes one descriptor per ROI and reuses it for all edits of that ROI.
  That is correct for the structure head (the descriptor is of the *original* tissue) but means
  the auxiliary regression target does not vary within a ROI.
- The synthetic cohort uses ring/solid/multifocal shapes. Good for plumbing, useless for
  pathology claims.

## License

MIT.
