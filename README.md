# pathstruct-24g

A **24 GB feasibility pilot** for structure-supervised pathology VLMs. It is a separate,
self-contained repo: it does **not** touch [pathstruct](https://github.com/LinKX-known/pathstruct),
which stays the full method. Here we only ask one question:

> On a single RTX 4090 (24 GB), does training a model to judge **structural changes**
> move the needle, or is the idea dead?

## Repository layout

| path | what |
|---|---|
| src/pilot/ | the pipeline: structures, interventions, dataset builders, heads, train loop, evaluation |
| scripts/data/ | import MoNuSeg / PanNuke exports into manifest.csv + cells/ |
| scripts/build/ | balance counterfactual sets by edit kind and label |
| scripts/eval/ | evaluate trained heads, bootstrap confidence intervals, probe the structure signal |
| scripts/controls/ | blank-box control generation |
| docs/RESULTS.md | the feasibility report: numbers, controls, what failed and why |
| docs/FEASIBILITY.md | full technical report, including the bugs that were found |
| 	ests/ | 26 CPU tests: python -m unittest discover -s tests -t . |

Everything below is the current state of the pilot; the older easibility/pilot-fixes
branch has been merged into main.

**Feasibility run (2026-10-21):** the easy LM task is learnable (0.523 → 0.674
on held-out patients; text-only and box-only controls both 0.500), **but the
0.674 is largely a shortcut**: 14% of validation boxes are empty and trivially
"no". On a strict non-empty, kind/label-balanced rebuild, the **LM** is 0.500
at 384 px and 0.534 at 1024 px (CI [0.481, 0.582]). The **trained auxiliary
heads** do better: the structure head reaches topology R² 0.315/0.411
(blank controls negative), and the counterfactual head reaches **0.640** on
strict 384 (blank 0.509) and 0.686 on strict 1024 (blank AUC 0.728, so the
latter is mostly box geometry). The module-level idea is partially validated;
the LM readout and P2 transfer are not. P2 is inconclusive because the
synthetic downstream task saturates.

**Real H&E follow-up:** MoNuSeg (37 train / 14 test patients, expert nucleus
masks): the trained counterfactual head averages **0.611** over 3 seeds
(0.583/0.600/0.650), blank-box 0.500, box-geometry 0.467; the **expand** edit is
stable at 0.722, translate ~0.56. The structure head does **not** generalise on
this 37-image cohort (topology R² negative). PanNuke (405 train patches, 5 cell
types, 4 completely held-out tissues): cf head **0.588** over 2 seeds
(0.595/0.582), blank 0.530, box-geometry 0.577, but the head is underfit
(1.3 epochs) and the structure head again has negative topology R². Full
numbers, controls and next steps: [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md).

**2026-10-22 extension:** PanNuke 600 steps + topology-only reached **0.600**
(blank 0.547; real−blank +5.3 pt, CI [0.5, 10.9]) — only +1.2 pt over 300
steps. MoNuSeg 150-step topology-only averaged **0.594** (blank 0.489; expand
0.750). Structure-head topology R² remained negative on both cohorts;
dropping the 3×3 regional block was not the fix. Details:
[`docs/RESULTS.md`](docs/RESULTS.md) §3.4.

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

The feasibility run is a concrete example of that failure mode: the easy
counterfactual task is learnable, but the strict non-empty version is 0.500 at
384 px (144 visual tokens) and only 0.534 at 1024 px (1024 tokens,
ground-truth cells). See the resolution check in
[`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) before interpreting a P1 result.

## Three pilots, three gates

| step | what | time on 4090 | pass if |
|---|---|---|---|
| **P0** | zero-shot the counterfactual task on held-out patients | 10 min | balanced accuracy ≈ 0.5–0.65 (near chance → the task is non-trivial) |
| **P1** | train on the counterfactual task only, evaluate held-out patients | 1–2 h | balanced accuracy climbs above **0.65** |
| **P2** | growth + aux vs the same data mix without aux, held-out patients | 2–3 h | downstream accuracy **+2 points** over the baseline |

P0 above 0.8 means the task is too easy — shrink the edit radius before training.
P1 stuck at 0.5 means the model cannot learn it from this much data — add data or simplify edits.
The feasibility run found one more failure mode: a P1 pass can come from
**empty boxes** (always "no"). Rebuild with `--min-affected 5` and report the
non-empty subset; the 384 px strict control was at chance. See
[`docs/FEASIBILITY.md`](docs/FEASIBILITY.md).
P2 flat means the auxiliary signal does not transfer — but first check that the
baseline is not already at 1.000 (a ceiling makes P2 uninformative), then check
`λ` and whether gradients reach the backbone. The feasibility run hit exactly
that ceiling on synthetic data.

`pilot build` writes `*_train.jsonl` / `*_val.jsonl`, split by **patient**, and
the script evaluates only on `*_val.jsonl`. Evaluating on the training ROIs
would make P1 a memorisation check and P2 a ceiling comparison.

The `--no-aux` arm still sees the counterfactual examples through the language
loss; it differs from the aux arm only in the auxiliary heads/losses. That is
the right ablation for "does the auxiliary loss help", but it is not a
growth-only LoRA baseline. Say which baseline you mean in the paper.

## Install

```bash
pip install -e ".[qwen]"        # torch, transformers>=4.57,<5, peft, bitsandbytes
pilot check                     # prints versions + GPU + the three presets
```

`pilot check` must print `"ready": true` and a GPU line before anything else.
`scikit-image` is a base dependency now; without it `pilot segment` fails.
Transformers 5.x requires torch>=2.6, so the extra pins `<5` — the pilot is
validated on 4.57.6 with torch 2.5.1.

## Presets for 24 GB

| preset | model | quant | image tokens | fits? |
|---|---|---|---|---|
| `4b_bf16` | Qwen3-VL-4B | none | 512 | ✅ comfortable (~16 GB) |
| `8b_4bit` | Qwen3-VL-8B | 4-bit | 512 | ✅ (~14 GB) |
| `8b_4bit_768` | Qwen3-VL-8B | 4-bit | 768 | ⚠️ tight, try it |
| `4b_bf16_1024` | Qwen3-VL-4B | none | 1024 | ✅ (~11 GB measured) |

Start with `4b_bf16`. It has no quantisation bugs, so any failure is your code.
`4b_bf16_1024` is the high-resolution check for small objects: at 384 px a
32 px visual token covers several synthetic cells, so the local structure the
counterfactual task asks about is not resolvable.

## Segmentation: pixels to cells

The topology descriptor only needs coordinates, so the first step is turning an ROI
image into a cell table:

```
ROI png ──► segmentation ──► cells/{roi_id}.csv (x, y, type) ──► topology
```

`pilot synth` writes the *ground-truth* tables to `data/synth/cells`. Always
segment into a different directory (`data/synth/cells_seg` in the script).
Writing into `cells/` made failures invisible: the old table stayed behind and
`doctor` saw a complete cohort. `segment_dir` now deletes the target file before
segmenting, so a failure leaves a hole that `doctor` reports.

Two routes, and you should use both at some point:

**1. Bundled, weight-free (start here)**

```bash
pilot segment --images data/synth/images --out data/synth/cells_seg --backend classical
```

Haematoxylin channel → Otsu → distance transform → watershed. No weights, no GPU.
On a 72-ROI synthetic cohort it produces a cell table for all 72 ROIs (the
24/class cohort is what the feasibility run used; the default script still
makes 12/class). Ring → 1 lumen, solid → 0 lumens, multifocal → 3 connected
components in the unit test. It merges touching nuclei (median 136 of 220
cells) and assigns one type; on real H&E it will under-segment further. It is
here so the pipeline runs anywhere.

Otsu needed a guard: it returns a histogram bin *centre*, and when the background
is a single near-zero value the centre can fall just below it. `>` then marks the
whole image as foreground. Four of the first 36 synthetic ROIs failed that way;
the guard splits halfway between the background and the upper tail instead.

**2. CellViT++ / HoVer-Net / StarDist (use for anything you report)**

Run their inference first (CellViT++ writes `{name}_cells.json` with a `centroid`
and a `class` per nucleus), then convert:

```bash
pilot cells --json-dir cellvit_out --out data/real/cells
```

That is a real bridge, not a stub: the parser tolerates the surrounding schema and
only requires the centroid and class, so a HoVer-Net export shaped the same way also
works. If your segmenter writes instance masks instead, use `pilot segment --backend mask`.

Whatever you pick, state it in the paper — the cell table is now an input to your
result, not a given.

## One command

```bash
bash scripts/run_pilot.sh            # synthetic: segment -> doctor -> build -> P0 -> P1 -> P2
DATA=real bash scripts/run_pilot.sh  # uses data/real/{manifest.csv,cells/}
```

It is idempotent (skips steps whose outputs exist; `FORCE=1` to redo), and it stops if
`doctor` finds a problem. `PER_CLASS=24 VAL_FRAC=0.3` gives the 72-ROI cohort used
in the feasibility run.

## Benchmarking the segmenter on PanNuke / MoNuSeg

Both datasets ship instance masks, so before training any VLM you can measure how
much of the topology target is real and how much is segmentation noise:

`ash
pilot segbench --images pannuke/images/fold1 --masks pannuke/masks/fold1 --out pannuke_segbench.csv
`

It reports detection F1 and centroid error, plus the two numbers that actually
matter here:

| column | why it matters |
|---|---|
| descriptor_delta_median | how far the arrangement descriptor moves when ground-truth nuclei are replaced by predicted ones |
| hole_agreement | how often the lumen count agrees; below ~0.8 the target is noisy |

Read the verdict line: a weak segmenter invalidates the supervision, not just the
reported segmentation score.

Two honest caveats about using these sets for *training* a segmenter:

- **Do not retrain a segmenter from scratch.** CellViT++ is already PanNuke-trained;
  a home-made U-Net would be worse and cost GPU hours you could spend on the VLM.
- **PanNuke patches are 256x256.** That is fine for detection and cell typing, but a
  lumen needs a larger field of view — most 256x256 patches contain no ring at all,
  so PanNuke cannot validate the hole feature. MoNuSeg's 1000x1000 fields are the
  right scale for that, and its multi-organ content doubles as a domain check.

## Run

```bash
# 0. synthetic cohort: proves the pipeline without any real data
pilot synth --out data/synth --per-class 24
pilot segment --images data/synth/images --out data/synth/cells_seg --backend classical
pilot build --manifest data/synth/manifest.csv --cells data/synth/cells_seg \
  --out data/synth --per-roi 6 --val-frac 0.3
# optional, recommended for a structural benchmark: add --min-affected 5
# (empty boxes are always "no" and are a free shortcut; see docs/FEASIBILITY.md)

# 1. P0 — zero shot on held-out patients
pilot p0 --data data/synth/cf_val.jsonl --preset 4b_bf16

# 2. P1 — learn the counterfactual task, evaluate held-out patients
pilot train --cf data/synth/cf_train.jsonl --steps 150 --preset 4b_bf16 --out out/p1
pilot eval  --data data/synth/cf_val.jsonl --adapter out/p1 --preset 4b_bf16

# 3. P2 — aux vs the same data mix without aux (same steps, only the loss differs)
pilot train --growth data/synth/growth_train.jsonl --cf data/synth/cf_train.jsonl \
  --steps 300 --preset 4b_bf16 --out out/p2_aux
pilot train --growth data/synth/growth_train.jsonl --cf data/synth/cf_train.jsonl \
  --steps 300 --preset 4b_bf16 --out out/p2_lora --no-aux
pilot eval  --data data/synth/growth_val.jsonl --adapter out/p2_aux  --out pred_aux.json
pilot eval  --data data/synth/growth_val.jsonl --adapter out/p2_lora --out pred_lora.json --compare pred_aux.json
```

`pilot build` also writes the unsplit `growth.jsonl` / `cf.jsonl`; use those only
for debugging. The split files are the ones the gates refer to.

`--no-annotate` is a debug/ablation mode: it stops drawing the box and puts the
region coordinates back into text. A text-only model can then solve part of the
task (0.69 balanced accuracy before the fix), so do not use it for a headline
counterfactual result.

## Topology diagnostics and P2

The old "five-dimension mean R²" headline is dominated by the zero-inflated
`hole_area_frac`; it is not a valid summary of the structure head. Use the
per-dimension diagnostics (train-mean baseline, skill, log-MAE, Spearman,
presence AUC) instead:

```bash
python scripts/eval/diagnose_structure.py \
  --pred out/heads_topo_real_s0.json \
  --train-data data/real/monuseg_train/cf_train_bal.jsonl --dims 5
```

Two controls/next steps are implemented:

```bash
# same image, box moved; label recomputed from the cell table
python scripts/controls/make_wrongbox_control.py \
  --data data/real/monuseg_test/cf_bal.jsonl \
  --cells data/real/monuseg/cells \
  --out data/real/monuseg_test/wrongbox \
  --jsonl data/real/monuseg_test/cf_bal_wrongbox.jsonl
python scripts/eval/analyze_wrongbox.py \
  --real out/heads_topo_real_s0.json \
  --wrongbox out/heads_wrongbox_topo_real_s0.json

# DHMC slide-level growth-pattern data -> patches
python scripts/data/import_dhmc.py --metadata MetaData.csv --wsi-dir /data/DHMC \
  --out data/real/dhmc --patch-size 512 --patches-per-slide 32

# aggregate patch predictions to a slide answer
python scripts/eval/aggregate_slide_predictions.py \
  --pred out/dhmc_aux_eval.json --out out/dhmc_aux_slide.json
```

The full reasoning, literature anchors (TopoSlide, PMIL, TG-HEM, GeoTop,
GraPHFormer) and the P2 design are in
[`docs/P2_TOPOLOGY_PLAN.md`](docs/P2_TOPOLOGY_PLAN.md).

The default synthetic cohort has 12 ROIs/class. That validates plumbing, not
learning; use `--per-class 24` or more for a learnability check, and replace it
with real data as soon as the commands run end to end:

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

A 4090 is roughly ¥1.5–2.5/h. The full P0+P1+P2 sequence is about 5–6 GPU-hours
on real data, so **under ¥15**. Measured on the 72-ROI synthetic cohort:
P0 <1 min, P1 4.2 min, each P2 arm ~8 min, plus ~20 min of CPU data work.
Do the data work on CPU, free.

## What to do with the result

- **P1 passes and P2 passes**: scale to 8B + 80 GB and 5k ROIs. The idea is alive.
- **P1 passes, P2 flat**: the auxiliary task is learnable but does not transfer. Try a larger
  `--lam-cf`, or check that the auxiliary loss is not being applied to a detached tensor.
- **P1 flat**: the task is too hard for 4B at this resolution. Try `8b_4bit_768` before giving up.
- **P0 already high**: the edits are too easy or too visible. Shrink `radius_frac` in
  `interventions.sample_edit`.

## Look before you train

Two commands exist because the most common failure is invisible in the loss:

```bash
pilot doctor --manifest data/real/manifest.csv --cells data/real/cells
pilot viz --image data/real/images/roi_001.png --cells data/real/cells --out viz/roi_001.png
```

`doctor` checks that every image exists, every ROI has cells, the descriptors are
finite and the classes are not 80% one label. `viz` writes a three-panel figure:
cells over the image, the rasterised sheet with lumina in red, and the connected
components. If the lumen is in the wrong place, the segmenter is wrong — no amount
of training will fix it.

## Three details borrowed from TopoSlide's preprocessing

Reading their `preprocessing/cluster_topology/*` was worth it; three things changed
this repo:

1. **Pad before computing topology.** They pad the map and set the outside to
   `±inf` so that structures touching the border are not mistaken for holes. Copying
   that exposed a real bug here: a ring that filled the frame was cut into arcs by
   the dilation clipping at the array edge, so its lumen leaked into the background
   and `hole_count` was 0. The raster frame is now expanded by the cell radius before
   mapping (see `raster_mask`).
2. **Guard against tiny maps.** They skip clusters with fewer than 5 points; this
   repo uses 8 for descriptors and 60 for intervention sampling, and reports how many
   ROIs were dropped.
3. **Make long runs resumable.** They write a `topo_done.txt` per slide. Here the
   shell script skips any step whose output already exists, which matters when a
   4090 is billed by the hour.

What was **not** taken: their persistence implementation. Theirs is a union-find over
a 2D image sublevel set; ours is H0 from an MST plus raster lumen counting, so there
is no shared code and no GPL dependency.

One more lesson from their training script: their loss weights are `1.0` for a binary
critical-point term but **`0.01`** for the persistence regression. When components have
different scales, the nominal weight says almost nothing — always log each component
separately and check its magnitude. That is why `train.py` prints `lm / struct / cf`
independently instead of one total.

## What reading other repos changed

| repo | what was taken |
|---|---|
| TopoSlide | pad before topology; guard tiny maps; resumable long runs |
| CellViT++ | its export is `{name}_cells.json` with `centroid` + `class` → added `pilot cells` to convert it directly |
| CPathOGen | grouped bootstrap (resample patients, not ROIs) and Benjamini–Hochberg → added `groups=` to `paired_bootstrap` and `benjamini_hochberg()` |
| ASPECT / PathBind | no public repository found; only their reported numbers and the PathoVernier dataset are available |

## Corrections made for the feasibility run (2026-10-21)

These were found while validating the pilot; the details and measurements are in
[`docs/FEASIBILITY.md`](docs/FEASIBILITY.md).

| what was wrong | why it mattered | fix |
|---|---|---|
| the counterfactual prompt omitted the edit kind (and the translation vector) | the same image + box had both yes and no labels; the task was not a function of the observation | prompt now names the operation; translate points at a drawn arrow |
| the prompt carried the box coordinates as text | a text-only classifier reached **0.69** balanced accuracy on held-out patients, above the 0.65 P1 gate | coordinates removed from the prompt; location is only the green box in the image (text-only now **0.50**) |
| no box was drawn, despite the README saying the model sees one | the model had to map 0–1000 text coordinates onto patches, a different task | `annotate_edit` draws a green box and, for translate, an orange arrow |
| the auxiliary heads used a global mean of image tokens | the 3×3 regional target is unlearnable from a mean, and the cf head could not see the edit region | both heads consume the image-token grid; structure head 3×3 pools, cf head pools the edit region |
| the regional descriptor was normalised by cell quantiles and indexed x-major | target tiles did not line up with the image token grid | `describe(..., frame=...)` uses the image frame; row-major tile order |
| translation vector was absent from the cf head features | the same (centre, radius, kind) had different labels for different directions | `MASK_DIM` 6 → 8 |
| CellViT classes were factorised per ROI | the same class got different ids in different ROIs, corrupting the type block | class labels stay strings; one global vocabulary |
| Otsu returned a bin centre below a constant background | 4/36 synthetic ROIs became all-foreground and silently fell back to ground-truth cells | threshold guard; `segment_dir` deletes stale output first; synth GT and `cells_seg` are separate |
| P1/P2 evaluated on the training jsonl | gates measured memorisation, not generalisation | `pilot build --val-frac` patient-disjoint split; script evaluates `*_val.jsonl` |
| `--no-aux` was described as LoRA-only | it still sees the cf examples through the LM loss | documented as the same-data-mix ablation; not a growth-only baseline |
| Qwen3-VL was sized at 28×28 px/token | it uses 16 px patches × 2 merge = 32×32; the presets used ~391 of 512 tokens | `vision.PATCH=32`; `max_pixels` updated |
| `transformers` had no upper bound | pip picks 5.x, which requires torch≥2.6 and failed to import here | extra pins `transformers>=4.57,<5` |
| `--compare` keyed on `roi_id|gold` | several cf examples share a ROI/answer and collapsed in the dict | `ex_id` key |
| doctor did not check coordinate scale | a cell table in slide coordinates produced finite but meaningless descriptors | >5% out-of-bounds cells is now a problem |
| scikit-image was not a dependency | the documented `pilot segment` failed after a clean install | added to base dependencies |

## Honest notes

- The counterfactual labels are exact (computed from the cell point set), but the *images* are
  not edited — the model sees the original ROI with a green box (and an orange arrow for
  translate). It must reason about what would happen inside the box, which is the ability we
  want, but it is not the same as seeing an edited image.
- `build_cf_examples` computes one descriptor per ROI and reuses it for all edits of that ROI.
  That is correct for the structure head (the descriptor is of the *original* tissue) but means
  the auxiliary regression target does not vary within a ROI.
- The synthetic cohort uses ring/solid/multifocal shapes. Good for plumbing, useless for
  pathology claims. P2 on it is likely to hit a ceiling or be underpowered; it validates
  the training loop, not a pathology effect.
- The bundled classical segmenter is a fallback. It is enough to prove the chain
  pixels → cells → topology works, but every number you intend to publish should come
  from CellViT++ or HoVer-Net masks. It also assigns one type to every nucleus, so the
  type-composition block is trivial on synthetic `cells_seg`; the CellViT path keeps types.
- The prompt defines "architecture change" operationally (connected groups or enclosed
  lumina). Report that definition; a human's notion of architecture is broader.
- Include the text-only baseline in the paper. If a future edit makes coordinates textual
  again, the baseline will catch the shortcut.

## License

MIT.
