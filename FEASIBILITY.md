# Feasibility report — 2026-10-21

This is the result of running the pilot end to end on one RTX 4090 D (24 GB),
with the synthetic cohort. It is a **pipeline + learnability** validation, not
a pathology result: no real H&E data were available.

## Verdict in one paragraph

The easy counterfactual task is **learnable**, but the advantage is **not
structural reasoning**. A Qwen3-VL-4B with LoRA goes from 0.523 balanced
accuracy zero-shot to **0.674** on 11 held-out patients (two seeds, identical
point estimate), while the same adapters score **0.500** when the cells are
removed from the image and only the green box / orange arrow remain; a
text-only model also scores **0.500**. So the model does look at cells on the
easy version. But 14% of validation boxes are empty and trivially "no" (P1
gets 19/19), and on a strict non-empty, kind/label-balanced rebuild at 384 px
P1 is **0.500**. The 0.674 is a shortcut composite (empty boxes plus residual
priors), not a structural result. At 144 visual tokens a synthetic cell is
smaller than one token (6 px cell vs 32 px token), so the local arrangement the
task asks about is not resolvable. The **LM** high-resolution check (1024
tokens, ground-truth cells, 80 cells/ROI) gives **0.534** [0.481, 0.582], again
indistinguishable from chance; the LM does not learn from 254 examples and 150
steps. The trained **heads** are a different story: the structure head learns
topology and the strict-384 counterfactual head learns a real visual signal —
see the module evaluation below.

**Bottom line (LM readout):** the pipeline is validated; the *language* yes/no
task is shortcut-driven at 384 px and not learned at 1024 px.

**Bottom line (module readout):** the modules do learn. The structure head
reaches topology R² 0.315 (strict 384) / 0.411 (strict 1024) on held-out
patients, with blank-box controls strongly negative. The counterfactual head
reaches 0.640 balanced / AUC 0.652 on strict 384, with blank 0.509 / AUC 0.516;
on strict 1024 it reaches 0.686/AUC 0.744 but the blank AUC is 0.728, so that
number is mostly box geometry. The cleanest evidence is the strict-384 cf head
(just below 0.65, modest but visual) plus the structure head. **The idea is
partially validated at the module level; the LM generation and downstream
transfer are the open halves.**

**Real H&E (MoNuSeg, 37 train / 14 test patients):** the counterfactual head
averages **0.611** over 3 seeds (0.583/0.600/0.650), with blank 0.500 and
box-geometry 0.467; the **expand** edit is stable at 0.722, translate is ~0.56,
collapse has only 8 test examples. The **structure head does not generalise**
on this cohort (topology R² negative for every seed).

**Real H&E (PanNuke, 405 train / 160 held-out patches, 4 untouched tissues):**
the cf head is **0.588** over 2 seeds (0.595/0.582), blank 0.530, box-geometry
0.577; real-minus-blank is +8.1 / +3.9 pt, but the training loss stayed at
0.5–0.8 (underfit, 1.3 epochs) and the score is only ~1 pt above the geometry
prior. The structure head again does **not** generalise (topology R² negative).

**Across both real cohorts:** the counterfactual head carries a small but real
visual signal (MoNuSeg expand 0.722; PanNuke translate 0.613; real > blank),
while the **descriptor/structure head fails on real H&E** at this data scale
(37–405 images). The real-data idea is partially supported for the cf task, not
yet for the structural regression.

**The auxiliary-supervision half of the idea is not yet demonstrated.** On the
same cf task, LoRA-only (no aux heads/losses, same data and steps) gets 0.644;
the aux arm gets 0.674, a +3.0-point difference with CI [−1.5, +6.8]. P2 is
inconclusive because the synthetic growth task saturates (both arms 1.000 on 22
held-out ROIs, zero-shot already 0.667), and the structure probe shows no
consistent advantage for the aux backbone. So: the **data pipeline is
feasible**; the **task presentation is not sufficient at this scale**, and the
**transfer claim needs a harder downstream task or real data**, plus possibly a
stronger auxiliary signal.

| experiment | balanced accuracy |
|---|---|
| P0, easy cf | 0.523 |
| P1, easy cf, two seeds | 0.674 |
| P1, easy cf, cells removed (box only) | 0.500 |
| P1, strict non-empty 384 px, kind/label balanced | **0.500** |
| P1, strict non-empty 1024 px, 1024 tokens (LM) | 0.534 |
| cf **head**, strict non-empty 384 px | **0.640** (blank 0.509) |
| cf **head**, strict non-empty 1024 px | 0.686 (blank AUC 0.728; geometry-confounded) |
| structure **head** topology R², strict 384 / 1024 | 0.315 / 0.411 (blank negative) |
| P2 growth, aux / no-aux | 1.000 / 1.000 (ceiling) |
| **real** MoNuSeg cf head, 3 seeds | **0.611** (blank 0.500; geometry 0.467) |
| **real** PanNuke cf head, 2 seeds | **0.588** (blank 0.530; geometry 0.577; underfit) |
| **real** structure head, both cohorts | negative topology R² |

## Real-data follow-up (MoNuSeg + PanNuke)

The synthetic pilot was followed by two small real H&E datasets, both with
expert nucleus masks (so the cell tables are annotations, not a segmenter's
guess):

- **MoNuSeg** (RationAI HF, CC BY-NC-SA): 37 train / 14 test images,
  1000×1000, 294–1862 nuclei/image, patient IDs, 8 tissues. No cell types.
  The provided test split is patient-disjoint and is used as an external val.
- **PanNuke fold1** (RationAI HF): 405 train patches from 15 tissues and 160
  held-out patches from 4 tissues (breast, colon, bile, cervix); 256×256
  upscaled ×4; 16–273 nuclei; 5 cell types. The split is by tissue, so the
  held-out result is a domain-generalisation test and avoids WSI-patch leakage.

The counterfactual labels are exact from the real cell tables; the box/arrow
presentation is unchanged. The heads are evaluated directly (the same
`aux_heads.pt` readout as above), not through LM generation.

### MoNuSeg

116 train cf rows (58/58) from 30 patients; 60 test rows (30/30) from 14
held-out patients. Box-geometry train→test is **0.467 balanced / AUC 0.528**:
on real data the box position carries no label prior. The cf/structure heads
were trained for 150 steps at 1024 tokens, 3 seeds.

| seed | cf real bal | real AUC | blank bal | blank AUC | expand | translate | collapse | topology R² real | blank |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.583 | 0.637 | 0.517 | 0.529 | 0.667 | 0.571 | 0.375 | −72.6 | −118.4 |
| 1 | 0.600 | 0.610 | 0.483 | 0.643 | 0.708 | 0.536 | 0.500 | −29.4 | −43.7 |
| 2 | **0.650** | **0.676** | 0.500 | 0.504 | 0.792 | 0.571 | 0.500 | −20.4 | −114.1 |
| mean | **0.611** | 0.641 | 0.500 | 0.559 | **0.722** | 0.559 | 0.458 | negative | negative |

- The **counterfactual head has a real but modest visual signal on real H&E**:
  mean 0.611, blank 0.500, geometry 0.467. The 3 seeds range 0.583–0.650;
  seed 2 reaches the 0.65 gate, seed 0's patient-bootstrap CI is
  [0.483, 0.689]. With 60 test rows / 14 patients, this is not a robust pass.
- The signal is concentrated in **expand** (0.722, stable across seeds);
  translate is ~0.56 (near chance); collapse has only 8 test examples
  (0.38–0.50) and is uninformative.
- The **structure head does not generalise on MoNuSeg**: topology R² is
  negative for all seeds (and for the blank control). With 37 training images
  it overfits the descriptor; the real topology regression needs more data or
  a different target. This is a clear negative for the structure head on this
  cohort, in contrast to the synthetic result.

### PanNuke

405 train patches from 15 tissues and 160 held-out patches from 4 tissues
(breast, colon, bile, cervix); 1796 train cf / 620 held-out cf, balanced;
5 cell types; 2 seeds × 300 steps at 1024 tokens. Box-geometry train→held-out
is 0.577 balanced / AUC 0.580.

| seed | cf real bal | real AUC | blank bal | blank AUC | collapse | expand | translate | topology R² real |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.595 | 0.645 | 0.515 | 0.591 | 0.544 | 0.597 | 0.627 | −3.76 |
| 1 | 0.582 | 0.626 | 0.544 | 0.580 | 0.544 | 0.593 | 0.598 | −5.84 |
| mean | **0.588** | 0.636 | 0.530 | 0.585 | 0.544 | 0.595 | 0.613 | negative |

Seed 0: 620 held-out rows, 4 tissue groups; balanced 0.595 (bootstrap CI
[0.560, 0.643]), AUC 0.645 (CI [0.606, 0.688]); blank 0.515; real-minus-blank
**+8.1 pt, CI [5.7, 10.3]**. Seed 1: real 0.582, blank 0.544, paired +3.9 pt.
The cf head training loss stayed around 0.5–0.8 (1796 rows, 300 steps ≈ 1.3
epochs), so the head is **underfit**, not converged; the held-out-tissue result
is a lower bound. It is also only ~1 point above the 0.577 box-geometry prior,
so the visual edge over geometry is small on this split. The blank paired
delta is positive but the geometry control absorbs most of it.

The structure head again fails to generalise on held-out tissues (topology
R² −3.76 / −5.84; blank −22 / −36); the 5-type descriptor is not learned across
the tissue split.

### 2026-10-22 extension: 600 steps and topology-only structure head

To test the two follow-ups (more steps; drop the 3×3 regional block), PanNuke
was trained for 600 steps (1 seed, `--structure-dims 5`, 1796 cf), and MoNuSeg
for 150 steps with the same topology-only head (3 seeds, checkpoints at 50/100).

**PanNuke 600.** Training cf fell 0.527→0.364 (lm 0.261→0.134), but held-out
cf was **0.600** (CI [0.563, 0.649]), AUC 0.657, versus 0.588 at 300 steps —
a +1.2 pt gain with overlapping CIs. Blank 0.547; paired real−blank +5.3 pt
(CI [0.5, 10.9]). Per-kind: collapse 0.544 / expand 0.606 / translate 0.631.
The topology-only structure head still had negative held-out R² (−6.16), driven
by `hole_area_frac` (−31.2); `h0_sum` was +0.52, other dims near zero.

**MoNuSeg topology-only (3 seeds).** Held-out cf 0.600/0.583/0.600 (mean 0.594,
vs 0.611 with the full descriptor), blank 0.489, expand 0.750, translate 0.524,
collapse 0.375 (n=8); paired real−blank +15.0/+5.0/+11.7 pt. Train cf collapsed
to 0.016/0.048/0.004 by step 150. Checkpoints at step 100 gave cf 0.583/0.600/
0.550 (mean 0.578), no better than the final model, but a less negative
structure R² for two seeds (−19 vs −45; −19 vs −22). Topology R² remained
negative, dominated by `hole_area_frac` (−223/−110/−193).

**Conclusion.** Neither longer training nor removing the regional block fixes
the held-out structure regression; the constraint is data scale and the target
definition (especially `hole_area_frac`), not optimisation length. The
counterfactual head keeps a small real signal (expand is the stable component),
still below a robust 0.65. Full tables, CIs and the recovery timeline are in
`FEISHU_REPORT.md` §3.4 / Appendix C.

## Environment

| | |
|---|---|
| GPU | RTX 4090 D, 24 GB |
| torch | 2.5.1+cu124 |
| transformers | 4.57.6 |
| peft | 0.21.2 |
| model | Qwen/Qwen3-VL-4B-Instruct, bf16, LoRA r=8 alpha=32, `all-linear` |
| training | batch 1, grad-accum 8, AdamW, cosine, 150 steps (P1) / 300 (P2) |
| VRAM | 9.2 GB peak at 144 tokens; 11.3 GB at 1024 tokens |
| speed | ~0.25 s per micro-batch at 144 tokens (P1 4.2 min, each P2 arm ~8 min); P1 hi 11 min for 150 steps |

## Data and split

- 72 synthetic ROIs (24 ring / 24 solid / 24 multifocal), 36 patients.
- Classical segmentation: 72/72 ROIs, median 136 cells (the generator makes
  220; touching nuclei merge), one type per nucleus.
- Counterfactual rows: 430 (6 per ROI, balanced within each edit kind and
  label), 298 train / 132 val.
- Growth rows: 72, 50 train / 22 val.
- Split by patient: 25 train patients / 11 val patients, the same patients for
  both tasks.
- The prompt has no coordinates. The location is a green box drawn on the
  image; translate edits also draw an orange direction arrow. The prompt
  defines "architecture change" as a change in the number of connected cell
  groups or enclosed lumina.

## Controls

| control | n | balanced acc | what it rules out |
|---|---|---|---|
| TF-IDF + logistic on the prompt text | 132 | **0.500** | reading the answer from wording/kind |
| P1 seed 0 on boxes only (cells removed) | 132 | **0.500** | using box position / edit type without cells |
| P1 seed 1 on boxes only (cells removed) | 132 | **0.500** | same |
| P1 seed 0 on real images | 132 | **0.674** | (the result) |
| P1 seed 1 on real images | 132 | **0.674** | (the result) |

The text-only control was **0.689** before the prompt was fixed. The original
prompt carried the box corners as text; a logistic regression on kind + box +
centre + radius + vector reached 0.689 on held-out patients, above the 0.65
gate. Removing the coordinates dropped it to 0.500. This was the single most
important correction: without it, P1 could have "passed" without using the
image.

Box geometry is also a **visual** shortcut, even after the text is removed: a
logistic regression on the same kind/box/centre/radius/vector features (no
image) reaches 0.689 on the original 384 rows, 0.596 on the strict 384 rows and
0.644 on the high-resolution rows. A VLM could in principle read the box
position instead of the cells. The original 384 adapter did not (blank control
0.500); the high-resolution blank control is reported below. Match box
geometry between yes and no, or centre the box in a crop, to remove this
shortcut by design.

## Results

### P0 — zero-shot

| run | n | accuracy | balanced | notes |
|---|---|---|---|---|
| counterfactual | 132 | 0.523 | 0.523 | 97.7% predicted "yes"; sensitivity 1.000, specificity 0.045 |
| growth | 22 | 0.727 | 0.667 | multifocal 1.0, ring 1.0, solid 0.0 |

P0 is in the intended 0.5–0.65 band: the task is not solved zero-shot, and the
model has a strong yes bias.

### P1 — learn the counterfactual task

Two seeds, same split, same 150 steps. The adapters differ (different hashes;
14/132 predictions differ), so this is a real replication, not a frozen seed.

| seed | balanced acc | accuracy | predicted yes | paired gain over P0 |
|---|---|---|---|---|
| 0 | **0.674** | 0.674 | 0.583 | **+15.2 pt**, CI [8.3, 22.0] |
| 1 | **0.674** | 0.674 | 0.568 | **+15.2 pt**, CI [8.3, 22.7] |

Patient-bootstrap 95% CI for seed 0: [0.591, 0.758]. Parse failures: 0/132 in
both seeds.

Per edit kind (44 val examples each):

| kind | seed 0 | seed 1 | seed 0 95% CI |
|---|---|---|---|
| collapse | 0.682 | 0.682 | [0.545, 0.818] |
| expand | 0.727 | 0.795 | [0.659, 0.795] |
| translate | 0.614 | 0.545 | [0.477, 0.750] |

Translation is the weak and unstable edit: the model must infer the effect of
a directed shift from a static arrow. Collapse is stable, expand improves with
seed 1. Translation is the obvious target for a before/after image or a
stronger edit representation.

**Reading the gate.** The point estimate passes 0.65 in both seeds, but with 11
held-out patients the CI still contains 0.65. The defensible claim is "the
task is learnable and the improvement is real"; not "the effect is robustly
above 0.65". More holdout patients, not more seeds on the same split, are what
would tighten the absolute estimate.

#### Empty boxes are a free shortcut

20.1% of training boxes and 14.4% of validation boxes contain no cells. Every
such example is "no" (moving nothing changes nothing), and P1 gets 19/19 of the
validation ones. Excluding them gives the structural result:

| arm | n (non-empty) | balanced acc | 95% CI | paired vs P0 |
|---|---|---|---|---|
| P1 seed 0 | 113 | 0.592 | [0.504, 0.674] | +9.2 pt, CI [0.4, 17.4] |
| P1 seed 1 | 113 | 0.605 | [0.522, 0.699] | +10.5 pt, CI [2.2, 19.9] |
| P1 no-aux | 113 | 0.570 | [0.458, 0.680] | +5.1 pt, CI [−4.2, 18.0] |
| P0 (constant yes) | 113 | 0.500 | — | — |

So the structural margin over chance is **modest**, not the 0.674 headline:
seed 0's CI nearly touches 0.5, seed 1's is just above it. The correct claim is
"a small, positive structural signal on non-empty regions", not "the model
reaches 0.674 on structural reasoning". The next dataset should pass
`--min-affected 5` (or 8) and report the non-empty subset separately.

#### Strict non-empty control at 384 px: the signal disappears

The post-hoc non-empty subset is not enough: it still has 66 yes / 47 no and
unequal counts per kind, so a prior can contribute. We rebuilt the cohort with
`--min-affected 5` and then down-sampled per `(kind, label)` to an exact
balance: 226 train / 114 val, 57 yes / 57 no in val, 34/40/40 per kind.

| arm | n | balanced acc |
|---|---|---|
| P0 zero-shot | 114 | 0.500 |
| P1, 150 steps, aux | 114 | **0.500** |
| P1, blank boxes | 114 | 0.500 |
| text-only prompt | 114 | 0.500 |

Per kind: collapse 0.500, expand 0.525, translate 0.475. The model does not
learn the non-empty, exactly balanced structural task at 384 px. It predicts
"no" for 94.7% of the validation rows (P0 predicts "yes" for 100%); balanced
accuracy 0.5 is the no-bias, not partial learning. The training loss stayed at
0.23 (it did not fit), which points at the input, not at label noise. A
logistic regression on the box geometry alone reaches 0.596 on this split, so
a box shortcut exists in the data; the VLM simply does not exploit it at
384 px.

This is the most important negative result in the pilot: the 0.674 headline
was a **shortcut composite** (empty boxes plus residual priors). With both
controlled, a 4B model at 144 visual tokens is at chance.

#### High-resolution check: is 144 tokens the problem?

At 384 px, Qwen3-VL gives a 12×12 token grid; one token covers 32×32 px while a
synthetic cell is a 6 px dot. The model literally cannot see the cells whose
arrangement decides the label. The README anticipated this ("if it fails, the
resolution may simply be too low"). To separate "idea dead" from "input too
coarse", we built a second cohort: 1024×1024 images, 80 cells per ROI, dot
radius 10 px, **ground-truth cell tables** (no segmentation noise), and the new
`4b_bf16_1024` preset (32×32 = 1024 visual tokens, 11 GB). The counterfactual
set is min-affected 5 and exactly balanced per kind and label: 254 train / 118
val, 59 yes / 59 no in val.

A logistic regression on the **box geometry alone** (kind + box corners +
centre + radius + vector, no image) reaches **0.644** on this val split. The
box position is therefore a real visual shortcut: on these synthetic layouts,
where the box sits already predicts the label. A high-res P1 number is only
interpretable against the blank-box control; see the result below.

| arm | n | balanced acc | 95% CI |
|---|---|---|---|
| box geometry logistic (no image) | 118 | 0.644 | — |
| P0 zero-shot | 118 | 0.500 | [0.500, 0.500] |
| P1 hi, real images | 118 | **0.534** | [0.481, 0.582] |
| P1 hi, blank boxes | 118 | 0.500 | [0.500, 0.500] |

Per kind: collapse 0.605, expand 0.432, translate 0.583. The adapter predicts
"yes" on 49.2% of rows (P0: 100%, blank: 0%). Real-minus-blank is **+3.4 pt,
CI [−3.4, +9.9]**; real-minus-P0 is the same, CI [−2.9, +9.4]. Both cross
zero. The training loss stayed around 0.18–0.27; the model did not fit the
task.

**The high-resolution check does not rescue the idea.** At 1024 tokens with
ground-truth cells and 80 cells per ROI, the model is at 0.534, statistically
indistinguishable from chance and far below the 0.65 gate. It also does not
exploit the box-geometry shortcut (the blank control is 0.500), so the 0.534
is a weak visual effect, not box reading. With 254 training examples and 150
steps, Qwen3-VL-4B cannot learn this counterfactual task from a full-ROI
box+arrow presentation. That is a limit of the data scale/presentation, not
of the topology code; the next experiments must change one of those (see
below).

#### Does the auxiliary loss help on the same task?

Same cf train/val, same 150 steps, only difference is `--no-aux`:

| arm | balanced acc | predicted yes |
|---|---|---|
| aux heads + losses | **0.674** | 0.583 |
| language loss only | 0.644 | 0.508 |
| paired difference | **+3.0 pt**, CI [−1.5, +6.8] | — |

The LM-only arm is just below the 0.65 gate; the aux arm is just above it, but
the difference is not significant. The auxiliary losses are not required to
learn the task (0.644 is close), and they do not demonstrably improve it. This
is consistent with the P2 cross-eval and the structure probe. It is a signal to
make the auxiliary target harder or more relevant, not a reason to discard the
task.

### Do the trained auxiliary heads learn? (direct module evaluation)

The LM numbers above are yes/no *generation*. The modules the pilot is built
around were also trained; evaluating them directly is the fair module-level
test. `scripts/eval_heads.py` loads `aux_heads.pt` and runs the two heads on
held-out patients.

| adapter / data | cf head bal | cf AUC | blank cf bal | blank cf AUC | structure topology R² | blank structure R² |
|---|---|---|---|---|---|---|
| P1 easy 384 | 0.659 | 0.751 | 0.500 | 0.584 | 0.422 | −1.824 |
| P1 strict 384 | **0.640** | **0.652** | **0.509** | **0.516** | **0.315** | −2.526 |
| P1 strict 1024 | 0.686 | 0.744 | 0.500 | 0.728 | 0.411 | −3.789 |
| P2 aux easy | 0.652 | 0.728 | — | — | 0.422 | — |

Box-geometry-only logistic AUC for reference: 0.743 (easy), 0.640 (strict
384), 0.706 (strict 1024).

**Reading this**

- The **structure head learns topology from cells**: topology R² = 0.315 on
  strict 384 and 0.411 on strict 1024, while the blank-box control is strongly
  negative. The regional block is not learned (mean R² negative); the topology
  block is.
- The **counterfactual head learns a real visual signal on strict 384**:
  0.640 balanced / AUC 0.652, with the blank-box control at 0.509 / AUC 0.516.
  It is just below the 0.65 gate, and the geometry-only prior is 0.596/AUC
  0.640, so the visual edge is modest — but it is not box reading.
- The **strict 1024 cf head at 0.686 is not clean evidence**: the blank AUC is
  0.728 and geometry-only is 0.706, so most of that score is the box prior.
  The structure head at 1024 is clean (R² 0.411; blank −3.789).
- The **LM is the bottleneck**: on strict 384 the same adapter gets 0.500
  yes/no generation but the head gets 0.640; on strict 1024, 0.534 vs 0.686.
  Training the modules with data works better than asking the LM to generate
  the answer.

**What this validates.** Your core claim — a structural head trained with
topology labels learns something about structure on held-out patients — is
supported: the structure head is clearly positive, and the cf head has a real,
if modest, visual signal on the clean 384 comparison. What is not yet shown is
a robust pass of the 0.65 gate by the cf head, a clean hi-resolution cf result
(it is geometry-confounded), transfer to the downstream growth task (P2
ceiling), or anything on real H&E.

### P2 — does the auxiliary loss transfer?

| run | data | n | balanced acc |
|---|---|---|---|
| aux | growth val | 22 | **1.000** |
| no-aux, same data mix | growth val | 22 | **1.000** |
| aux | cf val (cross-eval) | 132 | 0.689 |
| no-aux, same data mix | cf val (cross-eval) | 132 | 0.697 |

Paired comparisons: growth Δ = 0.0 (CI [0, 0], ceiling); cf Δ = −0.8 pt
(CI [−7.6, +6.1], no difference).

The synthetic growth task is too easy: zero-shot balanced accuracy is already
0.667 and both fine-tuned arms reach 1.000 on 22 held-out ROIs. **P2 cannot
distinguish the arms here.** This is not evidence against transfer; it is a
ceiling. A real downstream label, or a harder synthetic task with several
hundred ROIs, is required.

The `--no-aux` arm still sees the counterfactual examples through the language
loss. It isolates the auxiliary heads/losses, not a growth-only LoRA baseline.
A growth-only arm would answer a different question and should be added if the
paper reports "aux vs LoRA-only".

### Structure linear probe (P2 adapters, cf val, 132 examples)

A fresh ridge regression on the frozen adapter's image-token features,
predicting the 23-dimensional descriptor. Train on cf train patients, evaluate
on cf val patients. Lower MSE / higher R² is better.

| features | arm | MSE all | MAE | MSE topology | MSE regional | R² mean | R² topology |
|---|---|---|---|---|---|---|---|
| mean | aux | 9.58 | 0.583 | 44.07 | 6.22e-4 | 0.679 | 0.530 |
| mean | no-aux | **9.00** | **0.552** | **41.38** | 6.27e-4 | **0.685** | **0.552** |
| 3×3 grid | aux | **7.84** | **0.532** | **36.08** | 5.45e-4 | 0.726 | **0.585** |
| 3×3 grid | no-aux | 8.14 | 0.545 | 37.46 | **5.05e-4** | **0.728** | 0.563 |

The aux arm is not consistently better: grid features favour it on topology
MSE/R² by ~4%, mean features favour the no-aux arm by ~4%, and regional MSE
favours no-aux. The differences are small and mixed, not a transfer signal.
Both arms already decode the descriptor at R² ≈ 0.7, so this synthetic cohort
does not create the representation bottleneck that auxiliary supervision would
need to improve. A real-data probe with several seeds and a patient bootstrap
is the right next step; do not read the grid-vs-mean micro-differences as a
result.

## What this does and does not show

**Does show**

- The data pipeline is correct end to end: pixels → cells → topology →
  descriptors → counterfactual labels → LoRA training → held-out eval.
- The counterfactual task is well posed after the fixes: it is not solvable
  from text (0.500) or from the box alone (0.500), and a 4B VLM improves on the
  easy version with 150 steps of LoRA (+15.2 points over P0). Two seeds agree.
  On non-empty boxes the structural result is 0.592–0.605 (still above chance,
  but modest).
- The controls catch real failure modes: the strict non-empty, kind/label-
  balanced rebuild is 0.500 at 384 px and 0.534 [0.481, 0.582] at 1024 px with
  ground-truth cells. The pipeline is correct; the model is not doing
  structural reasoning in either full-ROI presentation.
- The auxiliary heads can be trained on the same 24 GB card with 9.2 GB peak
  VRAM, with gradients reaching the language backbone.
- The heads learn their training targets (struct loss 0.005, cf loss 0.05), so
  the absence of a transfer signal is not "the aux loss never converged".
- **The trained heads generalize better than the LM**: the structure head
  reaches topology R² 0.315/0.411 on held-out patients (blank controls
  negative), and the strict-384 cf head reaches 0.640 balanced / AUC 0.652
  with a blank control at 0.509 / AUC 0.516. That is a real visual signal,
  even though it is modest and just below the 0.65 gate.
- The topology labels are exact and balanced. The classical segmenter produces
  a cell table for all 72 ROIs; its ring/solid/multifocal unit test passes, but
  it merges touching nuclei (median 136 of 220 cells) and assigns one type, so
  the type block is trivial in this run.
- **The real-H&E pipeline works**: MoNuSeg and PanNuke expert nucleus masks →
  cell tables → cf rows → heads. The real cf head carries a small visual signal
  (MoNuSeg 0.611 mean; PanNuke 0.588; real > blank; expand 0.722 on MoNuSeg).
- **The structure head does not generalise on the real cohorts**: topology R²
  is negative on both MoNuSeg (14 held-out patients) and PanNuke (4 held-out
  tissues). The synthetic positive did not replicate at 37–405 images.

**Does not show**

- Any clinical/pathology effect. The synthetic cohort is dots; the real cohorts
  are small, the MoNuSeg test has 14 patients, and the PanNuke split is by
  tissue, not clinical outcome. No diagnostic claim.
- That the auxiliary structural loss improves downstream diagnosis. P2 is at
  ceiling; cf cross-eval shows no difference; the same-task aux ablation is
  +3.0 points with a CI crossing zero.
- That the structure head's training success means the backbone learned better
  structure. The probe suggests it did not change measurably.
- That the model is robust. Two seeds for P1 agree on 0.674 (per-kind varies);
  P2 is one seed; 11 held-out patients; the P1 CI includes 0.65.
- That the model reasons like a pathologist. It sees a box and an arrow and
  must imagine the edit; a before/after pair would be a stronger and probably
  easier task.
- That the structural effect is large. On the non-empty subset the CI lower
  bound is 0.504 (seed 0) / 0.522 (seed 1), barely above chance; the 1024 px
  strict result is 0.534 [0.481, 0.582].
- That the full-ROI box+arrow task is learnable by the **LM**: 0.500 at 384 px
  and 0.534 at 1024 px.
- That the cf head robustly passes 0.65: the clean strict-384 result is 0.640;
  the 1024 result (0.686) is confounded by box geometry (blank AUC 0.728). On
  real data the means are 0.611 (MoNuSeg) and 0.588 (PanNuke); the PanNuke head
  is underfit and only ~1 pt above its box-geometry prior.
- That the descriptor/structure regression transfers to real H&E: topology R²
  is negative on both real cohorts, in contrast to the synthetic 0.315/0.411.
- That the real-data result is large or stable: 14 MoNuSeg test patients,
  4 PanNuke tissue groups, 2–3 seeds, and a single 4B model.
- That the module-level signal is large; it is modest and needs more data, a
  better region representation, and real H&E to be convincing.

## Limitations and next steps

The module-level result changes the priority order: the heads learn, so the
next step is to make that result robust, not to abandon it.

1. **Scale the real data.** MoNuSeg has 37 train / 14 test patients; PanNuke
   405 train / 160 held-out patches. The real cf head means are 0.611 and
   0.588; the real structure head is negative on topology R². More real
   patients/patches is the direct next experiment. Report patient/tissue-level
   CIs (the MoNuSeg test has only 14 groups; PanNuke has 4).
2. **Fix the structure head on real H&E.** It learns topology on synthetic data
   (R² 0.315/0.411) but not on MoNuSeg or PanNuke. The regional block is
   consistently unlearned; consider dropping it, changing the target (e.g.
   local lumen/persistence rather than 3×3 type fractions), and scaling the
   training data. Do not report the synthetic structure-head result as a real
   result.
3. **Finish the PanNuke cf training.** The PanNuke cf head is underfit (train
   loss 0.5–0.8, 300 steps ≈ 1.3 epochs) and only ~1 pt above its box-geometry
   prior. Run 600–1200 steps or fewer patches; report the geometry and blank
   controls. The current 0.588 is a lower bound.
4. **MoNuSeg is the clean real cf signal.** Expand is stable at 0.722; translate
   is ~0.56; collapse has only 8 test examples. Add patients and examples for
   translate/collapse; a local crop / higher effective resolution is the
   natural next lever for the small nuclei.
5. **Keep the structure/descriptor target honest.** The synthetic positive did
   not replicate on real H&E. If the real structure head stays negative, the
   topology-supervision claim must be narrowed to the counterfactual head.
6. **Give P2 headroom.** Use a real growth/tissue label with a patch split for
   seen tissues, or a harder task. The synthetic P2 was at ceiling.
7. **Presentation is a separate lever.** The code now has
   `--presentation before_after`; it removes the mental-simulation step and the
   box-position shortcut. It is the right next test for the **LM** readout, not
   a substitute for the real-data structure-head result.
8. **Keep the controls and report the LM-only baseline.** Text-only, blank,
   empty-box, box-geometry and the LM-only arm are cheap and changed the
   interpretation here; run them for every real-data result too.

## Reproduce

```bash
pip install -e ".[qwen,dev]"
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-VL-4B-Instruct
export HF_HUB_OFFLINE=1

pilot synth --out data/synth --per-class 24 --size 384
pilot segment --images data/synth/images --out data/synth/cells_seg --backend classical
pilot doctor --manifest data/synth/manifest.csv --cells data/synth/cells_seg
pilot build --manifest data/synth/manifest.csv --cells data/synth/cells_seg \
  --out data/synth --per-roi 6 --val-frac 0.3

pilot p0   --data data/synth/cf_val.jsonl --preset 4b_bf16 --out out/pred_p0_cf.json
pilot train --cf data/synth/cf_train.jsonl --steps 150 --preset 4b_bf16 --out out/p1
pilot eval  --data data/synth/cf_val.jsonl --adapter out/p1 --preset 4b_bf16 \
  --out out/pred_p1.json

pilot train --cf data/synth/cf_train.jsonl --steps 150 --seed 1 --preset 4b_bf16 --out out/p1_seed1
pilot eval  --data data/synth/cf_val.jsonl --adapter out/p1_seed1 --preset 4b_bf16 \
  --out out/pred_p1_seed1.json

pilot train --cf data/synth/cf_train.jsonl --steps 150 --preset 4b_bf16 --out out/p1_noaux --no-aux
pilot eval  --data data/synth/cf_val.jsonl --adapter out/p1_noaux --preset 4b_bf16 \
  --out out/pred_p1_noaux.json

pilot train --growth data/synth/growth_train.jsonl --cf data/synth/cf_train.jsonl \
  --steps 300 --preset 4b_bf16 --out out/p2_aux
pilot train --growth data/synth/growth_train.jsonl --cf data/synth/cf_train.jsonl \
  --steps 300 --preset 4b_bf16 --out out/p2_lora --no-aux
pilot eval --data data/synth/growth_val.jsonl --adapter out/p2_aux  --out out/pred_aux.json
pilot eval --data data/synth/growth_val.jsonl --adapter out/p2_lora --out out/pred_lora.json \
  --compare out/pred_aux.json

python scripts/probe_structure.py --aux out/p2_aux --lora out/p2_lora \
  --train data/synth/cf_train.jsonl --val data/synth/cf_val.jsonl

# direct evaluation of the trained heads (the module-level test)
python scripts/eval_heads.py --adapter out/p1_hard \
  --data data/synth_hard/cf_val_bal.jsonl --preset 4b_bf16
python scripts/eval_heads.py --adapter out/p1_hi \
  --data data/synth_hi/cf_val_bal.jsonl --preset 4b_bf16_1024
python scripts/eval_heads.py --adapter out/p1_hard \
  --data data/synth_hard/cf_val_bal_blank.jsonl --preset 4b_bf16
python scripts/eval_heads.py --adapter out/p1_hi \
  --data data/synth_hi/cf_val_bal_blank.jsonl --preset 4b_bf16_1024

# strict non-empty control, 384 px
pilot build --manifest data/synth/manifest.csv --cells data/synth/cells_seg \
  --out data/synth_hard --per-roi 6 --val-frac 0.3 --min-affected 5
python scripts/balance_cf.py --train data/synth_hard/cf_train.jsonl \
  --val data/synth_hard/cf_val.jsonl \
  --out-train data/synth_hard/cf_train_bal.jsonl \
  --out-val data/synth_hard/cf_val_bal.jsonl
pilot train --cf data/synth_hard/cf_train_bal.jsonl --steps 150 \
  --preset 4b_bf16 --out out/p1_hard
pilot eval --data data/synth_hard/cf_val_bal.jsonl --adapter out/p1_hard \
  --preset 4b_bf16 --out out/pred_p1_hard.json

# high-resolution check: 1024 px, 80 cells, ground-truth tables
pilot synth --out data/synth_hi --per-class 24 --size 1024 --n-cells 80 --dot-radius 10
pilot build --manifest data/synth_hi/manifest.csv --cells data/synth_hi/cells \
  --out data/synth_hi --per-roi 6 --val-frac 0.3 --min-affected 5
python scripts/balance_cf.py --train data/synth_hi/cf_train.jsonl \
  --val data/synth_hi/cf_val.jsonl \
  --out-train data/synth_hi/cf_train_bal.jsonl \
  --out-val data/synth_hi/cf_val_bal.jsonl
pilot train --cf data/synth_hi/cf_train_bal.jsonl --steps 150 \
  --preset 4b_bf16_1024 --out out/p1_hi
pilot eval --data data/synth_hi/cf_val_bal.jsonl --adapter out/p1_hi \
  --preset 4b_bf16_1024 --out out/pred_p1_hi.json
python scripts/make_blank_control.py --data data/synth_hi/cf_val_bal.jsonl \
  --out data/synth_hi/blank --jsonl data/synth_hi/cf_val_bal_blank.jsonl
pilot eval --data data/synth_hi/cf_val_bal_blank.jsonl --adapter out/p1_hi \
  --preset 4b_bf16_1024 --out out/pred_p1_hi_blank.json
```

```bash
python scripts/make_blank_control.py --data data/synth/cf_val.jsonl \
  --out data/synth/blank --jsonl data/synth/cf_val_blank.jsonl
pilot eval --data data/synth/cf_val_blank.jsonl --adapter out/p1 \
  --preset 4b_bf16 --out out/pred_p1_blank.json
```

## Raw result files

`out/pred_p0_cf.json`, `out/pred_p1.json`, `out/pred_p1_blank.json`,
`out/pred_p1_seed1.json`, `out/pred_p1_seed1_blank.json`,
`out/pred_p1_noaux.json`, `out/pred_p0_hard.json`, `out/pred_p1_hard.json`,
`out/pred_p1_hard_blank.json`, `out/pred_p0_hi.json`, `out/pred_p1_hi.json`,
`out/pred_p1_hi_blank.json`, `out/pred_aux.json`, `out/pred_lora.json`,
`out/pred_p2aux_cf.json`, `out/pred_p2lora_cf.json`.
