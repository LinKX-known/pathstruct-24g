# Architecture: structure as generated evidence

The goal is a pathology VLM whose **visual evidence is enhanced by topology**, evaluated
head-to-head against ASPECT. This note pins down the one design decision everything else
follows from.

## The decision

Topology is **not** a side head. It is a block of tokens the model generates, a
measurement it reports, and the thing its answer has to be consistent with.

```
<|think|> <|uni|>x8 <|cell|>x6 <|struct|>x8 <|/think|>
<|observe|> {"components":"1","lumina":"3+","lumen_size":"large"} <|/observe|>
<|answer|> One component, three or more lumina, large lumen area. <|/answer|>
```

Why this shape:

- **A side head cannot change an answer.** The previous pilot proved it empirically: a
  counterfactual head reached 0.61 balanced accuracy while the language model reading the
  same backbone stayed at 0.50. Evidence that is not generated is not used.
- **ASPECT's own argument applies one level up.** It supervises cell tokens because mean
  pooling cannot preserve counts; the same argument says pooled patch tokens cannot
  preserve arrangement. `<struct>` is that third kind of evidence.
- **The observation is binned, not a float.** `components`, `lumina`, `lumen_size` are
  short and learnable, while a token-level regression head keeps full precision for the
  training signal. The bins are also exactly what a consistency reward can check.

## The three supervision signals on the evidence tokens

| tokens | target | loss |
|---|---|---|
| `<uni>` x8 | frozen pathology encoder features (UNI / CONCH) | feature reconstruction |
| `<cell>` x6 | frozen CellViT instance features per cell class | feature alignment |
| `<struct>` x8 | the structure descriptor (connectivity, lumina, persistence, regional composition) | smooth-L1 on `log1p` |

Then the language loss covers the whole response, including the observation JSON and the
answer, so the model learns to *write* what it encoded.

## Training stages (ASPECT's curriculum, with one more block)

1. **Perceive** — evidence tokens are supplied in the prompt; the model answers from them.
2. **Generate** — the model must emit the token block itself.
3. **Reason** — full response: tokens, observation, answer, trained with CE.
4. **Consistency (RL)** — reward = answer correctness + the answer agreeing with the
   model's own reported observation + the observation agreeing with the image's true
   descriptor within tolerance.

Stage 4 is where this differs most from ASPECT: its `Con` only checks counts, ours checks
structure as well.

## What is implemented now

| piece | where | state |
|---|---|---|
| token format, binning, parsing | `src/pilot/tokens.py` | done, tested |
| staging, masking, scoring | `src/pilot/evidence.py` | done, tested |
| SFT loop + evidence heads | `src/pilot/sft.py` | done, needs a GPU to run |
| corpus builder | `scripts/build/build_structure_qa.py` | done, CPU |
| training CLI | `scripts/train/train_structure_vlm.py` | done |
| `<uni>` reconstruction loss | `sft.py` | **off**: the corpus carries no encoder features yet |
| `<cell>` count loss | `sft.py` | on when the corpus has `cell_counts` (it does) |
| consistency RL (stage 4) | -- | not implemented |

The honest gap: `<uni>` reconstruction is the one ASPECT loss we cannot compute
yet, because it needs a frozen pathology encoder pass over every ROI. Adding it
is a feature-extraction job, not a modelling decision.

## What the comparison must hold fixed

| | ASPECT | this work |
|---|---|---|
| base model | Qwen3-VL-8B | same |
| training budget | their corpus | our structure-QA corpus at equivalent token budget |
| evaluation | PathoVernier (Acc/CA/RAWR) + PathCLS/PathVQA/Quilt-VQA | same, plus arrangement d' and structure-observation accuracy |

Their training data is not public, so the honest claim is **superiority on the structure
axis at matched budget**, with their published numbers quoted as reference and their
objective re-implemented as the baseline arm on our data. Claiming overall SOTA against a
million-sample corpus with a few thousand rows would not survive review.

## Data

`scripts/build/build_structure_qa.py` turns any manifest + cell tables into this format:

- `observe` rows: report the binned organisation of a region (the evidence-generation task);
- wrapped `cf` / `label` rows: existing QA, same response format, descriptor attached so the
  `<struct>` tokens stay supervised.

At 405 PanNuke patches and 51 MoNuSeg images, repeated over edits, this reaches tens of
thousands of rows without any new annotation.

## Open questions (deliberate, not oversights)

- Special tokens vs. plain-text markers: currently added as real special tokens so each slot
  is one position; if the tokenizer resize is a problem, `tokens.py` can fall back.
- Whether `<struct>` should be 8 tokens or fewer (a lumen is a handful of 32 px tokens at
  1024 px input; 8 is a guess to be ablated).
- Whether the observation should be JSON or free text. JSON is checkable; free text is more
  natural. Start with JSON.
