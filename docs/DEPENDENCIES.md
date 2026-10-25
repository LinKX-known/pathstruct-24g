# What is borrowed, and what is not

Rule: **if a maintained implementation exists, call it.** Hand-written versions are
allowed only for the part nobody has published (the evidence-token architecture), and
must sit behind a thin adapter so it can be swapped.

## Segmenters

| need | use | how | status |
|---|---|---|---|
| nuclei segmentation + 5-class typing | **CellViT++ / CellViT-Inference** (`pip install cellvit`) | `cellvit-inference --config <yaml>`, then `pilot cells --json-dir <out>` | official, config-driven; wrapper in `scripts/data/run_cellvit.py` |
| quick CPU sanity run | `pilot segment --backend classical` | Otsu + watershed | **debug only** — not for anything reported |
| WSI reading / tiling fallbacks | `tiatoolbox`, `openslide-python` | direct import | not wired in yet |

CellViT-Inference is the supported package (PyPI, docs, Colab). The research repos
(`CellViT`, `CellViT-plus-plus`) contain training code and a different inference
interface; do not mix them.

## Topology

| need | use | status |
|---|---|---|
| persistent homology (H0/H1) | **gudhi** (`DTMRipsComplex`) | `structures.gudhi_dtm_summary` — optional import |
| faster H1 for large point sets | **ripser** | interface present, not used yet |
| raster morphology (holes, components) | `scipy.ndimage` | used directly |
| our MST H0 fallback | hand-written | kept only when gudhi is absent |

## VLM training

| need | use | status |
|---|---|---|
| Qwen3-VL loading, processor, chat template | `transformers` + `qwen_vl_utils` | used |
| LoRA / QLoRA | `peft` (`get_peft_model`, `prepare_model_for_kbit_training`) | used |
| image resizing to a token budget | `qwen_vl_utils` (or `processor`), not our own `fit_image` | **should replace `vision.fit_image`** |
| multi-stage SFT | `ms-swift` is the documented route for Qwen3-VL | **not adopted** — the evidence-token loss needs a custom step, see below |

## The irreducible custom part

`src/pilot/sft.py` and the token layout in `src/pilot/tokens.py` have **no upstream**:
ASPECT's code is not public, and ms-swift has no hook for "supervise the hidden states
at these nine special tokens". Everything else in the repository is either a library
call or a thin adapter.

That is where bugs will live, so it is the part that needs the most tests. Current
coverage: staging/masking/positions/parsing/scoring are unit-tested; the model forward
is not (no GPU in the development environment).

## What should be deleted once the replacement is verified

| file | reason |
|---|---|
| `segment.segment_classical` default path | CellViT is the real segmenter |
| `vision.fit_image` | duplicate of `qwen_vl_utils` behaviour |
| `topology_h0.h0_deaths` | only a fallback for a missing gudhi |
| hand-written balanced accuracy in `eval.py` | `sklearn.metrics` already has it |
