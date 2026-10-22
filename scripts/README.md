# scripts

One-off drivers that sit on top of the `pilot` CLI. Run them from the repository
root; every path below is relative to it.

## data/

Import public nucleus-annotated datasets into the layout the pipeline expects
(`manifest.csv` with `roi_id, patient_id, image_path, label`, plus
`cells/{roi_id}.csv` with `x, y, type`).

| script | what |
|---|---|
| `data/import_monuseg.py` | MoNuSeg parquet export → 37/14 patient-disjoint splits |
| `data/import_pannuke.py` | PanNuke fold1 parquet export → 405 train / 160 tissue-held-out |

## build/

| script | what |
|---|---|
| `build/balance_cf.py` | rebalance a counterfactual JSONL by edit kind and label |

## eval/

| script | what |
|---|---|
| `eval/eval_heads.py` | load `aux_heads.pt` from an adapter and score the two heads on held-out rows |
| `eval/bootstrap_heads.py` | patient/tissue-clustered bootstrap CIs for head metrics |
| `eval/probe_structure.py` | linear/ridge probe for how much structure a representation carries |

## controls/

| script | what |
|---|---|
| `controls/make_blank_control.py` | same rows, cells removed, green box kept — the blank-box control |

## run_pilot.sh

End-to-end synthetic run (`bash scripts/run_pilot.sh`); idempotent, stops if
`pilot doctor` fails.

## Conventions

- Reports go to `docs/`, never to the repository root.
- Anything under `data/`, `out/`, `work/` is ignored by git.
- Numbers that will be quoted anywhere must come from a patient- or
  tissue-disjoint split, with the control row next to them.
