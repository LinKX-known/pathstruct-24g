# Datasets

Four datasets, two roles. **PathoVernier is the benchmarking table** (it is what ASPECT
reports on); DHMC is the downstream diagnostic task; PanNuke and MoNuSeg provide real H&E
with nuclei annotations for building and stress-testing the pipeline.

Every number in this repository must come from one of these, never from `pilot synth`.

| dataset | role | size | access | on-disk |
|---|---|---|---|---|
| **PathoVernier** | the table we compare on | 759 QA | HF, gated (`Mikezcy/PathoVernier`) | see below |
| **DHMC-LUAD** | diagnostic task (growth pattern) | 143 WSIs, ~50 GB | request form, email links expire in 4 h | `data/raw/dhmc/` |
| **PanNuke fold1** | real H&E + 5 cell types + 19 tissues | ~280 MB | RationAI / HF parquet | `data/raw/pannuke/` |
| **MoNuSeg** | real H&E, nuclei only, patient IDs | ~100 MB | RationAI / HF parquet | `data/raw/monuseg/` |

## PathoVernier (the benchmark)

1. Open <https://huggingface.co/datasets/Mikezcy/PathoVernier>, accept the terms, download.
2. Inspect what you got before writing any conversion code:

   ```bash
   python scripts/data/import_pathovermier.py --inspect --src data/raw/pathovermier
   ```

   It prints the files, the columns/keys, one sample row and the answer distribution.
3. Convert with an explicit field mapping (the schema is whatever the release ships):

   ```bash
   python scripts/data/import_pathovermier.py --convert \
     --src data/raw/pathovermier --out data/real/pathovermier \
     --question-key question --answer-key answer --image-key image --counts-key counts
   ```

4. Reproduce the published baseline **before** claiming anything. ASPECT's table has
   Qwen3-VL-8B at ~0.37 on PathoVernier; if our harness cannot reproduce that, the
   protocol is wrong and every later comparison is void.

## DHMC-LUAD (diagnostic task)

1. Request from <https://bmirds.github.io/LungCancer/>; the email links expire after 4 h.
2. Download **one zip at a time** (~6.7-16 GB each) and tile immediately; the disk is the
   constraint, not the network.

   ```bash
   python scripts/data/tile_dhmc.py --wsi-dir data/raw/dhmc/wsi_4 \
     --meta data/raw/dhmc/MetaData.csv --out data/real/dhmc --per-slide 20
   ```

3. Labels are the slide's **predominant** pattern. Train ROI-level, evaluate slide-level by
   majority vote, and say so in the paper.

## PanNuke / MoNuSeg

```bash
python scripts/data/import_pannuke.py --parquet data/raw/pannuke/fold1-*.parquet \
  --out data/real/pannuke --val-tissues breast,colon,bile,cervix --min-cells 16
python scripts/data/import_monuseg.py --train-parquet data/raw/monuseg/train-*.parquet \
  --test-parquet data/raw/monuseg/test-*.parquet --out data/real/monuseg
```

Both write `manifest.csv` + `cells/{roi_id}.csv`. If the export already carries masks, you
can skip CellViT; otherwise segment with the official package:

```bash
python scripts/data/run_cellvit.py --config cellvit.yaml --json-dir out/cellvit --cells-out data/real/<name>/cells
```

## Verification (run before every training job)

```bash
python scripts/data/check_dataset.py --manifest data/real/<name>/manifest.csv \
  --cells data/real/<name>/cells
```

It checks: required columns, missing images, missing cell tables, per-class counts, label
imbalance, sample-level split leakage (same `patient_id` on both sides), and cell-count
outliers. Exit code 0 means the cohort is safe to train on.

## Split rules

| dataset | split | why |
|---|---|---|
| MoNuSeg | patient-disjoint | patients share staining and slide batch |
| PanNuke | tissue-held out | patches from one tissue are near-duplicates |
| DHMC | slide-disjoint | ROIs from one slide share the region |
| PathoVernier | use the official split | comparability with ASPECT |

## What is deliberately not here

No dataset is redistributed in this repository. The importers read files you downloaded
yourself and write only derived cell tables, manifests and QA JSONL.
