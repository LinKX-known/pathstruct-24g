#!/usr/bin/env bash
# Prepare one dataset end to end. Raw downloads are never committed or moved:
# each step reads data/raw/<name> and writes data/real/<name>.
#
#   bash scripts/data/prepare_all.sh pannuke
#   bash scripts/data/prepare_all.sh monuseg
#   bash scripts/data/prepare_all.sh dhmc      # needs CellViT config
#   bash scripts/data/prepare_all.sh pathovermier
#
# Every branch ends with check_dataset.py; a non-zero exit means do not train.

set -euo pipefail

WHICH="${1:-}"
RAW="${RAW:-data/raw}"
OUT="${OUT:-data/real}"

usage() {
  echo "usage: bash scripts/data/prepare_all.sh {pannuke|monuseg|dhmc|pathovermier}"
  exit 2
}

case "$WHICH" in
  pannuke)
    python scripts/data/import_pannuke.py \
      --parquet "$RAW"/pannuke/fold1-*.parquet \
      --out "$OUT/pannuke" \
      --val-tissues breast,colon,bile,cervix --min-cells 16
    python scripts/data/check_dataset.py \
      --manifest "$OUT/pannuke/manifest_train.csv" --cells "$OUT/pannuke/cells" \
      --val-manifest "$OUT/pannuke/manifest_val.csv" --min-cells 16
    ;;

  monuseg)
    python scripts/data/import_monuseg.py \
      --train-parquet "$RAW"/monuseg/train-*.parquet \
      --test-parquet  "$RAW"/monuseg/test-*.parquet \
      --out "$OUT/monuseg"
    python scripts/data/check_dataset.py \
      --manifest "$OUT/monuseg/manifest_train.csv" --cells "$OUT/monuseg/cells" \
      --val-manifest "$OUT/monuseg/manifest_test.csv"
    ;;

  dhmc)
    if [[ ! -f "$RAW/dhmc/MetaData.csv" ]]; then
      echo "missing $RAW/dhmc/MetaData.csv"
      echo "request the dataset at https://bmirds.github.io/LungCancer/ (links expire in 4 h)"
      exit 2
    fi
    for z in "$RAW"/dhmc/wsi_*; do
      [[ -d "$z" ]] || continue
      echo "== tiling $z =="
      python scripts/data/tile_dhmc.py --wsi-dir "$z" --meta "$RAW/dhmc/MetaData.csv" \
        --out "$OUT/dhmc" --per-slide 20
      # delete the extracted WSI directory by hand once tiling looks right;
      # the disk is the constraint, not the download
    done
    echo
    echo "next: segment with CellViT, then rerun this branch's check:"
    echo "  python scripts/data/run_cellvit.py --config cellvit_dhmc.yaml \\"
    echo "      --json-dir out/cellvit/dhmc --cells-out $OUT/dhmc/cells"
    echo "  python scripts/data/check_dataset.py --manifest $OUT/dhmc/manifest.csv --cells $OUT/dhmc/cells"
    ;;

  pathovermier)
    echo "== inspect the release first =="
    python scripts/data/import_pathovermier.py --inspect --src "$RAW/pathovermier"
    echo
    echo "now convert with the field names printed above, e.g.:"
    echo "  python scripts/data/import_pathovermier.py --convert --src $RAW/pathovermier \\"
    echo "      --out $OUT/pathovermier --question-key question --answer-key answer --image-key image"
    ;;

  *)
    usage
    ;;
esac
