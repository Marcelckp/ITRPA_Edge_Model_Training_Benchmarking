#!/usr/bin/env bash
# Download the official dataset files (DESIGN.md 3.2) and prepare data/ + BM25 indices. Skips finished steps.
# Uses configs/matrix.yaml (not the pilot matrix) so data/ is byte-identical to the dissertation's data.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to 01_prepare_data

RAW=/cache/raw
fetch() {  # fetch <url> <file in $CACHE/raw>
  if [[ -s "$CACHE/raw/$2" ]]; then echo "have raw/$2"; return; fi
  in_container net fetch python -c \
    'import os, sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2] + ".part"); os.replace(sys.argv[2] + ".part", sys.argv[2])' \
    "$1" "$RAW/$2"
  echo "fetched raw/$2 ($(stat -c %s "$CACHE/raw/$2") bytes)"
}
fetch https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json ori_pqal.json
fetch https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json test_ground_truth.json
for f in train dev test; do
  fetch "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/$f.json" "finqa_$f.json"
done

done_ds() { [[ -s "$PROTO/$DATA/$1/manifest.json" ]]; }
done_idx() { [[ -s "$PROTO/$DATA/$1/retrieval.jsonl" ]]; }

# pqa_artificial is not in the GitHub repo; it comes from Hugging Face qiaojin/PubMedQA (--hf), the labelled set
# and the official 500-PMID test split from the GitHub files.
done_ds pubmedqa || sb net prepare prepare-data --matrix configs/matrix.yaml --dataset pubmedqa --out "$DATA" \
  --labeled "$RAW/ori_pqal.json" --test-ids "$RAW/test_ground_truth.json" --hf
done_ds finqa || sb net prepare prepare-data --matrix configs/matrix.yaml --dataset finqa --out "$DATA" \
  --train "$RAW/finqa_train.json" --dev "$RAW/finqa_dev.json" --test "$RAW/finqa_test.json"
done_ds arc_easy || sb net prepare prepare-data --matrix configs/matrix.yaml --dataset arc_easy --out "$DATA" --hf
done_idx pubmedqa || sb net index build-index --matrix configs/matrix.yaml --dataset pubmedqa --data-dir "$DATA"
done_idx finqa || sb net index build-index --matrix configs/matrix.yaml --dataset finqa --data-dir "$DATA"

for ds in pubmedqa finqa arc_easy; do
  in_container net manifest python -c \
    'import json, sys; m = json.load(open(sys.argv[1])); print(sys.argv[2], m["counts"], "official_test_split=%s" % m.get("official_test_split", "n/a"))' \
    "$DATA/$ds/manifest.json" "$ds"
done
