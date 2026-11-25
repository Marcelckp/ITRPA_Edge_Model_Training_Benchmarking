#!/usr/bin/env bash
# Workstation adaptation: LR-selection runs (seed 42, full grid), selection, cost-variability seeds, GGUF export.
# Usage: scripts/run_gpu_training.sh [selection|select|variability|export|all]
set -euo pipefail

PHASE="${1:-all}"
DB="${DB:-results/gpu-workstation.db}"
DATA="${DATA:-data}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODELS_DIR="${MODELS_DIR:-$HOME/slm/models}"
mkdir -p results logs

jobs() {  # prints "model strategy dataset seed lr" lines
  slmbench train --list-jobs "$1" | python -c '
import json, sys
for j in json.load(sys.stdin):
    print(j["model"], j["strategy"], j["dataset"], j["seed"], j["learning_rate"] if j["learning_rate"] is not None else "selected")'
}

run_phase() {
  jobs "$1" | while read -r model strategy dataset seed lr; do
    echo "== train $model $strategy $dataset seed=$seed lr=$lr =="
    slmbench train --db "$DB" --data-dir "$DATA" --models-dir "$MODELS_DIR" \
      --model "$model" --strategy "$strategy" --dataset "$dataset" --seed "$seed" --lr "$lr" \
      2>&1 | tee -a "logs/train-${model}-${strategy}-${dataset}-s${seed}.log"
  done
}

export_all() {
  python - "$DB" "$MODELS_DIR" <<'PY' | while read -r model strategy dataset dir; do
import sqlite3, sys
from slmbench import config as C
m = C.load_matrix()
conn = sqlite3.connect(sys.argv[1])
for model, strategy, dataset, lr in conn.execute(
        "SELECT model, strategy, dataset, learning_rate FROM training_runs WHERE selected=1"):
    print(model, strategy, dataset, C.trained_model_dir(m, model, strategy, dataset, lr, 42, sys.argv[2]))
PY
    slmbench export-gguf --db "$DB" --model "$model" --strategy "$strategy" --dataset "$dataset" \
      --trained-dir "$dir" --llama-cpp-dir "$LLAMA_CPP_DIR" --models-dir "$MODELS_DIR"
  done
  for model in $(python -c 'from slmbench import config as C; print(" ".join(m["id"] for m in C.load_matrix()["models"]))'); do
    slmbench export-gguf --db "$DB" --model "$model" --strategy base --llama-cpp-dir "$LLAMA_CPP_DIR" \
      --models-dir "$MODELS_DIR"
  done
}

case "$PHASE" in
  selection) run_phase selection ;;
  select) slmbench train --db "$DB" --select ;;
  variability) run_phase variability ;;
  export) export_all ;;
  all) run_phase selection; slmbench train --db "$DB" --select; run_phase variability; export_all ;;
  *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
