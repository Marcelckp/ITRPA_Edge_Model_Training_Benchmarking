#!/usr/bin/env bash
# End-to-end smoke of the real CPU training path on tiny data (nothing persists outside the container's /tmp):
# train (64 PubMedQA examples = 4 optimiser steps, 3 val items) -> train --select -> merge + GGUF export -> energy-estimate.
# Uses cores 20-23 only, so it may run beside the accuracy campaign but never beside the edge campaign.
#   SLM_SRC=~/slm-train-stage/src scripts/train-cpu/smoke.sh [lora|fft|qlora]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to train-cpu-smoke
STRATEGY="${1:-lora}"
if docker ps --format '{{.Names}}' | grep -qE '^slm-pilot-edge'; then
  echo "edge campaign running; refusing" >&2
  exit 3
fi
LR=0.0001
[[ "$STRATEGY" == fft ]] && LR=1e-05
train_container "smoke-$STRATEGY" 20-23 4 sh -c '
set -e
sed "s/pubmedqa: {n_train: 10000, class_balanced: true}/pubmedqa: {n_train: 10000, class_balanced: true, max_train_examples: 64}/" \
  configs/matrix-train-cpu.yaml > /tmp/m.yaml
grep -q "max_train_examples: 64" /tmp/m.yaml
A="--matrix /tmp/m.yaml --db /tmp/smoke.db"
python -m slmbench train $A --model smollm2-360m --strategy '"$STRATEGY"' --dataset pubmedqa --lr '"$LR"' --seed 42 \
  --data-dir data --models-dir /tmp/models --compute cpu --threads 4 --val-limit 3
python -m slmbench train $A --select
DIR=$(python -m slmbench train $A --model smollm2-360m --strategy '"$STRATEGY"' --dataset pubmedqa --lr '"$LR"' --seed 42 \
  --models-dir /tmp/models --dry-run | python -c "import json,sys; print(json.load(sys.stdin)[\"output_dir\"])")
python -m slmbench export-gguf $A --model smollm2-360m --strategy '"$STRATEGY"' --dataset pubmedqa --trained-dir "$DIR" \
  --llama-cpp-dir /opt/llama.cpp --models-dir /tmp/models --precision Q4_K_M
python -m slmbench energy-estimate --db /tmp/smoke.db --out /tmp/est
sqlite3_dump() { python - <<EOF
import sqlite3; c = sqlite3.connect("/tmp/smoke.db")
for t in ("training_runs", "cpu_usage", "energy_estimates", "artifacts"):
    print(t, [dict(zip([d[0] for d in c.execute(f"SELECT * FROM {t}").description], r)) for r in c.execute(f"SELECT * FROM {t}")])
print({k: v for k, v in c.execute("SELECT key, value FROM environment")})
EOF
}
sqlite3_dump
head -8 /tmp/est/estimated_energy.md
sha256sum /tmp/models/smollm2-360m/*/*.gguf
'
