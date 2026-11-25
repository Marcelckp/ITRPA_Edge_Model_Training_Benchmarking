#!/usr/bin/env bash
# Smoke test after 00 and 01: SmolLM2-360M Q4_K_M export, edge preflight, 5 PubMedQA base queries in the edge
# container, all into results/smoke.db (never mixed with results/pilot.db).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DB=results/smoke.db
SMOKE_FILTERS="model=smollm2-360m precision=Q4_K_M strategy=base dataset=pubmedqa"

MODELS=smollm2-360m PRECISIONS=Q4_K_M "$DIR/02_fetch_and_export_models.sh"
FILTERS="$SMOKE_FILTERS" "$DIR/03_preflight.sh" edge

source "$DIR/common.sh"
log_to smoke
require_no_campaign
filter_args=()
for f in $SMOKE_FILTERS; do filter_args+=(--filter "$f"); done
sb edge smoke run --matrix "$MATRIX" --db "$DB" --device edge-container-4c8g --plan edge "${filter_args[@]}" \
  --limit 5 --data-dir "$DATA" --server-log /work/pilot-logs/smoke-server.log
docker run --rm -v "$PROJECT:/work" -w /work/prototype --network none --user "$(id -u):$(id -g)" "$IMAGE" \
  python -c 'import sqlite3; c = sqlite3.connect("results/smoke.db"); print(*c.execute(
  "SELECT query_id, gold_answer, parsed_answer, correct, round(ttft_s, 3), round(e2e_latency_s, 3), "
  "round(peak_mem_mb, 1), mem_kind, error FROM runs ORDER BY run_id"), sep="\n")'
