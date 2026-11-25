#!/usr/bin/env bash
# Planned counts, then a timing pilot (first PILOT_N items of every config, separate DB) per tier, then
# `slmbench estimate` projections written to pilot-logs/estimate-<tier>.json. Resumable like `run`.
#   PILOT_N=3 scripts/pilot/04_estimate.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to 04_estimate
require_no_campaign

EST_DB="${EST_DB:-results/pilot-estimate.db}"
N="${PILOT_N:-3}"
ACC_PLANS=(--plan accuracy --plan retention --plan ref-gold)

sb local counts estimate --matrix "$MATRIX" --counts-only | tee "$LOGS/estimate-counts.json"

t0=$(date +%s)
sb ref estimate-ref run --matrix "$MATRIX" --db "$EST_DB" --device cpu-reference "${ACC_PLANS[@]}" \
  --data-dir "$DATA" --pilot "$N" --repeats 1 --server-log /work/pilot-logs/estimate-ref-server.log
echo "reference timing pilot: $(( $(date +%s) - t0 )) s"
t0=$(date +%s)
sb edge estimate-edge run --matrix "$MATRIX" --db "$EST_DB" --device edge-container-4c8g --plan edge \
  --data-dir "$DATA" --pilot "$N" --repeats 1 --server-log /work/pilot-logs/estimate-edge-server.log
echo "edge timing pilot: $(( $(date +%s) - t0 )) s"

sb local estimate estimate --matrix "$MATRIX" --db "$EST_DB" --device cpu-reference "${ACC_PLANS[@]}" \
  | tee "$LOGS/estimate-ref.json"
sb local estimate estimate --matrix "$MATRIX" --db "$EST_DB" --device edge-container-4c8g --plan edge \
  | tee "$LOGS/estimate-edge.json"
