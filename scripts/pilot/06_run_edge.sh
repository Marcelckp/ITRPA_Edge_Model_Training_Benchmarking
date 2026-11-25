#!/usr/bin/env bash
# Edge profiling (edge plan, 100-item subset, 1 block per config) in the edge-container-4c8g container.
# Resumable like 05. Never run alongside 05: the reference container would load the host and distort latencies.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to 06_run_edge
require_no_campaign

sb edge edge-env env --db "$DB" --device edge-container-4c8g --llama-cpp-dir /opt/llama.cpp \
  --note "pilot edge-container-4c8g: docker ${EDGE_FLAGS[*]}, llama-server -t 4; container-emulated, not a Pi" \
  >/dev/null
sb edge edge run --matrix "$MATRIX" --db "$DB" --device edge-container-4c8g --plan edge --data-dir "$DATA" \
  --server-log /work/pilot-logs/edge-server.log "$@"
