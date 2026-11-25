#!/usr/bin/env bash
# Accuracy of record (accuracy, retention, ref-gold plans) in the cpu-reference container. Resumable: rerun
# after any interruption and finished (config, query) pairs are skipped. Never run alongside 06.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to 05_run_accuracy
require_no_campaign

sb ref accuracy-env env --db "$DB" --device cpu-reference --llama-cpp-dir /opt/llama.cpp \
  --note "pilot cpu-reference: docker ${REF_FLAGS[*]}, llama-server -t 16" >/dev/null
sb ref accuracy run --matrix "$MATRIX" --db "$DB" --device cpu-reference \
  --plan accuracy --plan retention --plan ref-gold --data-dir "$DATA" \
  --server-log /work/pilot-logs/accuracy-server.log "$@"
