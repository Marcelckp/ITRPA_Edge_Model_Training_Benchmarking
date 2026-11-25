#!/usr/bin/env bash
# Read-only readiness check inside each tier's container (same limits as its campaign). Exit 1 on any FAIL.
#   scripts/pilot/03_preflight.sh [ref|edge|both]      FILTERS="model=smollm2-360m precision=Q4_K_M" to narrow
# Expected WARNs here: cpufreq not exposed, Pi/RAPL/NVML/meter not applicable, llama-server not running yet.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to 03_preflight
require_no_campaign

which="${1:-both}"
filter_args=()
for f in ${FILTERS:-}; do filter_args+=(--filter "$f"); done
rc=0
if [[ "$which" == ref || "$which" == both ]]; then
  echo "-- cpu-reference (${REF_FLAGS[*]})"
  sb ref preflight-ref preflight --matrix "$MATRIX" --device cpu-reference --db "$DB" ${filter_args[@]+"${filter_args[@]}"} || rc=1
fi
if [[ "$which" == edge || "$which" == both ]]; then
  echo "-- edge-container-4c8g (${EDGE_FLAGS[*]})"
  sb edge preflight-edge preflight --matrix "$MATRIX" --device edge-container-4c8g --db "$DB" ${filter_args[@]+"${filter_args[@]}"} || rc=1
fi
exit $rc
