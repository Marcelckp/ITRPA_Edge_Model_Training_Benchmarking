#!/usr/bin/env bash
# Edge profiling on one device: environment record, pilot + estimate, full plan, energy repeat blocks.
# Usage: scripts/run_edge_matrix.sh <edge-pi5|edge-thinclient> [extra `slmbench run` args...]
set -euo pipefail

DEVICE="${1:?device name, e.g. edge-pi5}"
shift || true
DB="${DB:-results/${DEVICE}.db}"
PILOT_DB="${PILOT_DB:-results/${DEVICE}-pilot.db}"
DATA="${DATA:-data}"
REPEATS="${REPEATS:-3}"
AMBIENT="${AMBIENT_C:-}"

mkdir -p results logs
env_args=(--db "$DB" --device "$DEVICE")
[[ -n "$AMBIENT" ]] && env_args+=(--ambient-c "$AMBIENT")
slmbench env "${env_args[@]}"

echo "== pilot (5 items per config) =="
slmbench run --db "$PILOT_DB" --device "$DEVICE" --plan edge --data-dir "$DATA" --pilot 5 --repeats 1 \
  --server-log "logs/${DEVICE}-pilot-server.log" "$@"
slmbench estimate --db "$PILOT_DB" --device "$DEVICE" --plan edge

read -r -p "Continue with the full edge plan? [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0

echo "== full edge plan, ${REPEATS} energy blocks per config =="
slmbench run --db "$DB" --device "$DEVICE" --plan edge --data-dir "$DATA" --repeats "$REPEATS" \
  --server-log "logs/${DEVICE}-server.log" "$@"
echo "done; copy $DB back to the workstation for merge-db"
