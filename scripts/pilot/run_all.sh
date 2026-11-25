#!/usr/bin/env bash
# Whole pilot as one sequential chain; every step is idempotent, so rerunning resumes where it stopped.
#   scripts/pilot/run_all.sh          all steps 00..07
#   scripts/pilot/run_all.sh 05       from step 05 (accuracy -> edge -> report)
# Run it detached (PILOT.md section 5): tmux new-session -d -s slm-pilot "cd $PWD && scripts/pilot/run_all.sh 05"
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM="${1:-00}"

steps=(00_build_image 01_prepare_data 02_fetch_and_export_models 03_preflight 04_estimate 05_run_accuracy
       06_run_edge 07_report)
for s in "${steps[@]}"; do
  [[ "${s%%_*}" < "$FROM" ]] && continue
  echo "== $(date -u +%FT%TZ) run_all: $s"
  "$DIR/$s.sh"
done
echo "== $(date -u +%FT%TZ) run_all: finished"
