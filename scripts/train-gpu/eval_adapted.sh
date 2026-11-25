#!/usr/bin/env bash
# Evaluate GPU-trained adapted models in the pilot containers (TRAINING-GPU.md section 7): same tiers, limits,
# llama-server settings and steps as scripts/pilot/03, 05, 06 and 07, but with configs/matrix-train-gpu.yaml and
# its own DB, so pilot results and CPU-trained models are never mixed in.
#   scripts/train-gpu/eval_adapted.sh init <training DB>...   copy results/pilot.db, merge the notebook DBs, select LRs
#   scripts/train-gpu/eval_adapted.sh preflight|accuracy|edge|report|all [extra `slmbench run` args]
# Only the adapted configs run: base and RAG rows come from the pilot.db copy (same config ids).
source "$(dirname "${BASH_SOURCE[0]}")/../pilot/common.sh"
MATRIX="configs/matrix-train-gpu.yaml"
DB="${GPU_DB:-results/pilot-adapted-gpu.db}"
ADAPTED=(--filter strategy=fft,lora,qlora)
step="${1:-all}"
shift || true
log_to "gpu-eval-$step"

# The pilot's require_no_campaign only knows its own two containers; CPU training uses slm-train-*.
no_other_campaign() {
  local running
  running="$(docker ps --format '{{.Names}}' | grep -E '^slm-(pilot|train)-' || true)"
  if [[ -n "$running" ]]; then
    echo "running: $running; wait until no pilot or training container is up" >&2
    exit 3
  fi
}

init_db() {
  [[ $# -gt 0 ]] || { echo "init needs the training DBs from the notebook sessions" >&2; exit 2; }
  if [[ -e "$PROTO/$DB" ]]; then
    echo "$DB exists; merging into it (merge-db skips rows it already has)"
  else
    in_container local gpu-copy-db python -c \
      'import sqlite3, sys; src = sqlite3.connect(sys.argv[1]); dst = sqlite3.connect(sys.argv[2]); src.backup(dst)' \
      results/pilot.db "$DB"
  fi
  sb local gpu-merge merge-db --into "$DB" "$@"
  sb local gpu-select train --matrix "$MATRIX" --db "$DB" --select >/dev/null
  # Artifacts point at training runs by train_id; merge-db renumbers training_runs, so check the links survived.
  in_container local gpu-check-db python -c '
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
bad = c.execute("SELECT a.path FROM artifacts a LEFT JOIN training_runs t ON t.train_id = a.train_id "
                "WHERE a.train_id IS NOT NULL AND (t.train_id IS NULL OR t.model IS NOT a.model OR "
                "t.strategy IS NOT a.strategy OR t.dataset IS NOT a.dataset OR t.selected IS NOT 1)").fetchall()
n = c.execute("SELECT COUNT(*) FROM training_runs WHERE is_mock = 0").fetchone()[0]
print(f"{n} training runs, {len(bad)} artifacts not linked to their selected run")
sys.exit(1 if bad else 0)' "$DB"
}

preflight() {
  sb ref gpu-preflight-ref preflight --matrix "$MATRIX" --device cpu-reference --db "$DB" "${ADAPTED[@]}"
  sb edge gpu-preflight-edge preflight --matrix "$MATRIX" --device edge-container-4c8g --db "$DB" "${ADAPTED[@]}"
}

accuracy() {
  sb ref gpu-accuracy-env env --db "$DB" --device cpu-reference --llama-cpp-dir /opt/llama.cpp \
    --note "gpu-adapted cpu-reference: docker ${REF_FLAGS[*]}, llama-server -t 16" >/dev/null
  sb ref gpu-accuracy run --matrix "$MATRIX" --db "$DB" --device cpu-reference \
    --plan accuracy --plan retention --data-dir "$DATA" "${ADAPTED[@]}" \
    --server-log /work/pilot-logs/gpu-accuracy-server.log "$@"
}

edge() {
  sb edge gpu-edge-env env --db "$DB" --device edge-container-4c8g --llama-cpp-dir /opt/llama.cpp \
    --note "gpu-adapted edge-container-4c8g: docker ${EDGE_FLAGS[*]}, llama-server -t 4; not a Pi" >/dev/null
  sb edge gpu-edge run --matrix "$MATRIX" --db "$DB" --device edge-container-4c8g --plan edge \
    --data-dir "$DATA" "${ADAPTED[@]}" --server-log /work/pilot-logs/gpu-edge-server.log "$@"
}

report() {
  sb local gpu-analyze analyze --matrix "$MATRIX" --db "$DB" --out results/pilot-gpu-analysis
  sb local gpu-report report --matrix "$MATRIX" --db "$DB" --out /work/dissertation/generated-pilot-gpu
  echo "report written to $PROJECT/dissertation/generated-pilot-gpu"
}

no_other_campaign
case "$step" in
  init) init_db "$@" ;;
  preflight) preflight ;;
  accuracy) accuracy "$@" ;;
  edge) edge "$@" ;;
  report) report ;;
  all) preflight && accuracy "$@" && edge "$@" && report ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
