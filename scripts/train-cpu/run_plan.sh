#!/usr/bin/env bash
# CPU fine-tuning plan (TRAINING-CPU.md section 4), resumable: every phase skips finished work.
#   1 copy results/pilot.db -> results/pilot-adapted.db (base/RAG rows are then skipped by the runs below)
#   2 training: the selection jobs of configs/matrix-train-cpu.yaml, PubMedQA first, one container per run
#   3 LR selection, then merge + GGUF export (F16, Q8_0, Q4_K_M) of the seed-42 selected models
#   4 adapted models through the pilot's own accuracy (05) and edge (06) scripts, with host cgroup loggers
#   5 estimated energy (from CPU time, not metered), analysis and report -> dissertation/generated-pilot-adapted/
# Never runs while a pilot campaign container is up. Options (environment):
#   PHASES="1 2 3 4 5"
#   FINQA_MAX_TRAIN=N    reduced plan: seeded FinQA training subset of N examples (same subset for every run)
#   LRS=from-pubmedqa    reduced plan: FinQA trains only at the LR selected on PubMedQA validation (same model/strategy)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to train-cpu-plan
PHASES="${PHASES:-1 2 3 4 5}"
LRS="${LRS:-all}"
DB_HOST="$PROTO/$TRAIN_DB"
MATRIX_USED="$TRAIN_MATRIX"

if campaign_running; then
  echo "a pilot campaign container is running; refusing to start" >&2
  exit 3
fi

# The FinQA cap goes into a generated matrix copy, so configs/matrix-train-cpu.yaml stays the full plan.
if [[ -n "${FINQA_MAX_TRAIN:-}" ]]; then
  MATRIX_USED="configs/matrix-train-cpu.generated.yaml"
  python3 "$TRAIN_DIR/cap_matrix.py" "$PROTO/$TRAIN_MATRIX" "$PROTO/$MATRIX_USED" "$FINQA_MAX_TRAIN"
fi
echo "plan: matrix $MATRIX_USED, FINQA_MAX_TRAIN=${FINQA_MAX_TRAIN:-all}, LRS=$LRS"

slm() { train_container "$1" "$TRAIN_CPUS" "$TRAIN_THREADS" python -m slmbench "${@:2}"; }

phase() { [[ " $PHASES " == *" $1 "* ]]; }

q() { sqlite3 "$DB_HOST" "$1" 2>/dev/null || true; }

if phase 1 && [[ ! -s "$DB_HOST" ]]; then
  echo "== $(date -u +%FT%TZ) phase 1: copy results/pilot.db -> $TRAIN_DB"
  sqlite3 "$PROTO/results/pilot.db" ".backup '$DB_HOST'"
fi

if phase 2; then
  echo "== $(date -u +%FT%TZ) phase 2: training"
  slm list train --matrix "$MATRIX_USED" --list-jobs selection 2>/dev/null > "$LOGS/train-cpu-jobs.json"
  python3 "$TRAIN_DIR/list_jobs.py" "$LOGS/train-cpu-jobs.json" > "$LOGS/train-cpu-jobs.txt"
  while read -r m s d lr; do
    done_n=$(q "SELECT COUNT(*) FROM training_runs WHERE model='$m' AND strategy='$s' AND dataset='$d' AND seed=42 AND abs(learning_rate-$lr)<1e-12 AND is_mock=0")
    if [[ "${done_n:-0}" != 0 ]]; then
      echo "skip $m $s $d lr=$lr (done)"
      continue
    fi
    if [[ "$LRS" == from-pubmedqa && "$d" == finqa ]]; then
      slm select train --matrix "$MATRIX_USED" --db "$TRAIN_DB" --select > /dev/null < /dev/null
      sel=$(q "SELECT learning_rate FROM training_runs WHERE model='$m' AND strategy='$s' AND dataset='pubmedqa' AND seed=42 AND selected=1 AND is_mock=0")
      if [[ -z "$sel" ]]; then
        echo "no PubMedQA-selected LR for $m $s; skipping $d lr=$lr" >&2
        continue
      fi
      if ! python3 "$TRAIN_DIR/list_jobs.py" --same "$sel" "$lr"; then
        echo "skip $m $s $d lr=$lr (PubMedQA selected lr=$sel)"
        continue
      fi
    fi
    if campaign_running; then
      echo "a pilot campaign container started; stopping before $m $s $d" >&2
      exit 3
    fi
    t0=$(date +%s)
    echo "== $(date -u +%FT%TZ) train $m $s $d lr=$lr"
    slm "train-$m-$s-$d" train --matrix "$MATRIX_USED" --db "$TRAIN_DB" --model "$m" --strategy "$s" --dataset "$d" \
      --lr "$lr" --seed 42 --data-dir "$DATA" --compute cpu --threads "$TRAIN_THREADS" < /dev/null
    echo "trained $m $s $d lr=$lr in $(( $(date +%s) - t0 )) s"
  done < "$LOGS/train-cpu-jobs.txt"
fi

if phase 3; then
  echo "== $(date -u +%FT%TZ) phase 3: LR selection and GGUF export"
  slm select train --matrix "$MATRIX_USED" --db "$TRAIN_DB" --select < /dev/null
  q "SELECT model, strategy, dataset, learning_rate FROM training_runs WHERE selected=1 AND seed=42 AND is_mock=0" \
    | tr '|' ' ' > "$LOGS/train-cpu-selected.txt"
  while read -r m s d lr; do
    dir=$(slm plan train --matrix "$MATRIX_USED" --db "$TRAIN_DB" --model "$m" --strategy "$s" --dataset "$d" \
            --lr "$lr" --seed 42 --dry-run 2>/dev/null < /dev/null | python3 "$TRAIN_DIR/list_jobs.py" --output-dir)
    echo "== $(date -u +%FT%TZ) export $m $s $d (lr=$lr, $dir)"
    slm "export-$m-$s-$d" export-gguf --matrix "$MATRIX_USED" --db "$TRAIN_DB" --model "$m" --strategy "$s" \
      --dataset "$d" --trained-dir "$dir" --llama-cpp-dir /opt/llama.cpp --models-dir /cache/models --skip-existing < /dev/null
  done < "$LOGS/train-cpu-selected.txt"
fi

if phase 4; then
  echo "== $(date -u +%FT%TZ) phase 4: adapted models, accuracy of record then edge"
  logger() {
    taskset -c 39 python3 "$PROTO/scripts/pilot/cgroup_logger.py" --name "$1" --out "$LOGS/$2" --once \
      >> "$LOGS/cgroup-logger-adapted.log" 2>&1 &
  }
  logger slm-pilot-accuracy accuracy-adapted-cgroup.csv
  DB="$TRAIN_DB" "$PROTO/scripts/pilot/05_run_accuracy.sh" --matrix "$MATRIX_USED"
  logger slm-pilot-edge edge-adapted-cgroup.csv
  DB="$TRAIN_DB" "$PROTO/scripts/pilot/06_run_edge.sh" --matrix "$MATRIX_USED"
  wait
fi

if phase 5; then
  echo "== $(date -u +%FT%TZ) phase 5: estimated energy, analysis, report"
  edge_csv=()
  ref_csv=()
  for f in edge-cgroup.csv edge-adapted-cgroup.csv; do
    if [[ -s "$LOGS/$f" ]]; then edge_csv+=(--cgroup-csv "/work/pilot-logs/$f"); fi
  done
  for f in accuracy-cgroup.csv accuracy-adapted-cgroup.csv; do
    if [[ -s "$LOGS/$f" ]]; then ref_csv+=(--cgroup-csv "/work/pilot-logs/$f"); fi
  done
  slm est-edge energy-estimate --db "$TRAIN_DB" --device edge-container-4c8g ${edge_csv[@]+"${edge_csv[@]}"} < /dev/null
  slm est-ref energy-estimate --db "$TRAIN_DB" --device cpu-reference ${ref_csv[@]+"${ref_csv[@]}"} \
    --matrix "$MATRIX_USED" --out /work/dissertation/generated-pilot-adapted/estimated-energy < /dev/null
  slm analyze analyze --matrix "$MATRIX_USED" --db "$TRAIN_DB" --out results/pilot-adapted-analysis < /dev/null
  slm report report --matrix "$MATRIX_USED" --db "$TRAIN_DB" --out /work/dissertation/generated-pilot-adapted < /dev/null
  echo "report written to $PROJECT/dissertation/generated-pilot-adapted"
fi
echo "== $(date -u +%FT%TZ) plan finished"
