#!/usr/bin/env bash
# Feasibility benchmark for CPU fine-tuning (planning only: nothing is written to a results DB).
# Runs only while the accuracy campaign is the sole workload, on physical cores 20-23 (siblings 44-47 idle),
# away from the accuracy set 0-15 and the edge set 16-19 / 40-43. Stops itself when the accuracy container exits.
#   SLM_SRC=~/slm-train-stage/src scripts/train-cpu/bench.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to train-cpu-bench
BENCH_CPUS="${BENCH_CPUS:-20-23}"
BENCH_THREADS="${BENCH_THREADS:-4}"
OUT="$LOGS/train-cpu-bench.jsonl"
STEPS="${STEPS:-20}"
TIME_LIMIT="${TIME_LIMIT:-420}"

guard() {
  while sleep 5; do
    if ! docker ps --format '{{.Names}}' | grep -qx slm-pilot-accuracy || \
       docker ps --format '{{.Names}}' | grep -qE '^slm-pilot-edge'; then
      echo "== $(date -u +%FT%TZ) accuracy campaign finished or edge started: stopping the benchmark"
      docker ps --format '{{.Names}}' | grep '^slm-train-bench' | xargs -r docker stop >/dev/null
      kill -TERM $$ 2>/dev/null
      return
    fi
  done
}
docker ps --format '{{.Names}}' | grep -qx slm-pilot-accuracy || { echo "accuracy campaign not running; refusing" >&2; exit 3; }
guard &
GUARD=$!
trap 'kill $GUARD 2>/dev/null; echo "== $(date -u +%FT%TZ) end train-cpu-bench"' EXIT

declare -A LR=([fft]=1e-05 [lora]=0.0001 [qlora]=0.0001)
sbt() { train_container "bench-$1" "$BENCH_CPUS" "$BENCH_THREADS" python -m slmbench train --matrix "$TRAIN_MATRIX" \
          --db /tmp/bench.db --data-dir "$DATA" --models-dir /tmp/bench-models --compute cpu "${@:2}"; }

run_combo() {
  local m="$1" s="$2" d="$3"
  shift 3
  echo "== $(date -u +%FT%TZ) bench $m $s $d $*"
  sbt "$s" --model "$m" --strategy "$s" --dataset "$d" --lr "${LR[$s]}" --threads "$BENCH_THREADS" --bench \
    --max-steps "$STEPS" --time-limit-s "$TIME_LIMIT" --max-train-examples 400 --per-device-batch-size 4 \
    --grad-accum 1 --val-limit 5 "$@" 2>&1 | tee -a "$LOGS/train-cpu-bench-detail.log" | grep '^{"bench"' >> "$OUT" \
    || echo "{\"bench\": true, \"model\": \"$m\", \"strategy\": \"$s\", \"dataset\": \"$d\", \"error\": \"see train-cpu-bench-detail.log\"}" >> "$OUT"
}

ONLY="${ONLY:-all}"
if [[ "$ONLY" == all || "$ONLY" == qlora ]]; then
  run_combo smollm2-360m qlora pubmedqa --max-steps 3 --time-limit-s 300
fi
if [[ "$ONLY" == all || "$ONLY" == qlora-full ]]; then
  for d in pubmedqa finqa; do
    run_combo qwen2.5-0.5b qlora "$d"
  done
fi
if [[ "$ONLY" == all || "$ONLY" == main ]]; then
  for d in pubmedqa finqa; do
    for m in smollm2-360m qwen2.5-0.5b; do
      for s in lora fft; do
        run_combo "$m" "$s" "$d"
      done
    done
  done
fi
if [[ "$ONLY" == all || "$ONLY" == scaling ]]; then
  # Thread scaling on the same 4 cores, for the 4 -> 24 core projection.
  BENCH_CPUS=20-21 BENCH_THREADS=2 run_combo qwen2.5-0.5b lora finqa --time-limit-s 300
fi
if [[ "$ONLY" == all || "$ONLY" == tokens ]]; then
  for d in pubmedqa finqa; do
    for m in smollm2-360m qwen2.5-0.5b; do
      echo "== $(date -u +%FT%TZ) token stats $m $d"
      sbt tok --model "$m" --strategy lora --dataset "$d" --lr 0.0001 --token-stats \
        2>>"$LOGS/train-cpu-bench-detail.log" | sed 's/^/{"token_stats": true, "s": /; s/$/}/' >> "$OUT"
    done
  done
fi
