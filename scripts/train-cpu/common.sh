# Shared settings for CPU fine-tuning on the pilot host (TRAINING-CPU.md). Sourced by the scripts in this folder.
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$(cd "$TRAIN_DIR/../.." && pwd)"
PROJECT="$(cd "$PROTO/.." && pwd)"
TRAIN_IMAGE="${TRAIN_IMAGE:-slmbench-train-cpu:b12532}"
PILOT_IMAGE="${PILOT_IMAGE:-slmbench-pilot:b12532}"
CACHE="${SLM_CACHE:-$HOME/slm-adaptation-research-cache}"
LOGS="$PROJECT/pilot-logs"
TRAIN_MATRIX="${TRAIN_MATRIX:-configs/matrix-train-cpu.yaml}"
TRAIN_DB="${TRAIN_DB:-results/pilot-adapted.db}"
DATA="${DATA:-data}"
# slmbench source mounted into the training containers. Defaults to the project's own src; the benchmark uses a
# staged copy while the pilot chain still runs from prototype/src.
SLM_SRC="${SLM_SRC:-$PROTO/src}"
# 24 physical cores, one thread each: the hyper-thread siblings (24-47) stay idle so CPU time equals core time.
TRAIN_CPUS="${TRAIN_CPUS:-0-23}"
TRAIN_THREADS="${TRAIN_THREADS:-24}"
TRAIN_MEM="${TRAIN_MEM:-64g}"
CAMPAIGN_CONTAINERS=(slm-pilot-accuracy slm-pilot-edge slm-pilot-edge-env)

mkdir -p "$LOGS" "$PROTO/results"

log_to() {
  exec > >(tee -a "$LOGS/$1.log") 2>&1
  _T0=$(date +%s)
  echo "== $(date -u +%FT%TZ) start $1 (image $TRAIN_IMAGE, cpus $TRAIN_CPUS, src $SLM_SRC)"
  trap '_rc=$?; echo "== $(date -u +%FT%TZ) end '"$1"' rc=$_rc elapsed=$(( $(date +%s) - _T0 ))s"' EXIT
}

campaign_running() {
  local c
  for c in "${CAMPAIGN_CONTAINERS[@]}"; do
    docker ps --format '{{.Names}}' | grep -qx "$c" && return 0
  done
  return 1
}

# train_container <name> <cpus> <threads> <cmd...>: no network (models come from the HF cache), host uid.
# slmbench looks for its default configs next to its source tree, hence the /opt/configs mount.
train_container() {
  local name="$1" cpus="$2" threads="$3"
  shift 3
  docker run --rm --init --name "slm-train-$name" --cpuset-cpus "$cpus" --memory "$TRAIN_MEM" \
    --memory-swap "$TRAIN_MEM" --network none --user "$(id -u):$(id -g)" \
    -v "$PROJECT:/work" -v "$CACHE:/cache" -v "$SLM_SRC:/opt/slm-src:ro" -v "$PROTO/configs:/opt/configs:ro" \
    -w /work/prototype \
    -e PYTHONPATH=/opt/slm-src -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e OMP_NUM_THREADS="$threads" -e MKL_NUM_THREADS="$threads" \
    "$TRAIN_IMAGE" "$@"
}
