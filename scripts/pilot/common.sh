# Shared settings for the container-emulated edge pilot (PILOT.md). Sourced by the numbered scripts.
# Override any of these with environment variables, e.g. SLM_CACHE=/data/slm-cache scripts/pilot/01_prepare_data.sh
set -euo pipefail

PILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$(cd "$PILOT_DIR/../.." && pwd)"
PROJECT="$(cd "$PROTO/.." && pwd)"
LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-b12532}"
IMAGE="${PILOT_IMAGE:-slmbench-pilot:${LLAMA_CPP_TAG}}"
CACHE="${SLM_CACHE:-$HOME/slm-adaptation-research-cache}"
LOGS="$PROJECT/pilot-logs"
MATRIX="configs/matrix-pilot.yaml"
DB="${DB:-results/pilot.db}"
DATA="${DATA:-data}"
REF_CPUS="${REF_CPUS:-0-15}"
EDGE_CPUS="${EDGE_CPUS:-16-19}"
REF_FLAGS=(--cpuset-cpus "$REF_CPUS" --memory 32g)
EDGE_FLAGS=(--cpuset-cpus "$EDGE_CPUS" --cpus 4 --memory 8g --memory-swap 8g)
CAMPAIGN_CONTAINERS=(slm-pilot-accuracy slm-pilot-edge)

mkdir -p "$CACHE/hf" "$CACHE/models" "$CACHE/raw" "$LOGS" "$PROTO/results"

# log_to <name>: append this script's stdout/stderr to pilot-logs/<name>.log with start/end markers.
log_to() {
  exec > >(tee -a "$LOGS/$1.log") 2>&1
  _T0=$(date +%s)
  echo "== $(date -u +%FT%TZ) start $1 (image $IMAGE, cache $CACHE)"
  trap '_rc=$?; echo "== $(date -u +%FT%TZ) end '"$1"' rc=$_rc elapsed=$(( $(date +%s) - _T0 ))s"' EXIT
}

# in_container <net|ref|edge> <name> <cmd...>
#   net  : default bridge network (downloads, conversion); no ports are published
#   local: no network, no resource limits (analysis, report)
#   ref : cpu-reference limits, no network
#   edge: edge-container-4c8g limits, no network
# The harness runs in the same container as llama-server so memory is read from the server PID.
in_container() {
  local tier="$1" name="$2"
  shift 2
  local flags=()
  case "$tier" in
    net) ;;
    local) flags=(--network none) ;;
    ref) flags=("${REF_FLAGS[@]}" --network none) ;;
    edge) flags=("${EDGE_FLAGS[@]}" --network none) ;;
    *) echo "unknown tier $tier" >&2; return 2 ;;
  esac
  local token=()
  # Passed by name only, so the value never appears on a command line or in a log.
  [[ -n "${HF_TOKEN:-}" ]] && token=(-e HF_TOKEN)
  docker run --rm --init --name "slm-pilot-$name" \
    --user "$(id -u):$(id -g)" --ulimit memlock=-1:-1 \
    ${flags[@]+"${flags[@]}"} ${token[@]+"${token[@]}"} \
    -v "$PROJECT:/work" -v "$CACHE:/cache" -w /work/prototype \
    -e PYTHONPATH=/work/prototype/src \
    "$IMAGE" "$@"
}

# sb <tier> <name> <slmbench args...>
sb() {
  local tier="$1" name="$2"
  shift 2
  in_container "$tier" "$name" python -m slmbench "$@"
}

# Refuse to start while a campaign container is running: a second load on the host distorts edge latencies.
require_no_campaign() {
  local c
  for c in "${CAMPAIGN_CONTAINERS[@]}"; do
    if docker ps --format '{{.Names}}' | grep -qx "$c"; then
      echo "container $c is running; wait for it or stop it (docker stop $c) first" >&2
      exit 3
    fi
  done
}

pilot_models() {
  in_container local list-models python -c \
    'import yaml; print(" ".join(m["id"] for m in yaml.safe_load(open("configs/matrix-pilot.yaml"))["models"]))'
}
