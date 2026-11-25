#!/usr/bin/env bash
# Download the base models from Hugging Face and export GGUF with the prototype's own path
# (convert_hf_to_gguf.py --outtype f16, then llama-quantize). Files already on disk are kept, hashed and
# recorded in the DB's artifacts table.
#   MODELS="smollm2-360m" PRECISIONS="Q4_K_M" DB=results/smoke.db scripts/pilot/02_fetch_and_export_models.sh
# Llama 3.2 is gated: enable it in configs/matrix-pilot.yaml and export HF_TOKEN first (PILOT.md).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to 02_fetch_and_export_models
require_no_campaign

models="${MODELS:-$(pilot_models)}"
prec_args=()
for p in ${PRECISIONS:-F16 Q8_0 Q4_K_M}; do prec_args+=(--precision "$p"); done

for m in $models; do
  if [[ "$m" == llama* && -z "${HF_TOKEN:-}" ]]; then
    echo "skip $m: gated model and HF_TOKEN is not set" >&2
    continue
  fi
  t0=$(date +%s)
  sb net "export-$m" export-gguf --matrix "$MATRIX" --db "$DB" --model "$m" --strategy base \
    --llama-cpp-dir /opt/llama.cpp --models-dir /cache/models --skip-existing "${prec_args[@]}"
  echo "exported $m in $(( $(date +%s) - t0 )) s"
done
ls -l "$CACHE"/models/*/base/ 2>/dev/null || true
