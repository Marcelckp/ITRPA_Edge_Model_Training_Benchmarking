#!/usr/bin/env bash
# Build the pilot image (llama.cpp at the pinned tag + conversion tools + slmbench). Re-running reuses the layer cache.
# Another tag needs its commit too: LLAMA_CPP_TAG=bNNNN LLAMA_CPP_COMMIT=<sha> scripts/pilot/00_build_image.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to 00_build_image

args=(--build-arg "LLAMA_CPP_TAG=$LLAMA_CPP_TAG")
[[ -n "${LLAMA_CPP_COMMIT:-}" ]] && args+=(--build-arg "LLAMA_CPP_COMMIT=$LLAMA_CPP_COMMIT")
docker build -f "$PROTO/docker/Dockerfile.pilot" "${args[@]}" -t "$IMAGE" "$PROTO"
docker image ls "$IMAGE"
docker run --rm "$IMAGE" sh -c \
  '/opt/llama.cpp/build/bin/llama-server --version 2>&1 | tail -2; echo "tag=$LLAMA_CPP_TAG commit=$LLAMA_CPP_COMMIT"'
