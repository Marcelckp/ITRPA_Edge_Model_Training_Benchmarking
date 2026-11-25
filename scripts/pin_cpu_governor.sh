#!/usr/bin/env bash
# Pin every CPU to a frequency governor (default: performance) for measurement; `restore` puts back the saved one.
# Usage: sudo scripts/pin_cpu_governor.sh [performance|restore]
set -euo pipefail

MODE="${1:-performance}"
STATE=/var/tmp/slmbench-governor.saved

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo)" >&2
  exit 1
fi

govs=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor)
if [[ ! -e "${govs[0]}" ]]; then
  echo "no cpufreq scaling_governor on this host" >&2
  exit 1
fi

if [[ "$MODE" == "restore" ]]; then
  [[ -f "$STATE" ]] || { echo "nothing saved at $STATE" >&2; exit 1; }
  saved=$(cat "$STATE")
  for g in "${govs[@]}"; do echo "$saved" > "$g"; done
  echo "restored governor: $saved"
  exit 0
fi

[[ -f "$STATE" ]] || cat "${govs[0]}" > "$STATE"
for g in "${govs[@]}"; do echo "$MODE" > "$g"; done
echo "governor set to $MODE on ${#govs[@]} CPUs (previous saved in $STATE)"
cat "${govs[0]}"
