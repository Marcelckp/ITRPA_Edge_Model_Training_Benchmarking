#!/usr/bin/env bash
# Start the CPU training plan only after the pilot chain has finished, so training never overlaps the edge campaign.
# Waits until tmux session slm-pilot is gone, no pilot container runs, run_all.log ends with "run_all: finished"
# and dissertation/generated-pilot/manifest.json exists. Then installs the staged slmbench source (the pilot chain
# ran from prototype/src, which therefore could not change while it ran) and runs run_plan.sh.
#   tmux new-session -d -s slm-train "cd ~/slm-adaptation-research/prototype && scripts/train-cpu/wait_then_run.sh"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_to train-cpu-waiter
STAGE_SRC="${STAGE_SRC:-$HOME/slm-train-stage/src}"
POLL_S="${POLL_S:-60}"

pilot_done() {
  ! tmux has-session -t slm-pilot 2>/dev/null \
    && ! campaign_running \
    && tail -n 1 "$LOGS/run_all.log" 2>/dev/null | grep -q "run_all: finished" \
    && [[ -s "$PROJECT/dissertation/generated-pilot/manifest.json" ]]
}

echo "waiting for the pilot chain (tmux slm-pilot) to finish"
until pilot_done; do
  if ! tmux has-session -t slm-pilot 2>/dev/null && ! campaign_running \
     && ! tail -n 1 "$LOGS/run_all.log" 2>/dev/null | grep -q "run_all: finished"; then
    echo "== $(date -u +%FT%TZ) tmux slm-pilot is gone but run_all.log has no 'finished' line: the chain failed;" \
         "resume it (PILOT.md section 5); still waiting"
    POLL_S=600
  fi
  sleep "$POLL_S"
done
echo "== $(date -u +%FT%TZ) pilot chain finished"

if [[ -d "$STAGE_SRC/slmbench" ]]; then
  backup="$HOME/slm-train-stage/src-pilot-$(date -u +%Y%m%dT%H%M%SZ)"
  cp -a "$PROTO/src" "$backup"
  rsync -a --delete --exclude='__pycache__/' --exclude='*.egg-info/' "$STAGE_SRC/" "$PROTO/src/"
  echo "installed staged src from $STAGE_SRC (previous src kept in $backup)"
fi
exec "$TRAIN_DIR/run_plan.sh"
