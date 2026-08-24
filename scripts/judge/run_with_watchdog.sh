#!/usr/bin/env bash
# Run a judge config to completion, restarting the runner if it stalls.
#
# Long OpenRouter runs wedge occasionally: sockets pile up in CLOSE_WAIT, the
# client stops making progress and never times out. The runner skips any phase
# that already has a successful log, so killing and re-running is safe and
# resumes where it stopped.
#
# Usage: scripts/judge/run_with_watchdog.sh <config> [log_root] [stall_minutes]
set -u

CONFIG="$1"
LOG_ROOT="${2:-logs/gpqa-pilot-rd2}"
STALL_MIN="${3:-12}"

pending_count() {
  uv run python scripts/judge/run_judging.py --config "$CONFIG" --dry-run 2>/dev/null \
    | grep -oE '^[0-9]+ phase-runs pending' | grep -oE '^[0-9]+' | head -1
}

# Minutes since this runner last made progress: time since the newest .eval, but
# never counting from before the runner started. Without the floor, a stall
# inherited from a previous runner would kill the new one during its first pass,
# and any run slower than STALL_MIN would be killed on repeat.
progress_age_min() {
  python3 -c "
from pathlib import Path
import time
ts = [p.stat().st_mtime for p in Path('$LOG_ROOT').rglob('*.eval')]
last = max(ts) if ts else 0
print(int((time.time() - max(last, $1)) / 60))
"
}

while :; do
  pending=$(pending_count); [ -z "$pending" ] && pending=0
  echo "[watchdog] $CONFIG: $pending phase-runs pending"
  [ "$pending" -eq 0 ] && break

  started=$(date +%s)
  mkdir -p "$LOG_ROOT"
  # Runner output kept: request errors (rate limits, exhausted credit) only show up here.
  uv run python scripts/judge/run_judging.py --config "$CONFIG" >>"$LOG_ROOT/runner.log" 2>&1 &
  runner=$!
  while kill -0 "$runner" 2>/dev/null; do
    sleep 60
    if [ "$(progress_age_min "$started")" -ge "$STALL_MIN" ]; then
      echo "[watchdog] no progress in ${STALL_MIN}m — restarting runner"
      kill -9 "$runner" 2>/dev/null
      # Only this config's runner: several watchdogs may run side by side.
      pkill -9 -f "run_judging.py --config $CONFIG" 2>/dev/null
      sleep 5
      break
    fi
  done
  wait "$runner" 2>/dev/null
done

echo "[watchdog] $CONFIG complete"
