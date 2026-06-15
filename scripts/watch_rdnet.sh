#!/usr/bin/env bash
# Real-time RDNet training monitor (only zihengcai RDNet jobs)
set -euo pipefail

cd "$(dirname "$0")/.."
INTERVAL="${1:-10}"
ME="$(whoami)"
GPU_ID="${GPU_IDS:-3}"

strip_ansi() { sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tr '\r' '\n'; }

current_experiment() {
  pgrep -u "$ME" -a python 2>/dev/null | grep errnet_rdnet | head -1 | sed -n 's/.*--name \([^ ]*\).*/\1/p' || echo "(none)"
}

current_log() {
  local exp
  exp=$(current_experiment)
  if [[ -f "logs/${exp}.log" ]]; then
    echo "logs/${exp}.log"
  elif [[ -f logs/errnet_rdnet_no_sup.log ]]; then
    echo logs/errnet_rdnet_no_sup.log
  else
    ls -t logs/errnet_rdnet_*.log 2>/dev/null | head -1 || echo ""
  fi
}

show_dashboard() {
  local logfile exp epoch progress ckpt_line gpu_line launcher
  exp=$(current_experiment)
  logfile=$(current_log)

  clear
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║           RDNet Training Monitor  (refresh: ${INTERVAL}s)           ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""

  echo "── Current experiment ──"
  echo "  name: $exp"
  if [[ -n "$logfile" && -f "$logfile" ]]; then
    echo "  log:  $logfile ($(du -h "$logfile" | cut -f1))"
  fi
  echo ""

  echo "── Process ──"
  pgrep -u "$ME" -a python 2>/dev/null | grep -E "errnet_rdnet" | sed 's/^/  /' || echo "  (no training process)"
  if [[ -f logs/rdnet_train.pid ]]; then
    launcher=$(cat logs/rdnet_train.pid)
    if ps -p "$launcher" > /dev/null 2>&1; then
      echo "  launcher PID $launcher: RUNNING"
    else
      echo "  launcher PID $launcher: stopped"
    fi
  fi
  echo ""

  echo "── GPU ${GPU_ID} ──"
  if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --id="$GPU_ID" --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null \
      | awk -F', ' '{printf "  util: %s | mem: %s / %s | temp: %s\n", $1, $2, $3, $4}' \
      || echo "  (nvidia-smi unavailable)"
  else
    echo "  (nvidia-smi not found)"
  fi
  echo ""

  echo "── Training progress ──"
  if [[ -n "$logfile" && -f "$logfile" ]]; then
    epoch=$(grep -E '^Epoch:' "$logfile" | tail -1 | awk '{print $2}' || true)
    [[ -n "${epoch:-}" ]] && echo "  epoch: $epoch / 60 (aligned) or / 80 (unaligned)"
    progress=$(tail -c 8000 "$logfile" | strip_ansi | grep -E '\[[=>].*\].*[0-9]+/[0-9]+' | tail -1 | sed 's/^[[:space:]]*//')
    [[ -n "${progress:-}" ]] && echo "  step:  $progress"
    grep -E 'saving the (latest )?model|START |DONE ' "$logfile" | tail -3 | sed 's/^/  /'
  else
    echo "  (no log yet)"
  fi
  echo ""

  echo "── Checkpoints ──"
  ls -lt checkpoints/errnet_rdnet_*/errnet_model_latest.pt 2>/dev/null | head -4 | awk '{print "  "$6" "$7" "$8"  "$9}' || echo "  (none yet)"
  echo ""

  echo "── Recent log (last 8 lines) ──"
  if [[ -n "$logfile" && -f "$logfile" ]]; then
    tail -c 4000 "$logfile" | strip_ansi | grep -v '^[[:space:]]*$' | tail -8 | sed 's/^/  /'
  fi
  echo ""
  echo "Press Ctrl+C to exit | Follow log: tail -f $logfile"
}

follow_log() {
  local logfile
  logfile=$(current_log)
  if [[ -z "$logfile" || ! -f "$logfile" ]]; then
    echo "No log file found. Is training running?"
    exit 1
  fi
  echo "Following $logfile (Ctrl+C to stop)..."
  tail -f "$logfile" | strip_ansi
}

case "${1:-}" in
  -f|--follow)
    follow_log
    ;;
  -h|--help)
    echo "Usage:"
    echo "  bash scripts/watch_rdnet.sh [interval_sec]   # refresh dashboard (default 10s)"
    echo "  bash scripts/watch_rdnet.sh --follow         # stream log in real time"
    echo "  GPU_IDS=3 bash scripts/watch_rdnet.sh        # monitor specific GPU"
    ;;
  *)
    while true; do
      show_dashboard
      sleep "$INTERVAL"
    done
    ;;
esac
