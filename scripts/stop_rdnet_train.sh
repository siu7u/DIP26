#!/usr/bin/env bash
# Stop only RDNet training started by run_rdnet_train.sh (zihengcai user)
set -euo pipefail

ME="$(whoami)"
echo "Stopping RDNet training for user $ME (will NOT touch rip_* or other jobs)"

if [[ -f logs/rdnet_train.pid ]]; then
  pid=$(cat logs/rdnet_train.pid)
  if ps -p "$pid" > /dev/null 2>&1; then
    echo "Killing launcher PID $pid"
    kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
  fi
fi

while read -r pid cmd; do
  echo "Killing PID $pid: $cmd"
  kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
done < <(pgrep -u "$ME" -a python 2>/dev/null | grep -E "errnet_rdnet|run_rdnet" || true)

echo "Done."
