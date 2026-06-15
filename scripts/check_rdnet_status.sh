#!/usr/bin/env bash
# Only show zihengcai RDNet training processes (not rip_* or other users)
echo "=== RDNet training processes (zihengcai) ==="
pgrep -u "$(whoami)" -a python 2>/dev/null | grep -E "errnet_rdnet|run_rdnet_train" || echo "(none)"

echo ""
echo "=== Training launcher ==="
if [[ -f logs/rdnet_train.pid ]]; then
  pid=$(cat logs/rdnet_train.pid)
  if ps -p "$pid" > /dev/null 2>&1; then
    echo "run_rdnet_train.sh PID $pid: RUNNING"
  else
    echo "run_rdnet_train.sh PID $pid: stopped"
  fi
else
  echo "no logs/rdnet_train.pid"
fi

echo ""
echo "=== Latest checkpoints ==="
ls -lt checkpoints/errnet_rdnet_*/errnet_model_latest.pt 2>/dev/null | head -6 || echo "(none yet)"

echo ""
echo "=== Log tails ==="
for f in logs/errnet_rdnet_no_sup.log logs/run_rdnet_train.log; do
  if [[ -f $f ]]; then
    echo "--- $f ---"
    tail -3 "$f"
  fi
done
