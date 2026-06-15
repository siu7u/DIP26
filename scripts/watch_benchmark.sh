#!/usr/bin/env bash
# Monitor run_rdnet_benchmark.sh progress (safe grep/pgrep handling)
set -euo pipefail
cd "$(dirname "$0")/.."

count_matches() {
  local n
  n=$(grep -c "$1" "$2" 2>/dev/null) || true
  echo "${n:-0}"
}

count_procs() {
  local n
  n=$(pgrep -u "$(whoami)" -c -f "$1" 2>/dev/null) || true
  echo "${n:-0}"
}

INTERVAL="${1:-30}"

while true; do
  metrics=$(count_matches '^LMSE:' logs/benchmark_results.txt)
  sections=$(count_matches '^===== ' logs/benchmark_results.txt)
  done_flag=$(grep -c '^===== DONE' logs/benchmark_results.txt 2>/dev/null) || done_flag=0
  done_flag=${done_flag:-0}
  procs=$(count_procs 'run_rdnet_benchmark|test_errnet.py')
  current=$(grep '^===== ' logs/benchmark_results.txt 2>/dev/null | tail -1 || echo "(none)")

  clear
  echo "Benchmark Monitor  $(date -u '+%Y-%m-%d %H:%M UTC')  refresh=${INTERVAL}s"
  echo "============================================================"
  echo "  Completed metrics: $metrics / 35"
  echo "  Section headers:   $sections"
  echo "  Running processes: $procs"
  echo "  Current:           $current"
  echo ""
  if [[ "$done_flag" -ge 1 ]]; then
    echo "  STATUS: DONE"
    echo ""
    echo "Run: python3 scripts/parse_benchmark_results.py"
    break
  fi
  if [[ "$metrics" -ge 35 && "$procs" -eq 0 ]]; then
    echo "  STATUS: likely complete (no process)"
    break
  fi
  echo "  STATUS: running..."
  echo ""
  echo "Press Ctrl+C to exit (does not stop benchmark)"
  sleep "$INTERVAL"
done
