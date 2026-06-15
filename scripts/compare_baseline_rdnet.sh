#!/usr/bin/env bash
# Quick benchmark: baseline vs RDNet (aligned). Uses GPU 0 by default.
set -euo pipefail
cd /home/zihengcai/dip/DIP26
mkdir -p logs
GPU="${GPU_IDS:-0}"
OUT="logs/compare_baseline_rdnet.txt"
DATASETS=(ceilnet_table2 real20)

echo "Benchmark compare $(date -u)" | tee "$OUT"
echo "GPU: $GPU" | tee -a "$OUT"

run_one() {
  local label=$1 ds=$2
  shift 2
  echo "" | tee -a "$OUT"
  echo "===== $label | $ds =====" | tee -a "$OUT"
  python test_errnet.py --dataset "$ds" --gpu_ids "$GPU" "$@" 2>&1 | tee -a "$OUT" | tail -3
}

for ds in "${DATASETS[@]}"; do
  run_one "baseline" "$ds" \
    --name errnet --hyper -r \
    --icnn_path checkpoints/errnet/errnet_060_00463920.pt

  run_one "rdnet_no_sup (aligned)" "$ds" \
    --name errnet_rdnet_no_sup --hyper --use_rdnet -r \
    --icnn_path checkpoints/errnet_rdnet_no_sup/errnet_latest.pt

  run_one "rdnet_sup (aligned)" "$ds" \
    --name errnet_rdnet_sup --hyper --use_rdnet -r \
    --icnn_path checkpoints/errnet_rdnet_sup/errnet_latest.pt

  run_one "rdnet_full (aligned)" "$ds" \
    --name errnet_rdnet_full --hyper --use_rdnet -r \
    --icnn_path checkpoints/errnet_rdnet_full/errnet_latest.pt
done

echo "" | tee -a "$OUT"
echo "Done $(date -u)" | tee -a "$OUT"
