#!/usr/bin/env bash
set -euo pipefail

cd /home/zihengcai/dip/DIP26
mkdir -p logs
source scripts/ensure_checkpoint_symlinks.sh
ensure_checkpoint_symlinks checkpoints

GPU_IDS="${GPU_IDS:-3}"
DATASETS=(ceilnet_table2 real20 objects postcard wild)
MODELS=(
  "errnet|baseline|checkpoints/errnet/errnet_060_00463920.pt|"
  "errnet_rdnet_no_sup|aligned_no_sup|checkpoints/errnet_rdnet_no_sup/errnet_model_latest.pt|--use_rdnet"
  "errnet_rdnet_sup|aligned_sup|checkpoints/errnet_rdnet_sup/errnet_model_latest.pt|--use_rdnet"
  "errnet_rdnet_full|aligned_full|checkpoints/errnet_rdnet_full/errnet_model_latest.pt|--use_rdnet"
  "errnet_rdnet_no_sup_unaligned|unaligned_no_sup|checkpoints/errnet_rdnet_no_sup_unaligned/errnet_model_latest.pt|--use_rdnet"
  "errnet_rdnet_sup_unaligned|unaligned_sup|checkpoints/errnet_rdnet_sup_unaligned/errnet_model_latest.pt|--use_rdnet"
  "errnet_rdnet_full_unaligned|unaligned_full|checkpoints/errnet_rdnet_full_unaligned/errnet_model_latest.pt|--use_rdnet"
)

OUT="logs/benchmark_results.txt"
echo "RDNet Benchmark $(date)" | tee "$OUT"
echo "GPU: $GPU_IDS" | tee -a "$OUT"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name label ckpt extra <<< "$entry"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] $label ($name): missing $ckpt" | tee -a "$OUT"
    continue
  fi
  for ds in "${DATASETS[@]}"; do
    echo "===== $label | $ds =====" | tee -a "$OUT"
    python test_errnet.py \
      --name "$name" --dataset "$ds" --hyper -r \
      --icnn_path "$ckpt" --gpu_ids "$GPU_IDS" \
      $extra 2>&1 | tee -a "$OUT"
  done
done

echo "===== DONE $(date) =====" | tee -a "$OUT"
