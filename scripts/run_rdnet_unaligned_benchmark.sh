#!usr/bin/env bash
# Benchmark only the 3 unaligned RDNet models (after symlinks fixed).
set -euo pipefail
cd /home/zihengcai/dip/DIP26
mkdir -p logs
source scripts/ensure_checkpoint_symlinks.sh
ensure_checkpoint_symlinks checkpoints

GPU_IDS="${GPU_IDS:-1}"
OUT="logs/benchmark_unaligned.txt"
DATASETS=(ceilnet_table2 real20 objects postcard wild)
MODELS=(
  "errnet_rdnet_no_sup_unaligned|unaligned_no_sup|checkpoints/errnet_rdnet_no_sup_unaligned/errnet_model_latest.pt"
  "errnet_rdnet_sup_unaligned|unaligned_sup|checkpoints/errnet_rdnet_sup_unaligned/errnet_model_latest.pt"
  "errnet_rdnet_full_unaligned|unaligned_full|checkpoints/errnet_rdnet_full_unaligned/errnet_model_latest.pt"
)

echo "Unaligned Benchmark $(date -u)" | tee "$OUT"
echo "GPU: $GPU_IDS" | tee -a "$OUT"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name label ckpt <<< "$entry"
  for ds in "${DATASETS[@]}"; do
    echo "===== $label | $ds =====" | tee -a "$OUT"
    python test_errnet.py \
      --name "$name" --dataset "$ds" --hyper -r --use_rdnet \
      --icnn_path "$ckpt" --gpu_ids "$GPU_IDS" 2>&1 | tee -a "$OUT"
  done
done

echo "===== DONE $(date -u) =====" | tee -a "$OUT"
