#!/usr/bin/env bash
set -euo pipefail

cd /home/zihengcai/dip/DIP26
mkdir -p logs

BASELINE="checkpoints/errnet/errnet_060_00463920.pt"
GPU_IDS="${GPU_IDS:-2}"

run_train() {
  local name=$1
  shift
  echo "===== $(date) START $name =====" | tee -a "logs/${name}.log"
  python "$@" --gpu_ids "$GPU_IDS" 2>&1 | tee -a "logs/${name}.log"
  echo "===== $(date) DONE  $name =====" | tee -a "logs/${name}.log"
}

# aligned stage
run_train errnet_rdnet_no_sup train_errnet.py \
  --name errnet_rdnet_no_sup --hyper --use_rdnet --lambda_rd 0 \
  -r --icnn_path "$BASELINE"

run_train errnet_rdnet_sup train_errnet.py \
  --name errnet_rdnet_sup --hyper --use_rdnet --lambda_rd 1.0 --lambda_rd_tv 0 \
  -r --icnn_path "$BASELINE"

run_train errnet_rdnet_full train_errnet.py \
  --name errnet_rdnet_full --hyper --use_rdnet --lambda_rd 1.0 --lambda_rd_tv 0.00005 \
  -r --icnn_path "$BASELINE"

# unaligned stage
run_train errnet_rdnet_no_sup_unaligned train_errnet_unaligned.py \
  --name errnet_rdnet_no_sup_unaligned --hyper --use_rdnet \
  -r --icnn_path checkpoints/errnet_rdnet_no_sup/errnet_model_latest.pt \
  --unaligned_loss ctx_vgg

run_train errnet_rdnet_sup_unaligned train_errnet_unaligned.py \
  --name errnet_rdnet_sup_unaligned --hyper --use_rdnet \
  -r --icnn_path checkpoints/errnet_rdnet_sup/errnet_model_latest.pt \
  --unaligned_loss ctx_vgg

run_train errnet_rdnet_full_unaligned train_errnet_unaligned.py \
  --name errnet_rdnet_full_unaligned --hyper --use_rdnet \
  -r --icnn_path checkpoints/errnet_rdnet_full/errnet_model_latest.pt \
  --unaligned_loss ctx_vgg

echo "===== $(date) ALL TRAINING COMPLETE =====" | tee -a logs/run_rdnet_train.log

echo "===== $(date) START BENCHMARK =====" | tee -a logs/run_rdnet_train.log
GPU_IDS="$GPU_IDS" bash scripts/run_rdnet_benchmark.sh 2>&1 | tee -a logs/benchmark_results.log
echo "===== $(date) ALL DONE =====" | tee -a logs/run_rdnet_train.log
