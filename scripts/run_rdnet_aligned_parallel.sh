#!/usr/bin/env bash
# Launch 3 aligned RDNet ablations in parallel on different GPUs.
# Usage: bash scripts/run_rdnet_aligned_parallel.sh
set -euo pipefail

cd /home/zihengcai/dip/DIP26
mkdir -p logs

BASELINE="checkpoints/errnet/errnet_060_00463920.pt"
ME="$(whoami)"

launch() {
  local name=$1 gpu=$2
  shift 2

  if pgrep -u "$ME" -f "train_errnet.py --name ${name} " > /dev/null 2>&1; then
    echo "[SKIP] $name already running"
    return 0
  fi
  if [[ -f "checkpoints/${name}/errnet_model_latest.pt" ]]; then
    echo "[SKIP] $name checkpoint exists"
    return 0
  fi

  echo "[START] $name on GPU $gpu"
  nohup python "$@" --gpu_ids "$gpu" \
    >> "logs/${name}.log" 2>&1 &
  echo $! > "logs/${name}.pid"
  echo "  PID $(cat logs/${name}.pid)"
}

# GPU assignment (share with rip_* is OK, ~10GB each on 98GB cards)
launch errnet_rdnet_no_sup  3 train_errnet.py \
  --name errnet_rdnet_no_sup --hyper --use_rdnet --lambda_rd 0 \
  -r --icnn_path "$BASELINE"

launch errnet_rdnet_sup     1 train_errnet.py \
  --name errnet_rdnet_sup --hyper --use_rdnet --lambda_rd 1.0 --lambda_rd_tv 0 \
  -r --icnn_path "$BASELINE"

launch errnet_rdnet_full    2 train_errnet.py \
  --name errnet_rdnet_full --hyper --use_rdnet --lambda_rd 1.0 --lambda_rd_tv 0.00005 \
  -r --icnn_path "$BASELINE"

echo ""
echo "Launched. Monitor: bash scripts/watch_rdnet.sh"
