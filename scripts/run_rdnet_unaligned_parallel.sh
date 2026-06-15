#!/usr/bin/env bash
# Launch 3 unaligned fine-tunes in parallel (vgg loss, conservative LR).
set -euo pipefail

cd /home/zihengcai/dip/DIP26
mkdir -p logs
ME="$(whoami)"

# Backup corrupted checkpoints from ctx_vgg run (weights contained NaN).
for d in errnet_rdnet_no_sup_unaligned errnet_rdnet_sup_unaligned errnet_rdnet_full_unaligned; do
  if [[ -d "checkpoints/$d" ]]; then
    bak="checkpoints/${d}_nan_bak"
    if [[ ! -d "$bak" ]]; then
      mv "checkpoints/$d" "$bak"
      echo "[BACKUP] checkpoints/$d -> $bak"
    fi
  fi
done

launch() {
  local name=$1 ckpt=$2 gpu=$3
  shift 3

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] $name: missing $ckpt"
    return 0
  fi
  if pgrep -u "$ME" -f "train_errnet_unaligned.py --name ${name} " > /dev/null 2>&1; then
    echo "[SKIP] $name already running"
    return 0
  fi

  echo "===== $(date) RETRY $name (vgg, lr=5e-5) =====" >> "logs/${name}.log"
  echo "[START] $name on GPU $gpu"
  nohup python "$@" --gpu_ids "$gpu" \
    >> "logs/${name}.log" 2>&1 &
  echo $! > "logs/${name}.pid"
}

# Official ERRNet unaligned fine-tune uses vgg loss; ctx_vgg caused NaN at ~epoch 65.
COMMON=(--hyper --use_rdnet -r --unaligned_loss vgg --lr 5e-5 --save_epoch_freq 5)

launch errnet_rdnet_no_sup_unaligned \
  checkpoints/errnet_rdnet_no_sup/errnet_model_latest.pt 3 \
  train_errnet_unaligned.py --name errnet_rdnet_no_sup_unaligned \
  "${COMMON[@]}" \
  --icnn_path checkpoints/errnet_rdnet_no_sup/errnet_model_latest.pt

launch errnet_rdnet_sup_unaligned \
  checkpoints/errnet_rdnet_sup/errnet_model_latest.pt 1 \
  train_errnet_unaligned.py --name errnet_rdnet_sup_unaligned \
  "${COMMON[@]}" \
  --icnn_path checkpoints/errnet_rdnet_sup/errnet_model_latest.pt

launch errnet_rdnet_full_unaligned \
  checkpoints/errnet_rdnet_full/errnet_model_latest.pt 2 \
  train_errnet_unaligned.py --name errnet_rdnet_full_unaligned \
  "${COMMON[@]}" \
  --icnn_path checkpoints/errnet_rdnet_full/errnet_model_latest.pt

echo "Done. Monitor: bash scripts/watch_rdnet.sh"
