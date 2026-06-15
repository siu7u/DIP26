#!/usr/bin/env bash
set -euo pipefail
cd /home/zihengcai/dip/DIP26
source scripts/ensure_checkpoint_symlinks.sh
ensure_checkpoint_symlinks checkpoints
GPU_IDS="${GPU_IDS:-0}" python3 scripts/visualize_rdnet_comparison.py --gpu_ids "$GPU_IDS"
