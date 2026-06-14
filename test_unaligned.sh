#!/usr/bin/env bash

set -u

RESULT_DIR="${1:-results_ablation}"
GPU_IDS="${GPU_IDS:-0}"

DATASETS=(
  ceilnet_table2
  real20
  objects
  postcard
  wild
  sir2_withgt
)

EXPERIMENTS=(
  "rip_full:1"
  "rip_full_unaligned:1"
  "rip_input_only:1"
  "rip_input_only_unaligned:1"
  "rip_prior_bg:1"
  "rip_prior_bg_unaligned:1"
  "rip_prior_sup:1"
  "rip_prior_sup_unaligned:1"
)

find_checkpoint() {
  local exp="$1"
  local dir="checkpoints/${exp}"

  if [[ -f "${dir}/errnet_latest.pt" ]]; then
    echo "${dir}/errnet_latest.pt"
    return 0
  fi

  local ckpt
  ckpt=$(find "${dir}" -maxdepth 1 -type f -name "errnet_*.pt" 2>/dev/null | sort | tail -n 1)
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"
    return 0
  fi

  return 1
}

for item in "${EXPERIMENTS[@]}"; do
  exp="${item%%:*}"
  use_rpen="${item##*:}"

  ckpt="$(find_checkpoint "${exp}")"
  if [[ -z "${ckpt:-}" ]]; then
    echo "[skip] ${exp}: checkpoint not found under checkpoints/${exp}"
    continue
  fi

  for dataset in "${DATASETS[@]}"; do
    save_subdir="${exp}_${dataset}"
    echo "[run] ${exp} on ${dataset}"

    cmd=(
      python test_errnet.py
      --dataset "${dataset}"
      --name "${exp}"
      --hyper
      --gpu_ids "${GPU_IDS}"
      -r
      --icnn_path "${ckpt}"
      --result_dir "${RESULT_DIR}"
      --save_subdir "${save_subdir}"
    )

    if [[ "${use_rpen}" == "1" ]]; then
      cmd+=(--use_rpen)
    fi

    "${cmd[@]}"
  done
done

echo "[done] metrics summary: ${RESULT_DIR}/metrics.json"
