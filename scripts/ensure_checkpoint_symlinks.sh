#!/usr/bin/env bash
# Ensure errnet_model_latest.pt points at errnet_latest.pt (training save name).
ensure_checkpoint_symlinks() {
  local root="${1:-checkpoints}"
  for d in "$root"/errnet_rdnet_*; do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/errnet_latest.pt" ]]; then
      ln -sfn errnet_latest.pt "$d/errnet_model_latest.pt"
    fi
  done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  cd "$(dirname "$0")/.."
  ensure_checkpoint_symlinks "${1:-checkpoints}"
  echo "Checkpoint symlinks updated under ${1:-checkpoints}/errnet_rdnet_*"
fi
