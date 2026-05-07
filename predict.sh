#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CHECKPOINT="${CHECKPOINT:-checkpoints/S1GFloods-HA-CQI/S1GFloods-HA-CQI_efficientnet_b0_best.pth}"
STATS_FILE="${STATS_FILE:-datasets/train_set/channel_stats_s1gfloods_train.json}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
THRESHOLD="${THRESHOLD:-0.40}"
ZHENGZHOU_TILES_ROOT="${ZHENGZHOU_TILES_ROOT:-datasets/test_set_Zhengzhou}"
ZHUOZHOU_TILES_ROOT="${ZHUOZHOU_TILES_ROOT:-datasets/test_set_Zhuozhou}"
ZHENGZHOU_OUTPUT_DIR="${ZHENGZHOU_OUTPUT_DIR:-outputs/test_set_Zhengzhou}"
ZHUOZHOU_OUTPUT_DIR="${ZHUOZHOU_OUTPUT_DIR:-outputs/test_set_Zhuozhou}"

for arg in "$@"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    exec python predict.py --help
  fi
done

common_args=(
  --checkpoint "${CHECKPOINT}"
  --stats_file "${STATS_FILE}"
  --gpu_ids "${GPU_IDS}"
  --batch_size "${BATCH_SIZE}"
  --threshold "${THRESHOLD}"
)

python predict.py \
  --tiles-root "${ZHENGZHOU_TILES_ROOT}" \
  --output-dir "${ZHENGZHOU_OUTPUT_DIR}" \
  "${common_args[@]}" \
  "$@"

python predict.py \
  --tiles-root "${ZHUOZHOU_TILES_ROOT}" \
  --output-dir "${ZHUOZHOU_OUTPUT_DIR}" \
  "${common_args[@]}" \
  "$@"
