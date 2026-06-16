#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PY_FILES=(
  forward_inference.py
  real_codec_inference.py
  utils/inference_common.py
  utils/bitstream_container.py
  utils/profile_accounting.py
  utils/real_codec_stats.py
  tokenizer/tokenizer_image/models/ar_predictor.py
  tokenizer/tokenizer_image/models/compression_model.py
  tokenizer/tokenizer_image/models/vq_model.py
  tokenizer/tokenizer_image/models/quantizer.py
  tokenizer/tokenizer_image/codec/latent_io.py
  tokenizer/tokenizer_image/codec/real/simple.py
  tokenizer/tokenizer_image/codec/real/causal_pipeline.py
  tokenizer/tokenizer_image/codec/real/causal_ar_loop.py
  tokenizer/tokenizer_image/codec/entropy_coding/codecs/topk_tensor_rans.py
  tokenizer/tokenizer_image/codec/entropy_coding/native/tensor_rans.py
  dataset/build.py
  dataset/openimage.py
  evaluations/utils/evaluator.py
  autoregressive/models/mask_generation.py
  autoregressive/models/gpt.py
)

SH_FILES=(
  test_forward.sh
  test_real.sh
  scripts/tokenizer/train_vq.sh
)

echo "[1/3] Python syntax check"
python3 -m py_compile "${PY_FILES[@]}"

echo "[2/3] Shell syntax check"
for script in "${SH_FILES[@]}"; do
  bash -n "${script}"
done

if [[ "${RUN_FORWARD_SMOKE:-0}" == "1" ]]; then
  echo "[3/3] One-image forward smoke test"
  if [[ -z "${TEST_IMAGE_DIR:-}" ]]; then
    echo "TEST_IMAGE_DIR is required when RUN_FORWARD_SMOKE=1" >&2
    exit 1
  fi
  if [[ -z "${TEST_CKPT_PATH:-}" ]]; then
    echo "TEST_CKPT_PATH is required when RUN_FORWARD_SMOKE=1" >&2
    exit 1
  fi
  smoke_dir=$(mktemp -d /tmp/rdvq_smoke_images.XXXXXX)
  first_image=$(find "${TEST_IMAGE_DIR}" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort | head -n 1)
  if [[ -z "${first_image}" ]]; then
    echo "No image found for forward smoke test" >&2
    exit 1
  fi
  ln -s "${first_image}" "${smoke_dir}/$(basename "${first_image}")"
  TEST_IMAGE_DIR="${smoke_dir}" TEST_MAX_IMAGES=1 TEST_METRICS="bpp,psnr,msssim" bash test_forward.sh
else
  echo "[3/3] Forward smoke skipped; set RUN_FORWARD_SMOKE=1 to enable it"
fi
