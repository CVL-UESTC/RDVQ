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

  tokenizer/tokenizer_image/models/__init__.py
  tokenizer/tokenizer_image/models/autoencoder.py
  tokenizer/tokenizer_image/models/ar_transformer.py
  tokenizer/tokenizer_image/models/quantizer.py
  tokenizer/tokenizer_image/models/vq_model.py

  tokenizer/tokenizer_image/entropy/__init__.py
  tokenizer/tokenizer_image/entropy/models/__init__.py
  tokenizer/tokenizer_image/entropy/models/ar_predictor.py
  tokenizer/tokenizer_image/entropy/models/legacy_model.py
  tokenizer/tokenizer_image/entropy/streams/__init__.py
  tokenizer/tokenizer_image/entropy/streams/coding.py
  tokenizer/tokenizer_image/entropy/streams/packet.py
  tokenizer/tokenizer_image/entropy/symbols/__init__.py
  tokenizer/tokenizer_image/entropy/symbols/probability.py
  tokenizer/tokenizer_image/entropy/symbols/specs.py
  tokenizer/tokenizer_image/entropy/symbols/symbol_mapping.py
  tokenizer/tokenizer_image/entropy/utils/__init__.py
  tokenizer/tokenizer_image/entropy/utils/profiling.py
  tokenizer/tokenizer_image/entropy/native/__init__.py
  tokenizer/tokenizer_image/entropy/native/fast_cdf.py
  tokenizer/tokenizer_image/entropy/native/tensor_rans.py
  tokenizer/tokenizer_image/entropy/codecs/__init__.py
  tokenizer/tokenizer_image/entropy/codecs/base.py
  tokenizer/tokenizer_image/entropy/codecs/compressai_codec.py
  tokenizer/tokenizer_image/entropy/codecs/tensor_rans_codec.py
  tokenizer/tokenizer_image/entropy/codecs/topk_tensor_rans.py
  tokenizer/tokenizer_image/entropy/pipelines/__init__.py
  tokenizer/tokenizer_image/entropy/pipelines/causal_ar_loop.py
  tokenizer/tokenizer_image/entropy/pipelines/causal_tensor.py

  tokenizer/tokenizer_image/compression/__init__.py
  tokenizer/tokenizer_image/compression/real/__init__.py
  tokenizer/tokenizer_image/compression/real/latents.py
  tokenizer/tokenizer_image/compression/real/legacy_pipeline.py
  tokenizer/tokenizer_image/compression/real/profiling.py
  tokenizer/tokenizer_image/compression/real/sampling.py
  tokenizer/tokenizer_image/compression/real/simple_codec.py
  tokenizer/tokenizer_image/compression/real/streaming.py
  tokenizer/tokenizer_image/compression/real/validation.py

  tokenizer/tokenizer_image/training/__init__.py
  tokenizer/tokenizer_image/training/build.py
  tokenizer/tokenizer_image/training/config.py
  tokenizer/tokenizer_image/training/visualization.py
  tokenizer/tokenizer_image/training/train_vq.py
  tokenizer/tokenizer_image/training/train_utils.py
  tokenizer/tokenizer_image/training/losses/__init__.py
  tokenizer/tokenizer_image/training/losses/discriminator_patchgan.py
  tokenizer/tokenizer_image/training/losses/discriminator_stylegan.py
  tokenizer/tokenizer_image/training/losses/lpips.py
  tokenizer/tokenizer_image/training/losses/vq_loss.py
  tokenizer/tokenizer_image/training/optim/__init__.py
  tokenizer/tokenizer_image/training/optim/muon.py
  tokenizer/tokenizer_image/training/optim/scheduler.py

  autoregressive/models/generate_single_stage_real.py
  autoregressive/models/mask_generation.py
  autoregressive/models/gpt.py
  dataset/build.py
  dataset/openimage.py
  evaluations/utils/evaluator.py
)

SH_FILES=(
  test_forward.sh
  test_real.sh
  scripts/smoke_phase0.sh
  scripts/tokenizer/train_vq.sh
)

echo "[1/5] Python syntax check"
python3 -m py_compile "${PY_FILES[@]}"

echo "[2/5] Shell syntax check"
for script in "${SH_FILES[@]}"; do
  bash -n "${script}"
done

echo "[3/5] Compact bitstream container roundtrip"
python3 - <<'PY'
from pathlib import Path
from utils.bitstream_container import EncodedBinRecord, bitstream_size_report, read_rdvq_bin, write_rdvq_bin
path = Path('/tmp/rdvq_release_validate.bin')
write_rdvq_bin(
    path,
    [EncodedBinRecord(0, 'tensor', tensor_top=b'top', tensor_residual=b'res')],
    transfer_slices=28,
    original_shape=(1, 3, 24, 32),
    split_image=False,
)
info = read_rdvq_bin(path)
report = bitstream_size_report(path)
assert info['payload_bytes'] == 6
assert report['container_bytes'] > report['payload_bytes']
print('container roundtrip ok')
PY

if [[ "${RUN_IMPORT_CHECK:-0}" == "1" ]]; then
  echo "[4/5] Import check"
  python3 - <<'PY'
from tokenizer.tokenizer_image.entropy import VQ_AR_Predictor, encode_entropy_packets
from tokenizer.tokenizer_image.models.vq_model import VQ_models
from tokenizer.tokenizer_image.compression.real.simple_codec import SimpleRealCodec
assert callable(encode_entropy_packets)
assert VQ_AR_Predictor is not None
assert VQ_models
assert SimpleRealCodec is not None
print('imports ok')
PY
else
  echo "[4/5] Import check skipped; set RUN_IMPORT_CHECK=1 to enable it"
fi

if [[ "${RUN_TENSOR_RANS:-0}" == "1" ]]; then
  echo "[5/5] Tensor rANS validation"
  python3 scripts/validate_tensor_rans.py --rows "${RANS_ROWS:-128}" --symbols "${RANS_SYMBOLS:-1025}" --compare-compressai
else
  echo "[5/5] Tensor rANS validation skipped; set RUN_TENSOR_RANS=1 to enable it"
fi

if [[ "${RUN_FORWARD_SMOKE:-0}" == "1" ]]; then
  echo "[optional] Forward one-image smoke"
  TEST_MAX_IMAGES="${TEST_MAX_IMAGES:-1}" TEST_METRICS="${TEST_METRICS:-bpp,psnr,msssim}" bash test_forward.sh
fi

if [[ "${RUN_REAL_SMOKE:-0}" == "1" ]]; then
  echo "[optional] Real one-image smoke"
  DISABLE_FID="${DISABLE_FID:-1}" SAVE_IMAGES="${SAVE_IMAGES:-0}" \
  TEST_MAX_IMAGES="${TEST_MAX_IMAGES:-1}" TEST_METRICS="${TEST_METRICS:-bpp,psnr,msssim}" bash test_real.sh
fi

echo "Release validation checks completed."
