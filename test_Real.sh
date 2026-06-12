#!/bin/bash

# CPU thread limits keep metric workers and entropy utilities from oversubscribing cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}

## Real codec defaults. The public path is fixed to causal top-k tensor-rANS.
entropy_topk=${TEST_TOPK:-${TOPK:-1024}}

# 1. Runtime selection: checkpoint, GPU, and one dataset.
if [[ -z "${TEST_CKPT_PATH:-}" ]]; then
  echo "❌ TEST_CKPT_PATH is required. Example: TEST_CKPT_PATH=/path/to/best.pt bash test_Real.sh" >&2
  exit 1
fi
ckpt_path=${TEST_CKPT_PATH}
device=${TEST_DEVICE:-0}

# Use TEST_IMAGE_DIR for the image folder. TEST_DATASET is only the evaluation label.
# Example:
#   TEST_IMAGE_DIR=/path/to/images TEST_DATASET=kodak bash test_Real.sh
declare -A DATASET_FID_TESTS=(
  [div2k]="div2k"
  [clic]="clic"
)

dataset_name=$(echo "${TEST_DATASET:-kodak}" | tr '[:upper:]' '[:lower:]' | xargs)
if [[ "${dataset_name}" != "kodak" && "${dataset_name}" != "div2k" && "${dataset_name}" != "clic" ]]; then
  echo "❌ Unknown dataset: ${dataset_name}. Available: kodak, div2k, clic" >&2
  exit 1
fi

image_dir=${TEST_IMAGE_DIR:-}
if [[ -z "${image_dir}" ]]; then
  echo "❌ TEST_IMAGE_DIR is required for ${dataset_name}." >&2
  exit 1
fi

fid_test=${FID_TEST:-${DATASET_FID_TESTS[$dataset_name]}}
if [[ "${DISABLE_FID:-0}" == "1" ]]; then
  fid_test=""
fi

fid_ref_args=()
if [[ -n "${fid_test}" ]]; then
  fid_ref_dir=${FID_REF_DIR:-}
  if [[ -z "${fid_ref_dir}" ]]; then
    case "${dataset_name}" in
      kodak) fid_ref_dir=${KODAK_FID_REF_DIR:-} ;;
      div2k) fid_ref_dir=${DIV2K_FID_REF_DIR:-} ;;
      clic) fid_ref_dir=${CLIC_FID_REF_DIR:-} ;;
    esac
  fi
  if [[ -z "${fid_ref_dir}" && -n "${FID_REF_ROOT:-}" ]]; then
    fid_ref_dir="${FID_REF_ROOT}/${fid_test}_256teles"
  fi
  if [[ -n "${fid_ref_dir}" ]]; then
    fid_ref_args+=(--fid-ref-dir "${fid_ref_dir}")
  fi
fi

# 2. Optional Python arguments. Model, codebook, patch, and codec defaults live in parser.
runtime_args=()
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  runtime_args+=(-o "${OUTPUT_DIR}")
fi
if [[ -n "${TEST_MAX_IMAGES:-}" ]]; then
  runtime_args+=(--max-images "${TEST_MAX_IMAGES}")
fi
if [[ -n "${TEST_METRICS:-}" ]]; then
  runtime_args+=(--metrics "${TEST_METRICS}")
elif [[ "${REAL_BENCHMARK:-0}" == "1" ]]; then
  runtime_args+=(--metrics bpp,psnr,msssim)
fi
if [[ -n "${TEST_TRANSFER_SLICES:-}" ]]; then
  runtime_args+=(--transfer-slices "${TEST_TRANSFER_SLICES}")
fi
runtime_args+=(--entropy-topk "${entropy_topk}")
if [[ "${PROFILE_REAL:-0}" == "1" ]]; then
  runtime_args+=(--profile-real)
fi
if [[ "${TEST_CPU:-0}" == "1" ]]; then
  runtime_args+=(--no-cuda)
fi
if [[ "${SAVE_IMAGES:-1}" == "0" ]]; then
  runtime_args+=(--no-save_img)
fi
if [[ "${SAVE_BINS:-1}" == "0" ]]; then
  runtime_args+=(--no-save_bin)
fi
if [[ "${GENERATE_GT_TELES:-0}" == "1" ]]; then
  runtime_args+=(--generate-gt-teles)
fi
if [[ -n "${TEST_VERBOSE:-}" ]]; then
  runtime_args+=(--verbose)
fi

echo "===================================================================="
echo "🚀 [GPU: ${device}] Starting Real test"
echo "🧪 Dataset: ${dataset_name}"
echo "📁 Image dir: ${image_dir}"
echo "🏷️  FID test: ${fid_test}"
echo "📄 Checkpoint: ${ckpt_path}"
echo "🔢 Entropy topk: ${entropy_topk}"
echo "===================================================================="

CUDA_VISIBLE_DEVICES=$device python Real_Endecode_inference_single_stageAR.py \
  -i "${image_dir}" \
  --fid_test "${fid_test}" \
  --dataset_name "${dataset_name}" \
  --ckpt-path "${ckpt_path}" \
  "${runtime_args[@]}" \
  "${fid_ref_args[@]}"
