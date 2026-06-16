#!/bin/bash

# CPU thread limits keep metric workers and dataloading from oversubscribing cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}

# 1. Runtime selection: checkpoint, GPU, and one dataset.
if [[ -z "${TEST_CKPT_PATH:-}" ]]; then
    echo "❌ TEST_CKPT_PATH is required. Example: TEST_CKPT_PATH=/path/to/checkpoint bash test_forward.sh" >&2
    exit 1
fi
ckpt_path=${TEST_CKPT_PATH}
device=${TEST_DEVICE:-0}

# Use TEST_IMAGE_DIR for the image folder. TEST_DATASET is only the evaluation label.
# Example:
#   TEST_IMAGE_DIR=/path/to/images TEST_DATASET=kodak bash test_forward.sh
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

mkdir -p inference_logs

print_log_summary() {
    local log_path="$1"
    local image_count
    local metrics_json

    metrics_json=$(grep 'Metrics JSON:' "${log_path}" | tail -n 1 | sed 's/^Metrics JSON: //')
    if [[ -n "${metrics_json}" && -f "${metrics_json}" ]]; then
        image_count=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("image_count", ""))' "${metrics_json}" 2>/dev/null || true)
    fi
    if [[ -z "${image_count}" ]]; then
        image_count=$(grep -c '^image:' "${log_path}" || true)
    fi
    echo "✅ Test task completed"
    echo "📄 Log file: ${log_path}"
    echo "🖼️  Images processed: ${image_count}"
    echo "-------------------- Test Results --------------------"
    if ! grep -E '^(_fid:|Save path:|Metrics JSON:|average_|CD bpp:)' "${log_path}"; then
        echo "Average metrics not found, printing tail of log for inspection:"
        tail -n "${LOG_TAIL_LINES:-80}" "${log_path}" || true
    fi
    echo "--------------------------------------------------"
}

# 2. Optional runtime arguments. Model/patch defaults live in forward_inference.py.
runtime_args=()
if [[ -n "${TEST_MAX_IMAGES:-}" ]]; then
    runtime_args+=(--max-images "${TEST_MAX_IMAGES}")
fi
if [[ -n "${TEST_METRICS:-}" ]]; then
    runtime_args+=(--metrics "${TEST_METRICS}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    runtime_args+=(-o "${OUTPUT_DIR}")
fi
if [[ "${TEST_CPU:-0}" == "1" ]]; then
    runtime_args+=(--no-cuda)
fi
if [[ "${SAVE_IMAGES:-1}" == "0" ]]; then
    runtime_args+=(--no-save_img)
fi
if [[ "${GENERATE_GT_TELES:-0}" == "1" ]]; then
    runtime_args+=(--generate-gt-teles)
fi
if [[ -n "${TEST_VERBOSE:-}" ]]; then
    runtime_args+=(--verbose)
fi

echo "===================================================================="
echo "🚀 [GPU: ${device}] Starting estimated-rate test"
echo "🧪 Dataset: ${dataset_name}"
echo "📁 Image dir: ${image_dir}"
echo "🏷️  FID test: ${fid_test}"
echo "📄 Checkpoint: ${ckpt_path}"
echo "===================================================================="

log_path="inference_logs/log_${dataset_name}.txt"
CUDA_VISIBLE_DEVICES=$device python forward_inference.py \
    -i "${image_dir}" \
    --fid_test "${fid_test}" \
    --dataset_name "${dataset_name}" \
    --ckpt-path "${ckpt_path}" \
    "${runtime_args[@]}" \
    "${fid_ref_args[@]}" 2>&1 | tee "${log_path}"
status=${PIPESTATUS[0]}

if (( status != 0 )); then
    echo "❌ Test task failed (Python exit=${status})"
    echo "📄 Log file: ${log_path}"
    echo "-------------------- Log Tail --------------------"
    tail -n "${LOG_TAIL_LINES:-80}" "${log_path}" || true
    echo "--------------------------------------------------"
    exit "${status}"
fi

print_log_summary "${log_path}"
echo "🎉 Test task completed successfully!"
