#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans_byte.h"
#include "tensor_rans_state.h"

namespace py = pybind11;

namespace {

void check_cpu_int32_1d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.dtype() == torch::kInt32, name, " must have dtype int32");
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cpu_int32_2d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.dtype() == torch::kInt32, name, " must have dtype int32");
    TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void validate_precision(int precision) {
    TORCH_CHECK(precision > 0 && precision <= 16, "precision must be in [1, 16], got ", precision);
}

void validate_cdf_row(const int32_t* row, int64_t width, int precision) {
    const int32_t total = 1 << precision;
    TORCH_CHECK(width >= 2, "CDF width must be at least 2");
    TORCH_CHECK(row[0] == 0, "CDF row must start at 0");
    TORCH_CHECK(row[width - 1] == total, "CDF row must end at 2**precision");
}

}  // namespace

py::bytes encode_indexed_cdf_cpu(torch::Tensor symbols, torch::Tensor cdfs, int precision) {
    validate_precision(precision);
    symbols = symbols.contiguous();
    cdfs = cdfs.contiguous();
    check_cpu_int32_1d(symbols, "symbols");
    check_cpu_int32_2d(cdfs, "cdfs");

    const int64_t n = symbols.numel();
    const int64_t rows = cdfs.size(0);
    const int64_t width = cdfs.size(1);
    TORCH_CHECK(rows == n, "cdfs rows must match symbols length");

    if (n == 0) {
        return py::bytes();
    }

    const auto* sym_ptr = symbols.data_ptr<int32_t>();
    const auto* cdf_ptr = cdfs.data_ptr<int32_t>();

    // Conservative upper bound for byte-aligned rANS output. If it ever proves
    // insufficient, fail loudly and let the Python fallback use CompressAI.
    const size_t max_size = std::max<size_t>(1024, static_cast<size_t>(n) * 16 + 1024);
    std::vector<uint8_t> buffer(max_size);
    uint8_t* begin = buffer.data();
    uint8_t* ptr = begin + buffer.size();

    RansState state;
    RansEncInit(&state);

    for (int64_t i = n - 1; i >= 0; --i) {
        const int32_t s = sym_ptr[i];
        TORCH_CHECK(s >= 0 && s + 1 < width, "symbol out of CDF range at row ", i);
        const int32_t* row = cdf_ptr + i * width;
        validate_cdf_row(row, width, precision);
        const uint32_t start = static_cast<uint32_t>(row[s]);
        const uint32_t end = static_cast<uint32_t>(row[s + 1]);
        TORCH_CHECK(end > start, "CDF frequency must be positive at row ", i);
        if (ptr - begin < 16) {
            throw std::runtime_error("tensor rANS output buffer exhausted");
        }
        RansEncPut(&state, &ptr, start, end - start, static_cast<uint32_t>(precision));
    }

    if (ptr - begin < 4) {
        throw std::runtime_error("tensor rANS output buffer exhausted during flush");
    }
    RansEncFlush(&state, &ptr);

    const char* out = reinterpret_cast<const char*>(ptr);
    const size_t out_size = static_cast<size_t>((begin + buffer.size()) - ptr);
    return py::bytes(out, out_size);
}

torch::Tensor decode_indexed_cdf_cpu(py::bytes stream, torch::Tensor cdfs, int precision) {
    validate_precision(precision);
    cdfs = cdfs.contiguous();
    check_cpu_int32_2d(cdfs, "cdfs");

    std::string bytes = stream;
    TORCH_CHECK(bytes.size() >= 4 || cdfs.size(0) == 0, "rANS stream is too short");

    const int64_t n = cdfs.size(0);
    const int64_t width = cdfs.size(1);
    auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    if (n == 0) {
        return out;
    }

    const auto* cdf_ptr = cdfs.data_ptr<int32_t>();
    auto* out_ptr = out.data_ptr<int32_t>();

    uint8_t* ptr = reinterpret_cast<uint8_t*>(bytes.data());
    uint8_t* end_ptr = ptr + bytes.size();

    RansState state;
    RansDecInit(&state, &ptr);

    for (int64_t i = 0; i < n; ++i) {
        const int32_t* row = cdf_ptr + i * width;
        validate_cdf_row(row, width, precision);
        const uint32_t value = RansDecGet(&state, static_cast<uint32_t>(precision));
        const int32_t* upper = std::upper_bound(row, row + width, static_cast<int32_t>(value));
        int64_t s = static_cast<int64_t>(upper - row) - 1;
        TORCH_CHECK(s >= 0 && s + 1 < width, "decoded symbol out of range at row ", i);
        const uint32_t start = static_cast<uint32_t>(row[s]);
        const uint32_t next = static_cast<uint32_t>(row[s + 1]);
        TORCH_CHECK(next > start, "CDF frequency must be positive at row ", i);
        out_ptr[i] = static_cast<int32_t>(s);
        RansDecAdvance(&state, &ptr, start, next - start, static_cast<uint32_t>(precision));
        TORCH_CHECK(ptr <= end_ptr, "rANS decoder consumed past end of stream");
    }

    return out;
}



IndexedRansDecoderCPU::IndexedRansDecoderCPU()
    : ptr_(nullptr), end_ptr_(nullptr), initialized_(false) {}

void IndexedRansDecoderCPU::set_stream(py::bytes stream) {
    bytes_ = static_cast<std::string>(stream);
    TORCH_CHECK(bytes_.size() >= 4, "rANS stream is too short");
    ptr_ = reinterpret_cast<uint8_t*>(bytes_.data());
    end_ptr_ = ptr_ + bytes_.size();
    RansDecInit(&state_, &ptr_);
    initialized_ = true;
}

torch::Tensor IndexedRansDecoderCPU::decode_chunk(torch::Tensor cdfs, int precision) {
    TORCH_CHECK(initialized_, "decoder stream is not initialized");
    validate_precision(precision);
    cdfs = cdfs.contiguous();
    check_cpu_int32_2d(cdfs, "cdfs");

    const int64_t n = cdfs.size(0);
    const int64_t width = cdfs.size(1);
    auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    if (n == 0) {
        return out;
    }

    const auto* cdf_ptr = cdfs.data_ptr<int32_t>();
    auto* out_ptr = out.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i) {
        const int32_t* row = cdf_ptr + i * width;
        validate_cdf_row(row, width, precision);
        const uint32_t value = RansDecGet(&state_, static_cast<uint32_t>(precision));
        const int32_t* upper = std::upper_bound(row, row + width, static_cast<int32_t>(value));
        int64_t s = static_cast<int64_t>(upper - row) - 1;
        TORCH_CHECK(s >= 0 && s + 1 < width, "decoded symbol out of range at chunk row ", i);
        const uint32_t start = static_cast<uint32_t>(row[s]);
        const uint32_t next = static_cast<uint32_t>(row[s + 1]);
        TORCH_CHECK(next > start, "CDF frequency must be positive at chunk row ", i);
        out_ptr[i] = static_cast<int32_t>(s);
        RansDecAdvance(&state_, &ptr_, start, next - start, static_cast<uint32_t>(precision));
        TORCH_CHECK(ptr_ <= end_ptr_, "rANS decoder consumed past end of stream");
    }
    return out;
}
