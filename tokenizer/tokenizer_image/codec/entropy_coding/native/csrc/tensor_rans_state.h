#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>

#include "rans_byte.h"

namespace py = pybind11;

class IndexedRansDecoderCPU {
public:
    IndexedRansDecoderCPU();
    void set_stream(py::bytes stream);
    torch::Tensor decode_chunk(torch::Tensor cdfs, int precision);

private:
    std::string bytes_;
    uint8_t* ptr_;
    uint8_t* end_ptr_;
    RansState state_;
    bool initialized_;
};
