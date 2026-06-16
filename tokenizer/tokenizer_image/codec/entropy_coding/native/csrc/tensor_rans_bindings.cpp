#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "tensor_rans_state.h"

pybind11::bytes encode_indexed_cdf_cpu(torch::Tensor symbols, torch::Tensor cdfs, int precision);
torch::Tensor decode_indexed_cdf_cpu(pybind11::bytes stream, torch::Tensor cdfs, int precision);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_indexed_cdf", &encode_indexed_cdf_cpu, "Encode int32 symbols with row-wise int32 CDFs");
    m.def("decode_indexed_cdf", &decode_indexed_cdf_cpu, "Decode int32 symbols with row-wise int32 CDFs");
    pybind11::class_<IndexedRansDecoderCPU>(m, "IndexedRansDecoder")
        .def(pybind11::init<>())
        .def("set_stream", &IndexedRansDecoderCPU::set_stream)
        .def("decode_chunk", &IndexedRansDecoderCPU::decode_chunk);
}
