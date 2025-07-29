#include "grouped_gemm.h"
#include <pybind11/pybind11.h>

namespace grouped_gemm {
using namespace pybind11::literals;
PYBIND11_MODULE(grouped_gemm, m) {
    py::enum_<GroupedGemmType>(m, "GroupedGemmType")
        .value("FP4", GroupedGemmType::FP4)
        .value("FP8", GroupedGemmType::FP8);
    m.def("grouped_gemm", &GroupedGEMM, "x"_a, "weight"_a,
          "antiquantScaleOptional"_a, "antiquantOffsetOptional"_a,
          "groupListOptional"_a, "type"_a, "output"_a, "GROUP GEMM.");
    m.def("grouped_gemv", &GroupedGEMV, "x"_a, "weight"_a, "scale"_a,
          "groupList"_a, "type"_a, "output"_a, "GROUP GEMV.");
}

} // namespace grouped_gemm