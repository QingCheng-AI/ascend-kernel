#include "grouped_gemm.h"
#include <pybind11/pybind11.h>

namespace grouped_gemm {
using namespace pybind11::literals;
PYBIND11_MODULE(cinfer_ascendc, m) {
    m.def("grouped_gemm", &GroupedGemm, "x"_a, "weight"_a,
          "antiquantScaleOptional"_a, "antiquantOffsetOptional"_a,
          "groupListOptional"_a, "computeType"_a, "output"_a, "GROUP GEMM.");
}

} // namespace grouped_gemm