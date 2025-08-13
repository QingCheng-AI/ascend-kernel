#include "grouped_soft_gemv.h"
#include <pybind11/pybind11.h>

namespace grouped_soft_gemv {
using namespace pybind11::literals;
PYBIND11_MODULE(cinfer_ascendc, m) {
    m.def("grouped_soft_gemv", &GroupedSoftGemv, "x"_a, "weight"_a, "scale"_a,
          "groupList"_a, "computeType"_a, "output"_a, "GROUP GEMV.");
}
} // namespace grouped_soft_gemv