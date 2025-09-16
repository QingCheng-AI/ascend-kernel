#include "grouped_gemm/src/grouped_gemm.h"
#include "grouped_soft_gemv/src/grouped_soft_gemv.h"
#ifdef ENABLE_GROUPED_QUERY_ATTENTION
#include "ascend-closed/grouped_query_attention/src/grouped_query_attention.h"
#endif
#ifdef ENABLE_INCRE_FLASH_ATTENTION
#include "ascend-closed/incre_flash_attention/src/incre_flash_attention.h"
#endif
#include <pybind11/pybind11.h>

namespace cinfer_ascendc {
using namespace pybind11::literals;
PYBIND11_MODULE(cinfer_ascendc, m) {
    m.def("grouped_soft_gemv", &grouped_soft_gemv::GroupedSoftGemv, "x"_a,
          "weight"_a, "scale"_a, "groupList"_a, "computeType"_a, "output"_a,
          "GROUP GEMV.");
    m.def("grouped_gemm", &grouped_gemm::GroupedGemm, "x"_a, "weight"_a,
          "antiquantScaleOptional"_a, "antiquantOffsetOptional"_a,
          "groupListOptional"_a, "computeType"_a, "output"_a, "GROUP GEMM.");
#ifdef ENABLE_GROUPED_QUERY_ATTENTION
    m.def("grouped_query_attention",
          &grouped_query_attention::GroupedQueryAttention, "query"_a, "key"_a,
          "value"_a, "seqLen"_a, "attentionOut"_a, "batchSize"_a, "layout"_a,
          "scale"_a, "GROUP QUERY ATTENTION.");
#endif
#ifdef ENABLE_INCRE_FLASH_ATTENTION
    m.def("incre_flash_attention", &incre_flash_attention::IncreFlashAttention,
          "query"_a, "key"_a, "value"_a, "seqLen"_a, "maxSeqLen"_a,
          "startIdxEachCore"_a, "attentionOut"_a, "numHeads"_a, "scale"_a,
          "layout"_a, "kvnumHeads"_a, "INCRE FLASH ATTENTION.");
#endif
}
} // namespace cinfer_ascendc