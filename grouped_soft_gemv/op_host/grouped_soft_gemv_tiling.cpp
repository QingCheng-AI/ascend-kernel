#include "grouped_soft_gemv_tiling.h"

using namespace ge;
using namespace AscendC;

template <typename T> static T CeilDiv(T a, T b) {
    if (b == 0)
        return 0;
    return (a + b - 1) / b;
}

namespace optiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t BASEN_FP8 = 128;
constexpr uint32_t BASEN_FP4 = 256;
constexpr uint32_t BASEK_FP8 = 128;
constexpr uint32_t BASEK_FP4 = 64;
constexpr uint32_t MAX_BSZ = 4;
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;
constexpr uint32_t TILING_KEY = 0;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

struct GMVCompileInfo {
    uint32_t aicNum;
    uint32_t aivNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    platform_ascendc::SocVersion socVersion;
};

class GMVTiling {
  public:
    GMVBaseParams tilingData;
    ge::graphStatus PrepareTilingData(const gert::TilingContext *context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);

  protected:
    ge::graphStatus CalGMVTiling(const gert::TilingContext *context);
    ge::graphStatus GetCompileInfo(const gert::TilingContext *context);

  private:
    int32_t m, k, n;
    int32_t ubBaseM_;
    int32_t ubBaseN_;
    int32_t ubBaseK_;
    int32_t groupNum_;
    uint32_t ubXInSize_;
    uint32_t ubWInSize_;
    uint32_t ubOutSize_;
    uint32_t ubSInSize_;
    uint32_t usedCoreNum_;
    uint64_t ubSize_;
    uint64_t workspaceSize_;
    GMVCompileInfo compileInfo;
    ge::DataType weightDtype_ = ge::DT_UNDEFINED;
};

ge::graphStatus GMVTiling::GetCompileInfo(const gert::TilingContext *context) {

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo.aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo.socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                   compileInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,
                                   compileInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,
                                   compileInfo.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,
                                   compileInfo.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,
                                   compileInfo.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,
                                   compileInfo.l2Size);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
GMVTiling::PrepareTilingData(const gert::TilingContext *context) {
    GetCompileInfo(context);
    auto xTensor = context->GetDynamicInputTensor(X_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, xTensor, return ge::GRAPH_FAILED);
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, wTensor, return ge::GRAPH_FAILED);
    gert::Shape xShape = xTensor->GetStorageShape();
    gert::Shape wShape = wTensor->GetStorageShape();
    m = xShape.GetDim(0);
    k = xShape.GetDim(1);
    n = wShape.GetDim(2);
    groupNum_ = wShape.GetDim(0);
    weightDtype_ = wTensor->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMVTiling::CalGMVTiling(const gert::TilingContext *context) {
    if (weightDtype_ == ge::DT_UINT8) {
        ubBaseN_ = BASEN_FP8;
        ubBaseK_ = BASEK_FP8;
        ubWInSize_ = 128 * 128;
        ubXInSize_ = 128 * MAX_BSZ;
        ubOutSize_ = 128 * MAX_BSZ;
        ubSInSize_ = 32;
        return ge::GRAPH_SUCCESS;
    }
    if (weightDtype_ == ge::DT_INT4) {
        ubBaseN_ = BASEN_FP4;
        ubBaseK_ = BASEK_FP4;
        ubWInSize_ = 64 * 256 / 2;
        ubXInSize_ = 64 * MAX_BSZ;
        ubOutSize_ = 256 * MAX_BSZ;
        ubSInSize_ = ubBaseN_ * 2 * ubBaseK_ / 16;
        return ge::GRAPH_SUCCESS;
    }
    OPS_LOG_E(context->GetNodeName(),
              "GMV Tiling: WeightType != UINT8 OR INT4");
    return ge::GRAPH_FAILED;
}

ge::graphStatus GMVTiling::RunFusionKernelTiling(gert::TilingContext *context) {
    auto compileInfoPtr = &compileInfo;
    ubSize_ = compileInfoPtr->ubSize;
    const uint32_t &aivNum = compileInfoPtr->aivNum;
    if (aivNum == 0)
        return ge::GRAPH_FAILED;
    usedCoreNum_ = aivNum;
    workspaceSize_ = 0;
    OPS_ERR_IF(CalGMVTiling(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMV Tiling Failed"),
               return ge::GRAPH_FAILED);
    tilingData.set_coreNum(aivNum);
    tilingData.set_workspaceSize(workspaceSize_);
    tilingData.set_ubBaseK(ubBaseK_);
    tilingData.set_ubBaseN(ubBaseN_);
    tilingData.set_ubBaseM(ubBaseM_);
    tilingData.set_m(m);
    tilingData.set_n(n);
    tilingData.set_k(k);
    tilingData.set_ubXInSize(ubXInSize_);
    tilingData.set_ubWInSize(ubWInSize_);
    tilingData.set_ubOutSize(ubOutSize_);
    tilingData.set_ubSInSize(ubSInSize_);
    tilingData.set_groupNum(groupNum_);
    context->SetTilingKey(TILING_KEY);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    // printf("%d \n", usedCoreNum_);
    context->SetBlockDim(usedCoreNum_);
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus TilingGMV(gert::TilingContext *context) {
    GMVTiling tiling;
    tiling.PrepareTilingData(context);
    return tiling.RunFusionKernelTiling(context);
}

} // namespace optiling