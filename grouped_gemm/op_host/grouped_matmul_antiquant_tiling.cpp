/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * =================================================================================================================
 * CANN Open Software License Agreement Version 1.0
 *
 * This CANN Open Software License Agreement Version 1.0 (hereinafter referred
 * to as this "Agreement") is a legal agreement between you and Huawei, and it
 * governs your use, modification, or distribution of CANN Open Software
 * (hereinafter referred to as "Software"). Please read this Agreement
 * carefully.
 *
 * If you are entering into this Agreement on behalf of a company or other legal
 * entity, you represent that you have the legal authority to bind that entity
 * to this Agreement, in which case "you" will mean the entity you represent.
 *
 * BY DOWNLOADING, INSTALLING, OR USING THE SOFTWARE, YOU AGREE YOU HAVE FULLY
 * UNDERSTOOD AND ACCEPTED THE TERMS CONTAINED HEREIN. IF YOU DO NOT AGREE TO
 * ANY OF THE TERMS OF THIS AGREEMENT, OR IF YOU DO NOT QUALIFY FOR AGREEING TO
 * THIS AGREEMENT, YOU ARE NOT AUTHORIZED TO AND SHALL NOT DOWNLOAD, INSTALL, OR
 * MAKE ANY USE OF THE SOFTWARE.
 *
 * 1. Definition
 *
 * 1.1   Software means the APIs, source code files, binaries, and related
 * documents of Compute Architecture for Neural Networks("CANN") that are
 * licensable by Huawei, and provided and licensed under this Agreement.
 *
 * 1.2   Ascend processors means the chipsets branded with "Ascend" that are
 * manufactured and supplied by Huawei.
 *
 * 2.  Grant of Intellectual Property Rights
 * Subject to the terms and conditions of this Agreement, including your full
 * compliance thereof, Huawei hereby grants you a limited, worldwide,
 * royalty-free, non-transferable, non-sublicensable, and revocable license for
 * you to (i) download, use, modify, integrate, and distribute the Software or
 * its derivative works for the purpose of developing software solely for use
 * with Ascend processors, and (ii) distribute the software developed under (i)
 * solely for use with Ascend processors.
 *
 * 3.  Restrictions
 * 3.1 You are not authorized to, and shall not use, modify, or distribute this
 * Software or its derivative works for any purpose except those expressly
 * permitted by this Agreement. You shall not make any use of the Software or
 * its derivative works to develop or distribute software for use in systems
 * with processors other than Ascend processors.
 *
 * 3.2 You are not authorized to, and shall not remove, obscure, or alter any
 * copyright or other notices in this Software or any part of it.
 *
 * 3.3 Distribution Restrictions.
 * You may distribute the Software or its derivative works in any medium,
 * whether in source or executable forms, provided that you comply with the
 * purpose restriction stipulated in Section 2, provide recipients with a copy
 * of this Agreement, and retain all notices in the Software.
 *
 * 4. Disclaimer of Warranty and Limitation of Liability
 * THE SOFTWARE IS PROVIDED WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED. IN NO EVENT SHALL HUAWEI OR ANY OTHER COPYRIGHT HOLDER BE LIABLE TO
 * YOU FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO ANY DIRECT, OR INDIRECT,
 * SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM YOUR USE OR INABILITY TO USE
 * THE SOFTWARE, IN WHOLE OR IN PART, NO MATTER HOW IT’S CAUSED OR THE LEGAL
 * THEORY IT IS BASED ON, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 *
 * 5. Termination
 * 5.1 This Agreement will continue to apply until terminated by either you or
 * Huawei as described below: a．You may terminate this Agreement by ceasing
 * your use of the Software; b.  Huawei may at any time, terminate this
 * Agreement if: (i) you fail to comply with any term of this Agreement; or (ii)
 * you directly or indirectly initiate any legal proceeding against any
 * individual or entity by alleging that the Software or any part of it
 * infringes your intellectual property rights.
 *
 * 5.2 By termination, all the rights granted to you under this Agreement are
 * terminated, and you shall cease to use and delete this Software or its
 * derivative works immediately.
 *
 * 6. MISCELLANEOUS
 * If the application of any provision of this Agreement to any particular facts
 * or circumstances is held to be invalid or unenforceable by a court of
 * competent jurisdiction, then (a) the validity and enforceability of such
 * provision as applied to any other particular facts or circumstances and the
 * validity of other provisions of this Agreement shall not in any way be
 * affected or impaired thereby and (b) such provision shall be enforced to the
 * maximum extent possible so as to affect the intent of the you and Huawei and
 * reformed without further action by you and Huawei to the extent necessary to
 * make such provision valid and enforceable.
 *
 * END OF THE TERMS AND CONDITIONS
 */
/*!
 * \file grouped_matmul_tiling.cpp
 * \brief
 */
#include "grouped_matmul_antiquant_tiling.h"

using namespace ge;
using namespace AscendC;

template <typename T1, typename T2> static T1 CeilDiv(T1 a, T2 b) {
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

namespace optiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 2;
constexpr uint32_t Y_INDEX = 0;
constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;
constexpr uint32_t L1_PARTA_SIZE = 256 * 1024;
constexpr int32_t BEST_BASEN = 256;
constexpr int32_t BEST_UB_BASEK = 256;
constexpr int32_t BEST_UB_BASEN = 256;
constexpr int32_t MAX_BASEM = 256;
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32; // 32: a block has 32 bytes data
constexpr uint32_t UB_ANTIQUANT_PER_BLOCK_ALIGN = 4 * 1024;
constexpr uint32_t UB_A16W8_BLOCK_NUM_FP16 =
    6; // 2 * sizeof(int8) + 2 * sizeof(half)
constexpr uint32_t UB_A16W8_IO_USED_BLOCK_FP16 = 6;
constexpr uint32_t UB_A16W8_BLOCK_NUM_BF16 = 8; // tmpUb used 2 blks
constexpr uint32_t UB_A16W8_IO_USED_BLOCK_BF16 = 6;
constexpr uint32_t UB_A16W4_BLOCK_NUM_FP16 =
    5; // 2 * sizeof(int4) + 2 * sizeof(half)
constexpr uint32_t UB_A16W4_IO_USED_BLOCK_FP16 = 5;
constexpr uint32_t UB_A16W4_BLOCK_NUM_BF16 = 11;
constexpr uint32_t UB_A16W4_IO_USED_BLOCK_BF16 = 5;
constexpr uint32_t QUEUE_DOUBLE_BUFFER = 2;
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint64_t TILING_KEY = 0;
constexpr uint64_t TILING_KEY_ANTIQUANT_PERFORMANCE = 3;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int32_t SPLIT_M = 0;

static inline uint32_t SixteenAlign(uint32_t a, bool up = false) {
    if (up) {
        a += 15; // 15: 16 bytes up-align
    }
    return a & ~15; // ~15: 16 bytes down-align
}

struct GMMCompileInfo {
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

class GMMTiling {
  public:
    GMMAntiquantTilingData tilingData;
    ge::graphStatus Init(const gert::TilingContext *context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);

  protected:
    ge::graphStatus CalMMTiling(const gert::TilingContext *context,
                                const GMMCompileInfo *compileInfoPtr);
    ge::graphStatus GetCompileInfo(GMMCompileInfo *compileInfoPtr,
                                   const gert::TilingContext *context);
    ge::graphStatus GMMSetMMTiling(const gert::TilingContext *context,
                                   const GMMCompileInfo *compileInfoPtr);
    void GMMSetTilingKey(gert::TilingContext *context) const;
    ge::graphStatus GMMGetAttrs(const gert::TilingContext *context);
    ge::graphStatus GMMSetUbDivideBlk();
    ge::graphStatus GMMSetUbDivideBlkAntiquant();
    ge::graphStatus GMMCalUbSize(const gert::TilingContext *context,
                                 uint32_t ubSize);
    int64_t GMMGetBS(const gert::Shape xShape) const;
    ge::graphStatus PrepareTilingData(const gert::TilingContext *context);
    ge::graphStatus SplitMSingleXSingleWeightSingleY(const gert::Shape xShape,
                                                     const gert::Shape wShape);
    ge::graphStatus DivideUbAndSetWorkspace(gert::TilingContext *context,
                                            const uint32_t &aicNum);
    void DivideUbAndSetWorkspaceAntiquant(size_t *workspaces,
                                          const uint32_t &aicNum,
                                          uint32_t &ubSize);
    ge::graphStatus CalcStepKaKb(const gert::TilingContext *context,
                                 const GMMCompileInfo *compileInfoPtr,
                                 int64_t mInMM, uint32_t &mmStepKa,
                                 uint32_t &mmStepKb);
    void SetTilingDataIsSingleTensor();
    ge::graphStatus GetPerGroupNum(const gert::TilingContext *context);
    ge::graphStatus CheckMKN(const gert::TilingContext *context);

  private:
    int32_t m;
    int32_t k;
    int32_t n;
    int64_t maxM_ = 0;
    int64_t maxN_ = 0;
    int64_t maxK_ = 0;
    int32_t minK_ = INT32_MAX;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    uint64_t ubSize_;
    uint32_t mmDataTypeSize_;
    uint32_t ubDivideBlkNum_;
    uint32_t ubIoBlkNum_;
    uint32_t ubBlockAlign_;
    uint64_t workspacesSize_ = 0; // for antiquant
    uint32_t groupNum_ = 0;
    bool transposeWeight_;
    bool transposeX_;
    bool isSingleWeight_;
    bool isSingleX_;
    bool isSingleY_;
    int32_t groupType_;
    int64_t splitItem_;
    uint32_t groupListType_;
    uint32_t xKDim_;
    uint32_t weightNDim_;
    uint32_t xDimNum_;
    bool antiquantPerformance_ = false;
    uint32_t usedCoreNum_ = 0;

    ge::DataType xDType_ = ge::DT_UNDEFINED;
    ge::DataType mmDType_ = ge::DT_UNDEFINED;
    ge::DataType weightDtype_ = ge::DT_UNDEFINED;
    ge::DataType scaleDtype_ = ge::DT_UNDEFINED;
    ge::DataType yDtype_ = ge::DT_UNDEFINED;
    uint32_t perTokenOrPerGroupSize_ =
        0; // in quant case, it indicates pertoken flag; in antiquant case, it
           // represents pergroup size
    matmul_tiling::CubeFormat wFormat_;
    int32_t nzFactor_; // for weight nz format
    GMMCompileInfo compileInfo;
};

ge::graphStatus GMMTiling::CheckMKN(const gert::TilingContext *context) {
    mmDataTypeSize_ = GetSizeByDataType(mmDType_);
    OPS_ERR_IF(mmDataTypeSize_ == 0,
               OPS_REPORT_VECTOR_INNER_ERR(
                   context->GetNodeName(), "GMM get mm dtype[%s] size is 0.",
                   TypeUtils::DataTypeToAscendString(mmDType_).GetString()),
               return ge::GRAPH_FAILED);
    uint32_t numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize_;
    OPS_ERR_IF(numInOneBlk == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMM numInOneBlk cannot be 0."),
               return ge::GRAPH_FAILED);
    int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
    OPS_ERR_IF(maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
               OPS_REPORT_VECTOR_INNER_ERR(
                   context->GetNodeName(),
                   "32B-aligned m, n or k axis is out of range int32!"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void GMMTiling::SetTilingDataIsSingleTensor() {
    tilingData.gmmBaseParams.set_singleWeight(
        static_cast<uint32_t>(isSingleWeight_));
    tilingData.gmmBaseParams.set_singleX(static_cast<uint32_t>(isSingleX_));
    tilingData.gmmBaseParams.set_singleY(static_cast<uint32_t>(isSingleY_));
}

ge::graphStatus
GMMTiling::PrepareTilingData(const gert::TilingContext *context) {
    // get transpose and groupType
    OPS_ERR_IF(GMMGetAttrs(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMMGetAttrs failed"),
               return ge::GRAPH_FAILED);
    // get the first tensor's shape of weight and x
    auto xTensor =
        context->GetDynamicInputTensor(X_INDEX, 0); // 0: get first tensor
    OPS_LOG_E_IF_NULL(context, xTensor, return ge::GRAPH_FAILED);
    gert::Shape xShape = xTensor->GetStorageShape();
    xDimNum_ = static_cast<uint32_t>(xShape.GetDimNum());
    xKDim_ = xDimNum_ - 1; // 0: when x is transposed, the first dim is k;
                           // -1：otherwise, the last dim is k

    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, wTensor, return ge::GRAPH_FAILED);
    gert::Shape wShape = wTensor->GetOriginShape();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    weightNDim_ = wDimNum - 1; // the last dim is n
    nzFactor_ = 1;             // init
    isSingleWeight_ =
        (context->GetDynamicInputTensor(WEIGHT_INDEX, 1) == nullptr);
    isSingleX_ = (context->GetDynamicInputTensor(X_INDEX, 1) == nullptr);
    isSingleY_ =
        (splitItem_ == 2 ||
         splitItem_ == 3); // 2: when x is multi-tensor, y is single-tensor; 3:
                           // when x is single-tensor, y is single-tensor
    SetTilingDataIsSingleTensor();

    if (groupType_ == SPLIT_M && isSingleX_ && isSingleWeight_ && isSingleY_) {
        return SplitMSingleXSingleWeightSingleY(xShape, wShape);
    }
    OPS_LOG_E(context->GetNodeName(),
              "GMM_tiling: not support groupType_=%d, isSingleWeight_=%d, "
              "isSingleX_=%d, isSingleY_=%d",
              groupType_, isSingleWeight_, isSingleX_, isSingleY_);
    return ge::GRAPH_FAILED;
}

/** @brief split M：single-single-single(s-s-s)
 */
ge::graphStatus
GMMTiling::SplitMSingleXSingleWeightSingleY(const gert::Shape xShape,
                                            const gert::Shape wShape) {
    groupNum_ = static_cast<int32_t>(wShape.GetDim(0));
    m = GMMGetBS(xShape);
    k = xShape.GetDim(xKDim_);
    n = wShape.GetDim(weightNDim_) * static_cast<int64_t>(nzFactor_);
    maxM_ = m;
    maxK_ = k;
    maxN_ = n;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GetCompileInfo(GMMCompileInfo *compileInfoPtr,
                                          const gert::TilingContext *context) {
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

    OPS_ERR_IF(
        (compileInfoPtr->aicNum == 0 || compileInfoPtr->aivNum == 0 ||
         compileInfoPtr->ubSize == 0 || compileInfoPtr->l1Size == 0 ||
         compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 ||
         compileInfoPtr->l0BSize == 0),
        OPS_REPORT_VECTOR_INNER_ERR(
            context->GetNodeName(),
            "platform info is invalid, aicNum=%u, aivNum=%u, ubSize=%lu, "
            "l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
            compileInfoPtr->aicNum, compileInfoPtr->aivNum,
            compileInfoPtr->ubSize, compileInfoPtr->l1Size,
            compileInfoPtr->l0CSize, compileInfoPtr->l0ASize,
            compileInfoPtr->l0BSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::Init(const gert::TilingContext *context) {
    OPS_ERR_IF(PrepareTilingData(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMM PrepareTilingData failed."),
               return ge::GRAPH_FAILED);
    auto compileInfoPtr = &compileInfo;
    GetCompileInfo(compileInfoPtr, context);
    OPS_LOG_E_IF_NULL(
        context, compileInfoPtr,
        return ge::GRAPH_FAILED); // check compileInfoPtr is not null
    mmDType_ = xDType_;
    OPS_ERR_IF(CheckMKN(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMM CheckMKN failed."),
               return ge::GRAPH_FAILED);
    tilingData.gmmBaseParams.set_groupNum(groupNum_);
    tilingData.gmmBaseParams.set_m(m);
    tilingData.gmmBaseParams.set_n(n);
    tilingData.gmmBaseParams.set_k(k);
    tilingData.gmmBaseParams.set_groupType(static_cast<int32_t>(groupType_));
    tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
    tilingData.gmmBaseParams.set_groupListType(groupListType_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GetPerGroupNum(const gert::TilingContext *context) {
    auto antiquantScale =
        context->GetDynamicInputTensor(ANTIQUANT_SCALE_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, antiquantScale, return ge::GRAPH_FAILED);
    auto antiquantScaleShape = antiquantScale->GetStorageShape();
    int64_t dimNum = antiquantScaleShape.GetDimNum();
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, wTensor, return ge::GRAPH_FAILED);
    gert::Shape wShape = wTensor->GetOriginShape();
    size_t wDimNum = wShape.GetDimNum();
    if ((isSingleWeight_ && wDimNum > 2 && dimNum == 3) ||
        (!isSingleWeight_ && dimNum == 2)) { // 2 and 3: dim threshold
        int64_t g = antiquantScaleShape.GetDim(dimNum - 2);
        perTokenOrPerGroupSize_ = g > 1 ? k / g : 0;
        tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
    }
    return ge::GRAPH_SUCCESS;
}

void GMMTiling::DivideUbAndSetWorkspaceAntiquant(size_t *workspaces,
                                                 const uint32_t &aicNum,
                                                 uint32_t &ubSize) {
    for (uint32_t i = 0; i < groupNum_; i++) {
        bool isAllSingleTensor = isSingleX_ && isSingleWeight_ && isSingleY_;
        minK_ = std::min(minK_, k);
        workspacesSize_ += static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
    }
    // when minK * baseN * coreNum * sizeof(float16) > 12M, it goes into
    // antiquantPerformance branch (12M is obtained by test).
    int32_t dimMN =
        CeilDiv(CeilDiv(maxM_, groupNum_), baseM_) * CeilDiv(maxN_, baseN_);
    bool goodCubeUtility =
        dimMN * (xDType_ == ge::DT_BF16 ? 2 : 1) >=
        static_cast<int32_t>(aicNum * 0.4); // 0.4: a factor, in practice.
    antiquantPerformance_ = false;
    uint32_t maxUbBaseN = BEST_UB_BASEN;
    if (antiquantPerformance_) {
        // 2: use 2 pieces of workspace in antiquantPerformance branch
        workspacesSize_ = static_cast<uint64_t>(maxN_) * maxK_ * 2;
    }
    workspaces[0] += workspacesSize_ * mmDataTypeSize_;
}

ge::graphStatus GMMTiling::DivideUbAndSetWorkspace(gert::TilingContext *context,
                                                   const uint32_t &aicNum) {
    size_t *workspaces = context->GetWorkspaceSizes(1); // get second variable
    OPS_LOG_E_IF_NULL(context, workspaces,
                      return ge::GRAPH_FAILED); // check workspaces is not null
    workspaces[0] = SYS_WORKSPACE_SIZE;         // default size
    uint32_t ubSize = static_cast<uint32_t>(ubSize_);
    if ((xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT16)) {
        DivideUbAndSetWorkspaceAntiquant(workspaces, aicNum, ubSize);
        OPS_ERR_IF(GetPerGroupNum(context) != ge::GRAPH_SUCCESS,
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                               "GetPerGroupNum failed."),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(GMMSetUbDivideBlk() != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMMSetUbDivideBlk failed."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(GMMCalUbSize(context, ubSize) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMMCalUbSize failed."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::RunFusionKernelTiling(gert::TilingContext *context) {
    auto compileInfoPtr = &compileInfo;
    OPS_LOG_E_IF_NULL(
        context, compileInfoPtr,
        return ge::GRAPH_FAILED); // check compileInfoPtr is not null

    ubSize_ = compileInfoPtr->ubSize; // get ubSize from compileInfo
    const uint32_t &aicNum =
        compileInfoPtr->aicNum; // get aicNum from compileInfo
    if (aicNum == 0) {          // invaild value
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = aicNum;

    OPS_ERR_IF(CalMMTiling(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMM CalMMTiling failed"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(GMMSetMMTiling(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMM GMMSetMMTiling failed"),
               return ge::GRAPH_FAILED);
    tilingData.gmmBaseParams.set_singleN(0); // 0 is the default value
    OPS_ERR_IF(
        DivideUbAndSetWorkspace(context, aicNum) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                    "GMM DivideUbAndSetWorkspace failed"),
        return ge::GRAPH_FAILED);
    tilingData.gmmBaseParams.set_workspaceSize(workspacesSize_);
    tilingData.mmTilingData.set_usedCoreNum(
        usedCoreNum_); // usedCoreNum is ai_core num
    tilingData.gmmBaseParams.set_coreNum(usedCoreNum_); // ai cube number
    GMMSetTilingKey(context);                           // set tilingkey
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->SetBlockDim(usedCoreNum_); // block dim is the number of aicube
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMCalUbSize(const gert::TilingContext *context,
                                        uint32_t ubSize) {
    OPS_ERR_IF((ubDivideBlkNum_ == 0 || ubBlockAlign_ == 0),
               OPS_REPORT_VECTOR_INNER_ERR(
                   context->GetNodeName(),
                   "ubDivideBlkNum and ubBlockAlign cannot be 0"),
               return ge::GRAPH_FAILED);
    uint32_t ubCalSize =
        ubSize / ubDivideBlkNum_; // divide the UB into ubDivideBlkNum_ pieces
    ubCalSize = ubCalSize / ubBlockAlign_ * ubBlockAlign_; // 16k/8k/4k align.
    uint32_t ubRestBytes =
        ubSize - ubCalSize * ubIoBlkNum_; // compute the rest memory in UB space
    ubRestBytes =
        ubRestBytes / UB_BLOCK_UNIT_SIZE * UB_BLOCK_UNIT_SIZE; // 32B align.
    OPS_ERR_IF(
        (ubCalSize == 0 || ubRestBytes == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                    "ubCalSize and ubRestBytes cannot be 0"),
        return ge::GRAPH_FAILED);
    uint32_t ubBaseN = 0; // init
    uint32_t ubBaseK = 0; // init
    if ((xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT16) &&
        (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
        if (perTokenOrPerGroupSize_ > 0) {
            ubBaseK = perTokenOrPerGroupSize_;
            static const uint32_t MIN_UB_BASEN = 128; // a threshold
            ubBaseN = std::min<uint32_t>(
                BEST_UB_BASEN,
                std::max<uint32_t>(MIN_UB_BASEN,
                                   (ubCalSize / ubBaseK + MIN_UB_BASEN - 1) /
                                       MIN_UB_BASEN * MIN_UB_BASEN));
        } else if (antiquantPerformance_) {
            ubBaseN = BEST_UB_BASEN;
        } else {
            ubBaseN = baseN_;
        }
    } else {
        ubBaseN = baseN_;
    }
    ubBaseK =
        ubCalSize /
        ubBaseN; // ubCalSize is the number of elements, not in bytes unit.
    if (xDType_ == ge::DT_BF16 &&
        (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
        OPS_ERR_IF(
            ubBaseK == 0 || ubBaseN == 0,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                        "ubBaseK or ubBaseN cannot be 0"),
            return ge::GRAPH_FAILED);
    }
    tilingData.gmmBaseParams.set_ubCalSize(ubCalSize);
    tilingData.gmmBaseParams.set_ubBaseK(ubBaseK);
    tilingData.gmmBaseParams.set_ubBaseN(ubBaseN);
    return ge::GRAPH_SUCCESS;
}

int64_t GMMTiling::GMMGetBS(const gert::Shape xShape) const {
    int64_t bs = 0; // init bs
    if (transposeX_) {
        bs = xShape.GetDim(1); // x shape is [k, m] if x is transpose_
    } else {
        if (groupType_ == -1) { // -1: no group case, may exits a situation that
                                // multi dims product equals to bs.
            bs = xShape.GetDim(0); // 0: x first dim
            size_t bsDimNum =
                xDimNum_ >= 1
                    ? xDimNum_ - 1
                    : 0; // 1: x last dim k, the other dimensions are bs
            for (size_t i = 1; i < bsDimNum; i++) {
                bs *= xShape.GetDim(i);
            }
        } else {
            bs = xShape.GetDim(
                0); // in group case，x's shapeis [m,k], 0 is the m axis.
        }
    }
    return bs;
}

void GMMTiling::GMMSetTilingKey(gert::TilingContext *context) const {
    if (antiquantPerformance_) {
        context->SetTilingKey(TILING_KEY_ANTIQUANT_PERFORMANCE);
        context->SetScheduleMode(
            1); // set as batchmod for template using SyncAll
    } else {
        context->SetTilingKey(TILING_KEY);
    }
}

ge::graphStatus GMMTiling::GMMGetAttrs(const gert::TilingContext *context) {
    transposeX_ = false;
    transposeWeight_ = false;
    groupType_ = SPLIT_M;
    splitItem_ = 3; // 0: 默认split_item
    groupListType_ = 0;

    auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, xDesc,
                      return ge::GRAPH_FAILED); // check xDesc is not null
    xDType_ = xDesc->GetDataType();
    auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, w0Desc, return ge::GRAPH_FAILED);
    weightDtype_ = w0Desc->GetDataType();
    auto yDesc = context->GetOutputDesc(Y_INDEX);
    OPS_LOG_E_IF_NULL(context, yDesc, return ge::GRAPH_FAILED);
    yDtype_ = yDesc->GetDataType();
    auto wFormat0 = static_cast<ge::Format>(
        ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
    wFormat_ = wFormat0 == ge::FORMAT_FRACTAL_NZ
                   ? matmul_tiling::CubeFormat::NZ
                   : matmul_tiling::CubeFormat::ND;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkAntiquant() {
    if (xDType_ == ge::DT_FLOAT16 &&
        (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
        if (weightDtype_ == ge::DT_INT8) {
            ubDivideBlkNum_ = UB_A16W8_BLOCK_NUM_FP16;
            ubIoBlkNum_ = UB_A16W8_IO_USED_BLOCK_FP16;
        } else { // int4
            ubDivideBlkNum_ = UB_A16W4_BLOCK_NUM_FP16;
            ubIoBlkNum_ = UB_A16W4_IO_USED_BLOCK_FP16;
        }
        ubBlockAlign_ = UB_ANTIQUANT_PER_BLOCK_ALIGN;
        return ge::GRAPH_SUCCESS;
    }
    if (xDType_ == ge::DT_BF16 &&
        (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
        if (weightDtype_ == ge::DT_INT8) {
            ubDivideBlkNum_ = UB_A16W8_BLOCK_NUM_BF16;
            ubIoBlkNum_ = UB_A16W8_IO_USED_BLOCK_BF16;
        } else {
            ubDivideBlkNum_ = UB_A16W4_BLOCK_NUM_BF16;
            ubIoBlkNum_ = UB_A16W4_IO_USED_BLOCK_BF16;
        }
        ubBlockAlign_ = UB_ANTIQUANT_PER_BLOCK_ALIGN;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlk() {
    ubDivideBlkNum_ = 0; // init ubDivideBlkNum_
    ubIoBlkNum_ = 0;     // init ubIoBlkNum_
    ubBlockAlign_ = 0;   // init ubBlockAlign_
    return GMMSetUbDivideBlkAntiquant();
    return ge::GRAPH_FAILED;
}

static void InitPlatformInfo(const GMMCompileInfo *compileInfoPtr,
                             matmul_tiling::PlatformInfo &platformInfo) {
    platformInfo.socVersion = compileInfoPtr->socVersion;
    platformInfo.l1Size = compileInfoPtr->l1Size;
    platformInfo.l0CSize = compileInfoPtr->l0CSize;
    platformInfo.ubSize = compileInfoPtr->ubSize;
    platformInfo.l0ASize = compileInfoPtr->l0ASize;
    platformInfo.l0BSize = compileInfoPtr->l0BSize;
}

ge::graphStatus GMMTiling::CalcStepKaKb(const gert::TilingContext *context,
                                        const GMMCompileInfo *compileInfoPtr,
                                        int64_t mInMM, uint32_t &mmStepKa,
                                        uint32_t &mmStepKb) {
    uint64_t availableL1Size = compileInfoPtr->l1Size;
    if (compileInfoPtr->socVersion ==
        platform_ascendc::SocVersion::ASCEND310P) {
        availableL1Size = BEST_L1_PARTA + BEST_L1_PARTB;
    }
    OPS_ERR_IF(availableL1Size < L1_PARTA_SIZE,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "availableL1Size is less than 256k"),
               return ge::GRAPH_FAILED);
    // according to double buffer, recompute the params used for data movement
    // from GM to L1
    uint64_t l1ASize =
        baseM_ > baseN_ ? L1_PARTA_SIZE : availableL1Size - L1_PARTA_SIZE;
    uint64_t l1BSize = availableL1Size - l1ASize;
    // 2: double buffer
    mmStepKa = (l1ASize / 2) / (baseM_ * baseK_ * mmDataTypeSize_);
    if (compileInfoPtr->socVersion ==
            platform_ascendc::SocVersion::ASCEND310P &&
        wFormat_ == matmul_tiling::CubeFormat::NZ && mInMM <= baseM_) {
        mmStepKa = std::min<uint32_t>(
            mmStepKa,
            std::max(
                1,
                128 / baseK_)); // 128: nz inner block size. In practice,
                                // baseK_*mmStepKa=128 makes performance better.
    }
    // 2: double buffer
    mmStepKb = (l1BSize / 2) / (baseN_ * baseK_ * mmDataTypeSize_);
    OPS_ERR_IF(mmStepKa == 0 || mmStepKb == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "stepka or stepkb cannot be 0"),
               return ge::GRAPH_FAILED);

    if (mmStepKa > mmStepKb) {
        mmStepKa = mmStepKa / mmStepKb * mmStepKb;
    } else if (mmStepKa < mmStepKb) {
        mmStepKb = mmStepKb / mmStepKa * mmStepKa;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
GMMTiling::GMMSetMMTiling(const gert::TilingContext *context,
                          const GMMCompileInfo *compileInfoPtr) {
    matmul_tiling::DataType matmulDtype =
        static_cast<matmul_tiling::DataType>(mmDType_);
    matmul_tiling::PlatformInfo platformInfo;
    InitPlatformInfo(compileInfoPtr, platformInfo);
    matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
    int64_t mInMM = maxM_; // if msd, m in matmul should mul steps
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmulDtype, false);
    mm.SetBType(matmul_tiling::TPosition::GM, wFormat_, matmulDtype, false);
    mm.SetCType(matmul_tiling::TPosition::GM,
                matmul_tiling::CubeFormat::ND_ALIGN,
                matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBias(false);
    mm.SetOrgShape(mInMM, maxN_, maxK_);
    mm.SetShape(mInMM, baseN_, maxK_);
    mm.SetFixSplit(baseM_, baseN_, baseK_);
    mm.SetBufferSpace(compileInfoPtr->l1Size, compileInfoPtr->l0CSize, ubSize_);
    OPS_ERR_IF(mm.GetTiling(tilingData.mmTilingData) == -1,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "matmul getTiling failed."),
               return ge::GRAPH_FAILED);
    uint32_t mmStepKa = 1;
    uint32_t mmStepKb = 1;
    OPS_ERR_IF(
        CalcStepKaKb(context, compileInfoPtr, mInMM, mmStepKa, mmStepKb) !=
            ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                    "matmul calc stepka or stepkb failed."),
        return ge::GRAPH_FAILED);

    constexpr uint32_t stepM = 1; // 1: stepM set fixed value 1
    constexpr uint32_t stepN = 1; // 1: stepN set fixed value 1
    uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
    uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;
    tilingData.mmTilingData.set_shareMode(0);
    if (compileInfoPtr->socVersion ==
        platform_ascendc::SocVersion::ASCEND310P) {
        tilingData.mmTilingData.set_shareUbSize(0);
        tilingData.mmTilingData.set_transLength(131072); // 131072: 128KB size
    }
    tilingData.mmTilingData.set_dbL0C(1);      // disable double buffer for LOC
    tilingData.mmTilingData.set_baseM(baseM_); // set precomputed baseM
    tilingData.mmTilingData.set_baseN(baseN_); // set precomputed baseN
    tilingData.mmTilingData.set_baseK(baseK_); // set precomputed baseK
    tilingData.mmTilingData.set_stepKa(mmStepKa);   // set precomputed mmStepKa
    tilingData.mmTilingData.set_depthA1(mmDepthA1); // set precomputed mmDepthA1
    tilingData.mmTilingData.set_stepKb(mmStepKb);   // set precomputed mmStepKb
    tilingData.mmTilingData.set_depthB1(mmDepthB1); // set precomputed mmDepthB1
    tilingData.mmTilingData.set_stepM(stepM);       // set precomputed stepM
    tilingData.mmTilingData.set_stepN(stepN);       // set precomputed stepN
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::CalMMTiling(const gert::TilingContext *context,
                                       const GMMCompileInfo *compileInfoPtr) {

    baseN_ = BEST_BASEN;
    // according to the double buffer enabled L0B, compute baseK
    baseK_ = (compileInfoPtr->l0BSize / DOUBLE_BUFFER_L0A_L0B) /
             (baseN_ * mmDataTypeSize_);
    baseK_ = SixteenAlign(baseK_);
    OPS_ERR_IF(baseK_ == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "baseK_ cannot be 0."),
               return ge::GRAPH_FAILED);
    // according to the double buffer enabled L0A/L0C, compute baseM(cube)
    uint32_t maxBaseM = compileInfoPtr->l0CSize / (baseN_ * FP32_DATATYPE_SIZE);
    baseM_ =
        std::min<uint32_t>((compileInfoPtr->l0ASize / DOUBLE_BUFFER_L0A_L0B) /
                               (baseK_ * mmDataTypeSize_),
                           maxBaseM);

    baseM_ = baseM_ > maxM_ ? SixteenAlign(maxM_, true) : SixteenAlign(baseM_);

    if (baseM_ > MAX_BASEM) {
        baseM_ = MAX_BASEM;
    }
    OPS_ERR_IF(baseM_ == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "baseM_ cannot be 0."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingGMM(gert::TilingContext *context) {
    auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, xDesc,
                      return ge::GRAPH_FAILED); // check xDesc is not null
    ge::DataType xDType_ = xDesc->GetDataType();
    auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OPS_LOG_E_IF_NULL(context, w0Desc, return ge::GRAPH_FAILED);
    ge::DataType weightDtype_ = w0Desc->GetDataType();
    GMMTiling tiling;
    OPS_ERR_IF(tiling.Init(context) != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "GMM tiling init failed"),
               return ge::GRAPH_FAILED);
    return tiling.RunFusionKernelTiling(context);
}

} // namespace optiling
