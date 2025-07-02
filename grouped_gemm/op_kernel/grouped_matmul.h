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
 * \file grouped_matmul.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_H
#define ASCENDC_GROUPED_MATMUL_H

#include "grouped_matmul_utils.h"

namespace GROUPED_MATMUL {

constexpr uint32_t thresholdBlockNum =
    8; // 8 is obtained by tests, indicating the threshold of basic block
       // numbers in both directions when assigning data blocks to cube cores
       // when using diagnal strategy
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
constexpr uint32_t thresholdDimM = 1; // not needs any special strategies
#else
constexpr uint32_t thresholdDimM =
    5; // 5 is obtained by tests, indicating the threshold for distinguishing
       // strategies for large/small shapes
#endif

/*@brief store variables for core split configuration
 */
struct MNConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint64_t wBaseOffset = 0;
    uint64_t nAxisBaseOffset = 0;
    uint64_t mAxisBaseOffset = 0;
    uint64_t xBaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t wOutOffset = 0;
    uint64_t workSpaceOffset = 0;
};

/** @brief GroupMatmul operator Class
 */
template <typename ComputeType> class GMMProcess {
  protected:
    using B = typename ComputeType::B;
    ComputeType &computeOp; // inernal computation operator
    const GMMBaseParams *__restrict gmmBaseParams;
    const TCubeTiling *__restrict mmTilingData;

    uint32_t blockIdx;
    uint32_t coreIdx;
    uint32_t groupNum;
    int32_t preOffset;
    GM_ADDR groupListPtr;
    GlobalTensor<int64_t> groupListGm;

  public:
    /** @brief constructor */
    __aicore__ inline GMMProcess(ComputeType &computeOp_)
        : computeOp(computeOp_) {}

    __aicore__ inline void Init(const GMMBaseParams *__restrict gmmBaseParamsIn,
                                const TCubeTiling *__restrict mmTilingDataIn,
                                GM_ADDR groupList, GM_ADDR tiling);

    __aicore__ inline void Process();

  protected:
    __aicore__ inline void SetMNConfig(MNConfig &mnConfig);

    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig);

    __aicore__ inline void MNBlockIdxCompute(MNConfig &mnConfig,
                                             const uint32_t curBlock,
                                             const uint32_t count,
                                             const uint32_t thresholdM_dimN);
};

template <typename ComputeType>
__aicore__ inline void
GMMProcess<ComputeType>::Init(const GMMBaseParams *__restrict gmmBaseParamsIn,
                              const TCubeTiling *__restrict mmTilingDataIn,
                              GM_ADDR groupList, GM_ADDR tiling) {
    blockIdx = GetBlockIdx();
    coreIdx = blockIdx;
    int64_t coreRation = GetTaskRation();
    if (coreRation > 1) {
        coreIdx /= coreRation;
    }
    gmmBaseParams = gmmBaseParamsIn;
    mmTilingData = mmTilingDataIn;
    groupNum = gmmBaseParams->groupNum;
    groupListPtr = groupList;
    if (groupListPtr != nullptr) {
        groupListGm.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }
}

template <typename ComputeType>
__aicore__ inline void
GMMProcess<ComputeType>::SetMNConfig(MNConfig &mnConfig) {
    mnConfig.baseM = mmTilingData->baseM;
    mnConfig.baseN = mmTilingData->baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.m = 0; // 默认值为0，在每次group的每次迭代会将其更新为splitValue
    mnConfig.k = gmmBaseParams->k;
    mnConfig.n = gmmBaseParams->n;
#if defined(GMM_QUANT_BF16) || defined(GMM_QUANT_FLOAT16) || defined(GMM_FLOAT)
    if (gmmBaseParams->singleN > 0) { // not sequential write
        mnConfig.singleN = gmmBaseParams->singleN;
    }
#endif
}

template <typename ComputeType>
__aicore__ inline void
GMMProcess<ComputeType>::UpdateMnConfig(MNConfig &mnConfig) {
    if constexpr (B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset +=
            AlignUp<16>(mnConfig.k) *
            AlignUp<16>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::MNBlockIdxCompute(
    MNConfig &mnConfig, const uint32_t curBlock, const uint32_t count,
    const uint32_t thresholdM_dimN) {
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM =
            relativeBlock >= AlignDown(mnConfig.blockDimM * mnConfig.blockDimN,
                                       thresholdM_dimN)
                ? mnConfig.blockDimM % thresholdBlockNum
                : thresholdBlockNum;
        uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
        uint32_t curThresholdN =
            relativeBlock % thresholdM_dimN >=
                    AlignDown(curThresholdM * mnConfig.blockDimN,
                              curThresholdM_thresholdN)
                ? mnConfig.blockDimN % thresholdBlockNum
                : thresholdBlockNum;

        uint32_t localRelativeBlock =
            relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM +
                        relativeBlock / thresholdM_dimN * thresholdBlockNum;
        mnConfig.nIdx =
            (localRelativeBlock +
             localRelativeBlock /
                 LeastCommonMultiple(curThresholdM, curThresholdN)) %
                curThresholdN +
            relativeBlock % thresholdM_dimN / curThresholdM_thresholdN *
                thresholdBlockNum;
    }
}

/** @brief intenal computation class
 */
template <class mmType, bool sync = false> class GMMCompute {
  public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = DTYPE_WEIGHT;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline GMMCompute(typename mmType::MT &mm_) : mm(mm_) {}

    __aicore__ inline void
    Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
         GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
         GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
         const GMMBaseParams *__restrict gmmBaseParams,
         const TCubeTiling *__restrict mmTilingData, TPipe *tPipe);

    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig &mnConfig,
                                     uint32_t coreIdx);

    __aicore__ inline void VectorCompute(MNConfig &mnConfig) {}

    __aicore__ inline void PostCompute() {}

  protected:
    __aicore__ inline GlobalTensor<BT>
    SetGlobalBufferW(uint32_t groupIdx, uint32_t tailN, MNConfig &mnConfig);

    __aicore__ inline uint64_t SetWOffset(uint32_t tailN, uint32_t k);

  protected:
    TPipe *pipe;
    typename mmType::MT &mm; // matmul operator
    GM_ADDR xTensorPtr;
    GM_ADDR weightTensorPtr;
    GM_ADDR biasTensorPtr;
    GM_ADDR yTensorPtr;
    GlobalTensor<AT> xGm;
    GlobalTensor<BT> weightGm;
    GlobalTensor<BiasT> biasGm;
    GlobalTensor<DTYPE_Y> yGm;
#if defined(GMM_QUANT_INT8)
    GM_ADDR scaleTensorPtr;
    GlobalTensor<DTYPE_SCALE> scaleGm;
#endif
    uint32_t ubBaseN;
    uint32_t ubBaseK;
    uint32_t ubCalSize;
    uint32_t singleWeight;
    uint32_t singleX;
    uint32_t singleY;
    uint32_t coreNum;
    uint32_t subBlockIdx;
    bool mmWaitStatus;
};

template <typename mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
    GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
    const GMMBaseParams *__restrict gmmBaseParams,
    const TCubeTiling *__restrict mmTilingData, TPipe *tPipe) {
    xTensorPtr = x;
    weightTensorPtr = weight;
    biasTensorPtr = bias;
    yTensorPtr = y;
    pipe = tPipe;
    ubBaseN = gmmBaseParams->ubBaseN;
    ubBaseK = gmmBaseParams->ubBaseK;
    ubCalSize = gmmBaseParams->ubCalSize;
    singleWeight = gmmBaseParams->singleWeight;
    singleX = gmmBaseParams->singleX;
    singleY = gmmBaseParams->singleY;
    coreNum = gmmBaseParams->coreNum;
    subBlockIdx = GetSubBlockIdx();
    mmWaitStatus = false;
#if defined(GMM_QUANT_INT8)
    scaleTensorPtr = scale;
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    TBuf<> ubBuf;
    pipe->InitBuffer(ubBuf, TOTAL_UB_SIZE / 2);
    LocalTensor<uint8_t> buf = ubBuf.template Get<uint8_t>();
    mm.SetLocalWorkspace(buf);
#endif
}

template <typename mmType, bool sync>
__aicore__ inline uint64_t GMMCompute<mmType, sync>::SetWOffset(uint32_t tailN,
                                                                uint32_t k) {
    uint64_t wOffset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ && transposeW) {
        wOffset = tailN * (UB_BLOCK_UNIT_SIZE /
                           sizeof(BT)); // 32: quant is 32, float16 is 16
    } else if constexpr (mmType::BT::format == CubeFormat::NZ) {
        wOffset = tailN * AlignUp<16>(k); // 16: nz format last two dim size
    } else if constexpr (transposeW) {
        wOffset = tailN * k;
    } else {
        wOffset = tailN;
    }
    return wOffset;
}

template <typename mmType, bool sync>
__aicore__ inline GlobalTensor<typename mmType::BT::T>
GMMCompute<mmType, sync>::SetGlobalBufferW(uint32_t groupIdx, uint32_t tailN,
                                           MNConfig &mnConfig) {
    uint64_t wOffset = SetWOffset(tailN, mnConfig.k);
#if defined(GMM_ANTI_QUANT)
    return weightGm[transposeW ? mnConfig.workSpaceOffset - tailN + wOffset
                               : mnConfig.workSpaceOffset];
#else
    GlobalTensor<BT> weightGmLocal;
    if (singleWeight == 0) {
        weightGmLocal.SetGlobalBuffer(
            GetTensorAddr<BT>(groupIdx, weightTensorPtr) + wOffset);
    } else {
        weightGmLocal.SetGlobalBuffer(GetTensorAddr<BT>(0, weightTensorPtr) +
                                      mnConfig.wBaseOffset + wOffset);
    }
#if !(defined(ASCENDC_OOM) && ASCENDC_OOM == 1)
    if (mnConfig.blockDimM == 1) {
        weightGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
#endif
    return weightGmLocal;
#endif
}

template <typename mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::MMCompute(uint32_t groupIdx,
                                                           MNConfig &mnConfig,
                                                           uint32_t coreIdx) {
    if (subBlockIdx != 0) {
        return;
    }
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1
                              ? mnConfig.singleN
                              : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1
                              ? mnConfig.singleM
                              : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    // init global buffer
    if (singleX == 0) {
        xGm.SetGlobalBuffer(GetTensorAddr<AT>(groupIdx, xTensorPtr));
    } else {
        xGm.SetGlobalBuffer(GetTensorAddr<AT>(0, xTensorPtr) +
                            mnConfig.xBaseOffset);
    }
    GlobalTensor<BT> weightGmLocal =
        SetGlobalBufferW(groupIdx, tailN, mnConfig);
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset], transposeX);
    mm.SetTensorB(weightGmLocal, transposeW);
#if defined(GMM_QUANT_INT8)
    if (singleWeight == 0) {
        scaleGm.SetGlobalBuffer(
            GetTensorAddr<DTYPE_SCALE>(groupIdx, scaleTensorPtr));
    } else {
        scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE>(0, scaleTensorPtr) +
                                mnConfig.nAxisBaseOffset);
    }
    mm.SetQuantVector(scaleGm[tailN]);
#endif
    if (singleY == 0) {
        yGm.SetGlobalBuffer(GetTensorAddr<CT>(groupIdx, yTensorPtr));
    } else {
        yGm.SetGlobalBuffer(GetTensorAddr<CT>(0, yTensorPtr) +
                            mnConfig.yBaseOffset);
    }
#if defined(GMM_ANTI_QUANT)
    mm.template IterateAll<false>(yGm[outOffset], 0, false, true);
    mmWaitStatus = true;
#else
    mm.template IterateAll<sync>(yGm[outOffset], 0);
#endif
}

} // namespace GROUPED_MATMUL

#endif // ASCENDC_GROUPED_MATMUL_H
