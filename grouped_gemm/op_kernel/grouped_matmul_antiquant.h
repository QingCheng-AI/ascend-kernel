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
 * \file grouped_matmul_antiquant.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_ANTIQUANT_H
#define ASCENDC_GROUPED_MATMUL_ANTIQUANT_H

#include "antiquant_impl.h"
#include "grouped_matmul.h"

// #ifdef GMM_ANTI_QUANT
namespace GROUPED_MATMUL {

constexpr uint32_t CAST_THRESHOLD_CACHE_BIG =
    16 * 1024 * 1024; // 16M is obtained by tests
constexpr uint32_t CAST_THRESHOLD_CACHE_SMALL =
    10 * 1024 * 1024; // 10M is obtained by tests
constexpr uint32_t CAST_PERFORMANCE_MAX_N = 5120;
constexpr uint32_t CAST_MIN_SINGLE_K = 8;
constexpr int32_t BEST_UB_BASEN = 512;

/*@brief store variables for core split configuration
 */
struct CastWeightConfig {
    uint32_t coreNum = 0;
    uint32_t nUsedCore = 0;
    uint32_t curDimN = 0;
    uint32_t castRoundIdx = 0;
    uint32_t workSpaceIdx = 0;
    uint64_t wInNOffset = 0;
    uint32_t wInKOffset = 0;
    uint32_t curSingleN = 0;
    uint32_t curSingleK = 0;
    uint32_t tailN = 0;
};

/** @brief GroupMatmul Antiquant operator Class
 */
template <typename ComputeType>
class GMMAntiquantProcess : public GMMProcess<ComputeType> {
  protected:
    constexpr static bool antiquantPerformance =
        ComputeType::antiquantPerformanceFlag;

  public:
    /** @brief constructor */
    __aicore__ inline GMMAntiquantProcess(ComputeType &computeOp_)
        : GMMProcess<ComputeType>(computeOp_) {}

    __aicore__ inline void Process();

  private:
    __aicore__ inline void
    SetAntiquantMNConfig(const uint64_t singleWorkSpaceSize,
                         const uint32_t curBlock, bool &validCore,
                         CastWeightConfig &castConfig, MNConfig &mnConfig);

    __aicore__ inline void SetAntiquantCastConfig(uint32_t &curCount,
                                                  MNConfig mnConfig,
                                                  CastWeightConfig &castConfig);
    __aicore__ inline void
    AntiquantUpdateSingleM(MNConfig &mnConfig, uint32_t &dimM, uint32_t dimN);
};

template <typename ComputeType>
__aicore__ inline void GMMAntiquantProcess<ComputeType>::SetAntiquantMNConfig(
    const uint64_t singleWorkSpaceSize, const uint32_t curBlock,
    bool &validCore, CastWeightConfig &castConfig, MNConfig &mnConfig) {
    mnConfig.workSpaceOffset = castConfig.workSpaceIdx * singleWorkSpaceSize;
    castConfig.workSpaceIdx = castConfig.workSpaceIdx == 0
                                  ? 1
                                  : 0; // next round use another workspace
    castConfig.castRoundIdx =
        Ceil(curBlock + 1, castConfig.coreNum) -
        1; // +1: let curBlock start from 1,-1: castRoundIdx start from 0
    castConfig.curDimN = castConfig.nUsedCore;
    if (castConfig.castRoundIdx ==
        Ceil(mnConfig.blockDimN, castConfig.nUsedCore) - 1) { // -1 last round
        castConfig.curDimN =
            mnConfig.blockDimN - castConfig.castRoundIdx * castConfig.nUsedCore;
    }
    // compute dimM
    uint32_t dimM = Max<uint32_t>(castConfig.coreNum / castConfig.curDimN,
                                  1); // 1: The minimum value of dimM is 1
    dimM = Min<uint32_t>(Ceil(mnConfig.m, this->mmTilingData->baseM), dimM);
    mnConfig.singleM = Ceil(mnConfig.m, dimM);
    mnConfig.blockDimM = dimM;
    mnConfig.mIdx = this->coreIdx / castConfig.curDimN;
    mnConfig.nIdx = this->coreIdx % castConfig.curDimN;
    validCore = this->coreIdx < dimM * castConfig.curDimN;
}

template <typename ComputeType>
__aicore__ inline void GMMAntiquantProcess<ComputeType>::SetAntiquantCastConfig(
    uint32_t &curCount, MNConfig mnConfig, CastWeightConfig &castConfig) {
    if (mnConfig.blockDimM > 0 && mnConfig.blockDimN > 0) {
        // 16M and 10M is obtained by tests. When N is greater than 5120, the
        // cache uses 10 MB for better performance
        uint32_t cacheThreshold = mnConfig.n > CAST_PERFORMANCE_MAX_N
                                      ? CAST_THRESHOLD_CACHE_SMALL
                                      : CAST_THRESHOLD_CACHE_BIG;
        // 16M/k is the length of N that needs to be calculated for single
        // round. 16M/k/baseN is the coreNum required for single round
        // calculation of the N-axis.
        castConfig.nUsedCore = Min<uint32_t>(
            Ceil(cacheThreshold, mnConfig.k * this->mmTilingData->baseN),
            castConfig.coreNum);
        castConfig.nUsedCore =
            Min<uint32_t>(castConfig.nUsedCore, mnConfig.blockDimN);
        curCount =
            Ceil(mnConfig.blockDimN, castConfig.nUsedCore) * castConfig.coreNum;
    }
}

template <typename ComputeType>
__aicore__ inline void GMMAntiquantProcess<ComputeType>::AntiquantUpdateSingleM(
    MNConfig &mnConfig, uint32_t &dimM, uint32_t dimN) {
    if (dimM > 1 && dimN < this->gmmBaseParams->coreNum) {
        uint32_t restCores = this->gmmBaseParams->coreNum / dimN;
        if (dimM > restCores) {
            mnConfig.singleM = Ceil(mnConfig.m, restCores);
            dimM = Ceil(mnConfig.m, mnConfig.singleM);
        }
    }
}

template <typename ComputeType>
__aicore__ inline void GMMAntiquantProcess<ComputeType>::Process() {
    MNConfig mnConfig;
    CastWeightConfig castConfig;
    castConfig.coreNum = this->gmmBaseParams->coreNum;
    bool validCore = true;
    uint64_t singleWorkSpaceSize =
        this->gmmBaseParams->workspaceSize /
        2; // 2: antiQuantNormal use 2 block workspace
    if (this->gmmBaseParams->groupType != -1) { // -1: no need to split
        this->preOffset = 0;
        if (unlikely(this->groupListPtr == nullptr)) {
            this->groupNum = 0;
        } // not continue Process
    }
    this->SetMNConfig(mnConfig);
    for (uint32_t groupIdx = 0, count = 0; groupIdx < this->groupNum;
         ++groupIdx) {
        int32_t splitValue = GetSplitValueFromGroupList(
            groupIdx, this->preOffset, this->gmmBaseParams, this->groupListGm);
        mnConfig.m = splitValue;
        uint32_t dimM = Ceil(mnConfig.m, mnConfig.singleM);
        uint32_t dimN = Ceil(mnConfig.n, mnConfig.singleN);
        if constexpr (!antiquantPerformance) {
            AntiquantUpdateSingleM(mnConfig, dimM, dimN);
        }
        mnConfig.blockDimM = dimM;
        mnConfig.blockDimN = dimN;
        uint32_t curCount = count + dimM * dimN;
        uint32_t curBlock = this->coreIdx >= count
                                ? this->coreIdx
                                : this->coreIdx + this->gmmBaseParams->coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * dimN;

        if constexpr (antiquantPerformance) {
            SetAntiquantCastConfig(curCount, mnConfig, castConfig);
        }

        while (curBlock < curCount) {
            if constexpr (antiquantPerformance) { // performance verison, will
                                                  // split dimN
                SetAntiquantMNConfig(singleWorkSpaceSize, curBlock, validCore,
                                     castConfig, mnConfig);
            } else {
                mnConfig.workSpaceOffset = mnConfig.wBaseOffset;
                this->MNBlockIdxCompute(mnConfig, curBlock, count,
                                        thresholdM_dimN);
            }
            this->computeOp.PreCompute(groupIdx, this->coreIdx % 2, mnConfig,
                                       castConfig);
            this->computeOp.MMSync();
            if (validCore) {
                mnConfig.workSpaceOffset += mnConfig.nIdx * mnConfig.singleN;
                if constexpr (antiquantPerformance) {
                    mnConfig.nIdx +=
                        castConfig.castRoundIdx * castConfig.nUsedCore;
                }
                this->computeOp.MMCompute(groupIdx, mnConfig, this->coreIdx);
            }
            curBlock += this->gmmBaseParams->coreNum;
        }
        this->UpdateMnConfig(mnConfig);
        count = curCount % this->gmmBaseParams->coreNum;
    }
}

/** @brief intenal computation class
 */
template <class mmType, bool sync = false, bool antiquantPerformance = false>
class GMMAntiquantCompute : public GMMCompute<mmType, sync> {
  public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = DTYPE_WEIGHT;
    using ScaleType = DTYPE_ANTIQUANT_SCALE;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;
    constexpr static bool antiquantPerformanceFlag = antiquantPerformance;

    __aicore__ inline GMMAntiquantCompute(typename mmType::MT &mm_)
        : GMMCompute<mmType, sync>(mm_) {}

    __aicore__ inline void
    Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
         GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
         GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
         const GMMBaseParams *__restrict gmmBaseParams,
         const TCubeTiling *__restrict mmTilingData, TPipe *tPipe);

    __aicore__ inline void PreCompute(uint32_t groupIdx, uint32_t coreIdx,
                                      MNConfig &mnConfig,
                                      CastWeightConfig &castConfig);

    __aicore__ inline void MMSync();

  private:
    __aicore__ inline void CastWeightProcess(MNConfig &mnConfig,
                                             CastWeightConfig &castConfig);
    __aicore__ inline void SetAntiQuantGlobalBuffer(uint32_t groupIdx,
                                                    const MNConfig mnConfig);
    __aicore__ inline void
    SetGmToUbDataCopyParams(const uint32_t curBaseN, const uint32_t curBaseK,
                            const MNConfig &mnConfig,
                            DataCopyExtParams &intriParams);
    __aicore__ inline void
    SetUbToGmDataCopyParams(const uint32_t curBaseN, const uint32_t alignRowLen,
                            const uint32_t curBaseK, const MNConfig &mnConfig,
                            DataCopyExtParams &intriParams);
    __aicore__ inline void CastWeightCompute(uint32_t curCalcK,
                                             uint32_t curCalcAlignN);
    __aicore__ inline void DataCopyScaleAndOffset(uint32_t curBaseN,
                                                  uint32_t alignBaseN,
                                                  uint64_t realScaleOffset,
                                                  const MNConfig &mnConfig);
    __aicore__ inline void ComputeUbBaseK(uint32_t curSingleK, uint32_t offsetK,
                                          uint32_t newBaseK,
                                          uint32_t &curUsedGroupSize,
                                          uint32_t &curBaseK);
    __aicore__ inline void FreeScaleAndOffset(bool &firstLoop);

    GlobalTensor<int8_t> weightAntiQuantGm;
    GM_ADDR antiScaleTensorPtr;
    GM_ADDR antiOffsetTensorPtr;
    LocalTensor<ScaleType> scaleInUb;
    LocalTensor<ScaleType> offsetInUb;
    GlobalTensor<ScaleType> antiScaleGM;
    GlobalTensor<ScaleType> antiOffsetGM;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> offsetInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<BT> tmpUb;
    bool isPerGroup = false;
    uint32_t perGroupSize;
};

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
    GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
    const GMMBaseParams *__restrict gmmBaseParams,
    const TCubeTiling *__restrict mmTilingData, TPipe *tPipe) {
    this->GMMCompute<mmType, sync>::Init(x, weight, bias, scale, offset,
                                         antiquantScale, antiquantOffset,
                                         groupList, perTokenScale, y, workspace,
                                         gmmBaseParams, mmTilingData, tPipe);
    antiScaleTensorPtr = antiquantScale;
    antiOffsetTensorPtr = antiquantOffset;
    perGroupSize = gmmBaseParams->quantParam;
    isPerGroup = perGroupSize > 0;
    this->weightGm.SetGlobalBuffer((__gm__ BT *)workspace);
    // uint32_t maxUbBaseN = BEST_UB_BASEN;
    uint32_t maxUbBaseN;
    if constexpr (IsSameType<WT, int4b_t>::value) {
        maxUbBaseN = 256;
    } else {
        maxUbBaseN = 128;
    }
    if constexpr (transposeW) {
        maxUbBaseN = this->ubBaseN;
    }
    if constexpr (IsSameType<WT, int4b_t>::value) {
        this->pipe->InitBuffer(scaleInQueue, 2,
                               maxUbBaseN * sizeof(ScaleType) * 4);
    } else {
        this->pipe->InitBuffer(scaleInQueue, 2, 32);
    }
    this->pipe->InitBuffer(offsetInQueue, 2, maxUbBaseN * sizeof(ScaleType));
    this->pipe->InitBuffer(vecInQueue, 2,
                           this->ubCalSize * GetTypeBits<WT>() / INT8_BITS);
    this->pipe->InitBuffer(vecOutQueue, 2,
                           this->ubCalSize * sizeof(BT) / 64 * 65);
    this->pipe->InitBuffer(tmpBuff, 102400);
    tmpUb = tmpBuff.Get<AT>();
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::PreCompute(
    uint32_t groupIdx, uint32_t coreIdx, MNConfig &mnConfig,
    CastWeightConfig &castConfig) {
    castConfig.curSingleN = 0;
    castConfig.curSingleK = 0;
    castConfig.wInKOffset = 0;
    castConfig.wInNOffset = 0;
    mnConfig.wOutOffset = mnConfig.workSpaceOffset;
    castConfig.tailN = 0;
    if constexpr (antiquantPerformance) { // antiquant normal version
        uint32_t blockDimK =
            Min<uint32_t>(this->coreNum, Ceil(mnConfig.k, CAST_MIN_SINGLE_K));
        if (coreIdx >= blockDimK) {
            return;
        }
        castConfig.curSingleK = Ceil(mnConfig.k, blockDimK);
        castConfig.tailN =
            castConfig.castRoundIdx * castConfig.nUsedCore * mnConfig.singleN;
        castConfig.wInNOffset = castConfig.tailN;
        castConfig.wInKOffset = coreIdx * castConfig.curSingleK;
        if (coreIdx == blockDimK - 1) { // -1: last dimK
            castConfig.curSingleK =
                mnConfig.k - castConfig.curSingleK * coreIdx;
        }
        mnConfig.wOutOffset += castConfig.wInKOffset * mnConfig.n;
        castConfig.curSingleN = castConfig.curDimN * mnConfig.singleN;
        if (castConfig.castRoundIdx ==
            Ceil(mnConfig.blockDimN, castConfig.nUsedCore) -
                1) { // -1: last round
            castConfig.curSingleN = mnConfig.n - castConfig.castRoundIdx *
                                                     castConfig.nUsedCore *
                                                     mnConfig.singleN;
        }
    } else { // antiquant generalized version
        castConfig.curSingleN = mnConfig.singleN;
        castConfig.curSingleK = mnConfig.k;
        castConfig.tailN = mnConfig.nIdx * mnConfig.singleN;
        castConfig.wInNOffset =
            this->transposeW ? castConfig.tailN * mnConfig.k : castConfig.tailN;
        mnConfig.wOutOffset += castConfig.wInNOffset;
        if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
            castConfig.curSingleN =
                mnConfig.n - mnConfig.nIdx * mnConfig.singleN;
        }
    }
    SetAntiQuantGlobalBuffer(groupIdx, mnConfig);
    CastWeightProcess(mnConfig, castConfig);
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::MMSync() {
    if (this->mmWaitStatus) {
        this->mm.WaitIterateAll();
        this->mmWaitStatus = false;
    }
    if constexpr (IsSameType<WT, int4b_t>::value) {
        if constexpr (antiquantPerformance) {
            SyncAll<true>();
        }
    } else {
        AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
        AscendC::CrossCoreWaitFlag(0x8);
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void GMMAntiquantCompute<mmType, sync, antiquantPerformance>::
    SetAntiQuantGlobalBuffer(uint32_t groupIdx, const MNConfig mnConfig) {
    if (this->singleWeight == 0) {
        weightAntiQuantGm.SetGlobalBuffer(
            GetTensorAddr<int8_t>(groupIdx, this->weightTensorPtr));
        antiScaleGM.SetGlobalBuffer(
            GetTensorAddr<ScaleType>(groupIdx, antiScaleTensorPtr));
    } else {
        weightAntiQuantGm.SetGlobalBuffer(
            GetTensorAddr<int8_t>(0, this->weightTensorPtr) +
            mnConfig.wBaseOffset * GetTypeBits<WT>() / INT8_BITS);
        uint64_t antiquantParamsOffset = mnConfig.nAxisBaseOffset;
        if (isPerGroup) {
            if constexpr (IsSameType<WT, int4b_t>::value) {
                // fp4的处理逻辑是，每次计算完一个group，就需要偏移一个矩阵，但因为只有k方向有倍数差（16），因此只需要除以perGroupSize
                antiquantParamsOffset *= (mnConfig.k / perGroupSize);
            } else {
                // assert(perGroupSize == 128 && "must check if scale is 128 *
                // 128 -> 1");
                antiquantParamsOffset =
                    (mnConfig.nAxisBaseOffset / perGroupSize);
                antiquantParamsOffset *= (mnConfig.k / perGroupSize);
            }
        }

        antiScaleGM.SetGlobalBuffer(
            GetTensorAddr<ScaleType>(0, antiScaleTensorPtr) +
            antiquantParamsOffset);
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::ComputeUbBaseK(
    uint32_t curSingleK, uint32_t offsetK, uint32_t newBaseK,
    uint32_t &curUsedGroupSize, uint32_t &curBaseK) {
    if (unlikely(offsetK + newBaseK >= curUsedGroupSize)) {
        curBaseK = curUsedGroupSize - offsetK;
        curUsedGroupSize += perGroupSize;
        if (offsetK + curBaseK > curSingleK) {
            curBaseK = curSingleK - offsetK;
        }
    } else if (unlikely(offsetK + newBaseK > curSingleK)) {
        curBaseK = curSingleK - offsetK;
    } else {
        curBaseK = newBaseK;
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::FreeScaleAndOffset(
    bool &firstLoop) {
    if (firstLoop) {
        firstLoop = false;
    } else {
        scaleInQueue.FreeTensor(scaleInUb);
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::CastWeightProcess(
    MNConfig &mnConfig, CastWeightConfig &castConfig) {
    uint64_t wInOffset =
        castConfig.wInNOffset +
        static_cast<uint64_t>(castConfig.wInKOffset) * mnConfig.n;
    const uint32_t &curSingleK = castConfig.curSingleK;
    const uint32_t &curSingleN = castConfig.curSingleN;
    const uint32_t &scaleOffset = castConfig.tailN;
    uint32_t newBaseK = this->ubBaseK;
    uint32_t newBaseN = this->ubBaseN;
    uint32_t usedGroupSize = mnConfig.k;
    if (isPerGroup) {
        usedGroupSize =
            perGroupSize + AlignDown(castConfig.wInKOffset, perGroupSize);
    }
    DataCopyPadExtParams<int8_t> padParams;
    for (uint32_t offsetN(0), curBaseN(newBaseN), nCount(0);
         offsetN < curSingleN; offsetN += newBaseN) {
        if (unlikely(offsetN + newBaseN > curSingleN)) {
            curBaseN = curSingleN - offsetN;
        }
        uint32_t alignBaseN = AlignUp(curBaseN, UB_BLOCK_UNIT_SIZE * INT8_BITS /
                                                    GetTypeBits<WT>());
        if (!isPerGroup) {
            DataCopyScaleAndOffset(curBaseN, alignBaseN, scaleOffset + offsetN,
                                   mnConfig);
        }
        uint32_t curBaseK = newBaseK;
        bool firstKLoop = true;
        int32_t prePergroupIdx = -1;
        int32_t curPergroupIdx = 0;
        for (uint32_t offsetK(0), subCoreCount(nCount); offsetK < curSingleK;
             offsetK += curBaseK) {
            if (this->subBlockIdx == (++subCoreCount) % 2) { // 2: two vectors
                continue;
            }
            if (isPerGroup) {
                curPergroupIdx =
                    (offsetK + castConfig.wInKOffset) / perGroupSize;
                if (firstKLoop ||
                    curPergroupIdx > prePergroupIdx) { // load new group
                    FreeScaleAndOffset(firstKLoop);
                    if constexpr (IsSameType<WT, int4b_t>::value) {
                        DataCopyScaleAndOffset(curBaseN, alignBaseN,
                                               scaleOffset + offsetN +
                                                   curPergroupIdx * mnConfig.n,
                                               mnConfig);
                    } else {
                        DataCopyScaleAndOffset(curBaseN, alignBaseN,
                                               ((scaleOffset + offsetN +
                                                 curPergroupIdx * mnConfig.n) /
                                                128),
                                               mnConfig);
                    }
                    prePergroupIdx = curPergroupIdx;
                }
            }
            LocalTensor<int8_t> inLocal = vecInQueue.AllocTensor<int8_t>();
            DataCopyExtParams gmToUbIntriParams;
            SetGmToUbDataCopyParams(curBaseN, curBaseK, mnConfig,
                                    gmToUbIntriParams);
            uint64_t weightInOffset =
                transposeW
                    ? offsetK + static_cast<uint64_t>(offsetN) * mnConfig.k
                    : static_cast<uint64_t>(offsetK) * mnConfig.n + offsetN;
            if constexpr (IsSameType<WT, int4b_t>::value) {
                DataCopyPad(inLocal,
                            weightAntiQuantGm[(weightInOffset + wInOffset) *
                                              GetTypeBits<WT>() / INT8_BITS],
                            gmToUbIntriParams, padParams);
            } else {
                DataCopyPad(inLocal,
                            weightAntiQuantGm[(weightInOffset + wInOffset) *
                                              GetTypeBits<WT>() / INT8_BITS],
                            gmToUbIntriParams, padParams);
            }
            vecInQueue.EnQue(inLocal);

            DataCopyExtParams ubToGmIntriParams;
            if constexpr (transposeW) {
                uint32_t alignBaseK =
                    AlignUp(curBaseK,
                            UB_BLOCK_UNIT_SIZE * INT8_BITS / GetTypeBits<WT>());
                CastWeightCompute(alignBaseK, alignBaseN);
                SetUbToGmDataCopyParams(curBaseN, alignBaseK, curBaseK,
                                        mnConfig, ubToGmIntriParams);
            } else {
                CastWeightCompute(curBaseK, alignBaseN);
                SetUbToGmDataCopyParams(curBaseN, alignBaseN, curBaseK,
                                        mnConfig, ubToGmIntriParams);
            }

            // ResultCopy2GM
            LocalTensor<BT> wResUb = vecOutQueue.DeQue<BT>();
            uint64_t weightOutOffset =
                transposeW
                    ? mnConfig.wOutOffset + offsetK + offsetN * mnConfig.k
                    : mnConfig.wOutOffset + offsetK * mnConfig.n + offsetN;
            if constexpr (IsSameType<WT, int4b_t>::value) {
                for (int i = 0; i < 4; i++) {
                    DataCopyPad(
                        this->weightGm[weightOutOffset + mnConfig.n * 16 * i],
                        wResUb[256 * i], ubToGmIntriParams);
                }
                for (int i = 0; i < 4; i++) {
                    DataCopyPad(this->weightGm[weightOutOffset +
                                               mnConfig.n * (16 * i + 8)],
                                wResUb[8320 + 256 * i],
                                ubToGmIntriParams); // 8320 = 520 * 16, 520 =
                                                    // 512 + 8 (padding)
                }
            } else {

                DataCopyPad(this->weightGm[weightOutOffset], wResUb,
                            ubToGmIntriParams);
            }
            vecOutQueue.FreeTensor(wResUb);
        }
        nCount = nCount == 0 ? 1 : 0;
        if (!(isPerGroup && firstKLoop)) {
            scaleInQueue.FreeTensor(scaleInUb);
        }
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::CastWeightCompute(
    uint32_t curCalcK, uint32_t curCalcAlignN) {
    LocalTensor<WT> wInUb = vecInQueue.DeQue<WT>();
    wInUb.SetSize(curCalcK * curCalcAlignN);
    LocalTensor<BT> wResUb = vecOutQueue.AllocTensor<BT>();
    LocalTensor<uint8_t> tmpLocal = tmpUb.template ReinterpretCast<uint8_t>();
    AntiQuantShapeInfo shapeInfo;
    if constexpr (IsSameType<WT, int4b_t>::value) {
        shapeInfo.perGroupSize = 16;
        shapeInfo.baseN = 256;
        shapeInfo.baseK = 64;
        AntiQuant2<bfloat16_t, bfloat16_t>(
            wInUb, wResUb.template ReinterpretCast<bfloat16_t>(),
            scaleInUb.template ReinterpretCast<bfloat16_t>(), tmpLocal,
            shapeInfo);
    } else {
        shapeInfo.perGroupSize = 128;
        shapeInfo.baseN = 128;
        shapeInfo.baseK = 128;
        AntiQuant2<bfloat16_t, float>(
            wInUb, wResUb.template ReinterpretCast<bfloat16_t>(), scaleInUb(0),
            tmpLocal, shapeInfo);
    }

    vecInQueue.FreeTensor(wInUb);
    vecOutQueue.EnQue<BT>(wResUb);
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void GMMAntiquantCompute<mmType, sync, antiquantPerformance>::
    SetGmToUbDataCopyParams(const uint32_t curBaseN, const uint32_t curBaseK,
                            const MNConfig &mnConfig,
                            DataCopyExtParams &intriParams) {
    if constexpr (IsSameType<WT, int4b_t>::value) {
        if constexpr (transposeW) {
            intriParams.blockLen =
                Ceil(curBaseK * GetTypeBits<WT>(), INT8_BITS);
            intriParams.blockCount = curBaseN;
            intriParams.srcStride =
                Ceil((mnConfig.k - curBaseK) * GetTypeBits<WT>(), INT8_BITS);
            intriParams.dstStride = 0;
        } else {
            intriParams.blockLen =
                Ceil(curBaseN / 2, 1); // 此处除以2因为用int8_t表示int4b_t
            intriParams.blockCount = curBaseK;
            intriParams.srcStride =
                Ceil((mnConfig.n - curBaseN) * GetTypeBits<WT>(), INT8_BITS);
            intriParams.dstStride = 0;
        }
    } else {
        if constexpr (transposeW) {
            intriParams.blockLen =
                Ceil(curBaseK * GetTypeBits<WT>(), INT8_BITS);
            intriParams.blockCount = curBaseN;
            intriParams.srcStride =
                Ceil((mnConfig.k - curBaseK) * GetTypeBits<WT>(), INT8_BITS);
            intriParams.dstStride = 0;
        } else {
            intriParams.blockLen = curBaseN;
            intriParams.blockCount = curBaseK;
            intriParams.srcStride =
                Ceil((mnConfig.n - curBaseN) * GetTypeBits<WT>(), INT8_BITS);
            intriParams.dstStride = 0;
        }
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void GMMAntiquantCompute<mmType, sync, antiquantPerformance>::
    SetUbToGmDataCopyParams(const uint32_t curBaseN, const uint32_t alignRowLen,
                            const uint32_t curBaseK, const MNConfig &mnConfig,
                            DataCopyExtParams &intriParams) {
    if (IsSameType<WT, int4b_t>::value) {
        if constexpr (transposeW) {
            uint32_t alignBaseK = AlignUp(curBaseK, UB_BLOCK_UNIT_SIZE);
            intriParams.blockLen = curBaseK * sizeof(BT);
            intriParams.blockCount = curBaseN;
            intriParams.srcStride =
                (alignRowLen - curBaseK) / (UB_BLOCK_UNIT_SIZE / sizeof(BT));
            intriParams.dstStride = (mnConfig.k - curBaseK) * sizeof(BT);
        } else {
            intriParams.blockLen = curBaseN * sizeof(BT);
            intriParams.blockCount = 8;
            intriParams.srcStride = 49;
            intriParams.dstStride = (mnConfig.n - curBaseN) * sizeof(BT);
        }
    } else {
        if constexpr (transposeW) {
            uint32_t alignBaseK = AlignUp(curBaseK, UB_BLOCK_UNIT_SIZE);
            intriParams.blockLen = curBaseK * sizeof(BT);
            intriParams.blockCount = curBaseN;
            intriParams.srcStride =
                (alignRowLen - curBaseK) / (UB_BLOCK_UNIT_SIZE / sizeof(BT));
            intriParams.dstStride = (mnConfig.k - curBaseK) * sizeof(BT);
        } else {
            intriParams.blockLen = curBaseN * sizeof(BT);
            intriParams.blockCount = curBaseK;
            intriParams.srcStride = 0;
            intriParams.dstStride = (mnConfig.n - curBaseN) * sizeof(BT);
        }
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::DataCopyScaleAndOffset(
    uint32_t curBaseN, uint32_t alignBaseN, uint64_t realScaleOffset,
    const MNConfig &mnConfig) {
    DataCopyPadParams padParams;
    DataCopyParams scaleParams;
    LocalTensor<ScaleType> scaleLocal = scaleInQueue.AllocTensor<ScaleType>();
    if constexpr (IsSameType<WT, int4b_t>::value) {
        scaleParams.blockLen =
            curBaseN * sizeof(ScaleType); // 每行scale对应256个数
        scaleParams.blockCount =
            4; // 64行weight对应1行scale，每次处理256行，也即4个block
        scaleParams.srcStride =
            (mnConfig.n - curBaseN) * sizeof(ScaleType); // 调整读取的区间
        scaleParams.dstStride = 0;
        DataCopyPad(scaleLocal, antiScaleGM[realScaleOffset], scaleParams,
                    padParams);
    } else {
        scaleParams.blockLen = sizeof(ScaleType); // 每128 * 128对应一个scale
        scaleParams.blockCount = 1;               // 每次处理128 * 128
        DataCopyPad(scaleLocal, antiScaleGM[realScaleOffset], scaleParams,
                    padParams);
    }
    scaleInQueue.EnQue(scaleLocal);

    scaleInUb = scaleInQueue.DeQue<ScaleType>();
    if constexpr (IsSameType<WT, int4b_t>::value) {
        scaleInUb.SetSize(alignBaseN * 4);
    }
}

template <class mmType, bool sync = false>
using GMMAntiquantComputePerformance = GMMAntiquantCompute<mmType, sync, true>;

template <class mmType, bool sync = false>
using GMMAntiquantComputeNorm = GMMAntiquantCompute<mmType, sync, false>;

} // namespace GROUPED_MATMUL

// #endif  // GMM_ANTI_QUANT
#endif // ASCENDC_GROUPED_MATMUL_ANTIQUANT_H
