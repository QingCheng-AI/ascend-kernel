#ifndef GROUPED_SOFT_GEMV_H
#define GROUPED_SOFT_GEMV_H

#include "gemv_impl.h"

using namespace AscendC;

namespace GROUPED_GEMV {

__aicore__ inline void GetValueFromGroupList(uint32_t groupIdx, uint32_t &m,
                                             int32_t &mIdx,
                                             GlobalTensor<int64_t> &groupList) {
    int32_t mIdx_ = static_cast<int32_t>(groupList.GetValue(groupIdx));
    m = mIdx_ - mIdx;
    mIdx = mIdx_;
}

template <typename T> __aicore__ inline T CeilDiv(T a, T b) {
    if (b == 0)
        return 0;
    return (a + b - 1) / b;
}

template <typename AT_, typename WT_, typename BT_, typename CT_, typename ST_>
struct MVType {
    using AT = AT_;
    using WT = WT_;
    using BT = BT_;
    using CT = CT_;
    using ST = ST_;
};

template <typename mvType> class GMVCompute {
  public:
    using AT = typename mvType::AT;
    using WT = typename mvType::WT;
    using BT = typename mvType::BT;
    using CT = typename mvType::CT;
    using ST = typename mvType::ST;

    __aicore__ inline GMVCompute() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scale,
                                GM_ADDR groupList, GM_ADDR y, GM_ADDR workspace,
                                const GMVBaseParams *__restrict gmvBaseParams,
                                TPipe *tPipe, GM_ADDR tiling);

    __aicore__ inline void GEMVProcess();

  private:
    __aicore__ inline void GEMVCompute();

    __aicore__ inline void DataCopyX(uint64_t offset);

    __aicore__ inline void DataCopyWeight(uint64_t offset);

    __aicore__ inline void SetGMPtr(uint64_t offsetX, uint64_t offsetW,
                                    uint64_t offsetY, uint64_t offsetS);

    __aicore__ inline void DataCopyScale(uint64_t offset);

    __aicore__ inline void DataCopyOut();

    __aicore__ inline void FreeTensor();

    __aicore__ inline void Epilogue();

    uint32_t m, n, k;
    uint32_t ubBaseN;
    uint32_t ubBaseK;
    uint32_t ubBaseM;
    uint32_t coreNum;
    uint32_t coreIdx;
    uint32_t groupNum;
    int32_t mIdx = 0;
    const GMVBaseParams *__restrict gmvBaseParams;
    GM_ADDR xPtr;
    GM_ADDR weightPtr;
    GM_ADDR yPtr;
    GM_ADDR scalePtr;
    GM_ADDR groupListPtr;
    GlobalTensor<int8_t> weightTensorGm;
    GlobalTensor<int64_t> groupListGm;
    GlobalTensor<AT> xGm;
    GlobalTensor<int8_t> weightGm;
    GlobalTensor<CT> yGm;
    GlobalTensor<ST> scaleGm;
    LocalTensor<AT> xInUb;
    LocalTensor<WT> wInUb;
    LocalTensor<ST> sInUb;
    LocalTensor<uint8_t> tmpUb;
    LocalTensor<float> aInUb;
    TQue<QuePosition::VECIN, 1> weightInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> xInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    TBuf<TPosition::VECCALC> accBuff;
    TPipe *pipe;
};

template <typename mvType>
__aicore__ inline void
GMVCompute<mvType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scale,
                         GM_ADDR groupList, GM_ADDR y, GM_ADDR workspace,
                         const GMVBaseParams *__restrict gmvBaseParams_,
                         TPipe *tPipe, GM_ADDR tiling) {
    coreIdx = static_cast<uint32_t>(GetBlockIdx());
    coreNum = static_cast<uint32_t>(GetBlockNum());
    gmvBaseParams = gmvBaseParams_;
    groupListPtr = groupList;
    pipe = tPipe;
    xPtr = x;
    weightPtr = weight;
    scalePtr = scale;
    yPtr = y;
    n = gmvBaseParams->n;
    k = gmvBaseParams->k;
    groupNum = gmvBaseParams->groupNum;
    ubBaseN = gmvBaseParams->ubBaseN;
    ubBaseK = gmvBaseParams->ubBaseK;
    ubBaseM = gmvBaseParams->ubBaseM;
    pipe->InitBuffer(xInQueue, 2, gmvBaseParams->ubXInSize * sizeof(AT));
    pipe->InitBuffer(weightInQueue, 2, gmvBaseParams->ubWInSize);
    pipe->InitBuffer(scaleInQueue, 2, gmvBaseParams->ubSInSize);
    pipe->InitBuffer(yOutQueue, 2, gmvBaseParams->ubOutSize * sizeof(CT));
    pipe->InitBuffer(accBuff, gmvBaseParams->ubOutSize * sizeof(float));
    pipe->InitBuffer(tmpBuff, 143360);
    tmpUb = tmpBuff.Get<uint8_t>();
    aInUb = accBuff.Get<float>();
    groupListGm.SetGlobalBuffer((__gm__ int64_t *)groupList);
}

template <typename mvType>
__aicore__ inline void GMVCompute<mvType>::DataCopyX(uint64_t offset) {
    DataCopyPadParams padParams;
    DataCopyParams xParams;
    LocalTensor<AT> xLocal = xInQueue.AllocTensor<AT>();
    xParams.blockLen = ubBaseK * sizeof(AT);
    xParams.blockCount = m;
    xParams.srcStride = (k - ubBaseK) * sizeof(AT);
    xParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], xParams, padParams);
    xInQueue.EnQue(xLocal);
    xInUb = xInQueue.template DeQue<AT>();
}

template <typename mvType>
__aicore__ inline void GMVCompute<mvType>::DataCopyScale(uint64_t offset) {
    DataCopyPadParams padParams;
    DataCopyParams scaleParams;
    LocalTensor<ST> scaleLocal = scaleInQueue.AllocTensor<ST>();
    if constexpr (IsSameType<WT, int4b_t>::value) {
        scaleParams.blockLen = ubBaseN * sizeof(ST);
        scaleParams.blockCount = ubBaseK / 16;
        scaleParams.srcStride = (n - ubBaseN) * sizeof(ST);
        scaleParams.dstStride = 0;
    } else {
        scaleParams.blockLen = sizeof(ST);
        scaleParams.blockCount = 1;
        scaleParams.srcStride = (n - ubBaseN) / 128 * sizeof(ST);
        scaleParams.dstStride = 0;
    }
    DataCopyPad(scaleLocal, scaleGm[offset], scaleParams, padParams);
    scaleInQueue.template EnQue(scaleLocal);
    sInUb = scaleInQueue.DeQue<ST>();
}

template <typename mvType>
__aicore__ inline void GMVCompute<mvType>::DataCopyWeight(uint64_t offset) {
    LocalTensor<int8_t> inLocal = weightInQueue.AllocTensor<int8_t>();
    DataCopyExtParams weightParams;
    weightParams.blockCount = ubBaseK;
    weightParams.dstStride = 0;
    DataCopyPadExtParams<int8_t> padParams;
    if constexpr (IsSameType<WT, int4b_t>::value) {
        weightParams.blockLen = ubBaseN / 2;
        weightParams.srcStride = (n - ubBaseN) / 2;
    } else {
        weightParams.blockLen = ubBaseN;
        weightParams.srcStride = n - ubBaseN;
    }
    DataCopyPad(inLocal, weightGm[offset], weightParams, padParams);
    weightInQueue.template EnQue(inLocal);
    wInUb = weightInQueue.DeQue<WT>();
}

template <typename mvType>
__aicore__ inline void GMVCompute<mvType>::DataCopyOut() {
    LocalTensor<CT> resUb = yOutQueue.DeQue<CT>();
    DataCopyParams outParams;
    outParams.blockLen = ubBaseN * sizeof(CT);
    outParams.blockCount = m;
    outParams.srcStride = 0;
    outParams.dstStride = (n - ubBaseN) * sizeof(CT);
    DataCopyPad(yGm, resUb, outParams);
    yOutQueue.FreeTensor(resUb);
}

template <typename mvType>
__aicore__ inline void
GMVCompute<mvType>::SetGMPtr(uint64_t offsetX, uint64_t offsetW,
                             uint64_t offsetY, uint64_t offsetS) {
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ AT *>(xPtr) + offsetX);
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weightPtr) +
                             offsetW);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(yPtr) + offsetY);
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ ST *>(scalePtr) + offsetS);
}

template <typename mvType>
__aicore__ inline void GMVCompute<mvType>::FreeTensor() {
    weightInQueue.FreeTensor(wInUb);
    scaleInQueue.FreeTensor(sInUb);
    xInQueue.FreeTensor(xInUb);
}

template <typename mvType>
__aicore__ inline void GMVCompute<mvType>::Epilogue() {
    LocalTensor<CT> resInUb = yOutQueue.AllocTensor<CT>();
    UnaryRepeatParams f322f16unaryParam;
    f322f16unaryParam.dstRepStride = ONE_REPEAT_BLOCKS / 2;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, m * ubBaseN);
    Cast<CT, float, false>(resInUb, aInUb, RoundMode::CAST_RINT,
                           MASK_PLACEHOLDER, REPEAT_PLACE_HOLDER,
                           f322f16unaryParam);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    yOutQueue.template EnQue(resInUb);
    DataCopyOut();
}

template <typename mvType>
__aicore__ inline void GMVCompute<mvType>::GEMVProcess() {
    uint32_t numBlockN = n / ubBaseN;
    uint32_t blockCount = 0;
    uint32_t curBlock = coreIdx;
    uint32_t pastBlock = 0;
    for (int groupIdx = 0; groupIdx < groupNum; groupIdx++) {
        GetValueFromGroupList(groupIdx, m, mIdx, groupListGm);
        uint32_t numBlockM = CeilDiv<uint32_t>(m, ubBaseM);
        pastBlock = blockCount;
        blockCount += numBlockM * numBlockN;
        while (curBlock < blockCount) {
            uint32_t nIdx = curBlock - pastBlock;
            uint64_t offsetW, offsetS;
            uint64_t offsetX = (mIdx - m) * k;
            uint64_t offsetY = (mIdx - m) * n + nIdx * ubBaseN;
            if constexpr (IsSameType<WT, int4b_t>::value) {
                offsetW = groupIdx * n * k;
                offsetS = offsetW / 16 + nIdx * ubBaseN;
                offsetW = (offsetW + nIdx * ubBaseN) / 2;
            } else {
                offsetW = groupIdx * n * k;
                offsetS = offsetW / (128 * 128) + nIdx;
                offsetW += nIdx * ubBaseN;
            }
            SetGMPtr(offsetX, offsetW, offsetY, offsetS);
            Duplicate<float>(aInUb, 0, ubBaseN * m);
            GEMVCompute();
            curBlock += coreNum;
        }
    }
}

template <typename mvType>
__aicore__ inline void GMVCompute<mvType>::GEMVCompute() {
    for (int offsetK = 0; offsetK < k; offsetK += ubBaseK) {
        uint64_t offsetS = 0;
        DeQuantShapeInfo shapeInfo;
        shapeInfo.baseN = ubBaseN;
        shapeInfo.baseK = ubBaseK;
        shapeInfo.M = m;

        if constexpr (IsSameType<WT, int4b_t>::value) {
            offsetS = offsetK * n / 16;
            DataCopyWeight(offsetK * n / 2);
            DataCopyX(offsetK);
            DataCopyScale(offsetS);
            SoftGEMV(wInUb, aInUb, xInUb, sInUb, tmpUb, shapeInfo);
        } else {
            offsetS = offsetK * n / (128 * 128);
            DataCopyWeight(offsetK * n);
            DataCopyX(offsetK);
            DataCopyScale(offsetS);
            SoftGEMV(wInUb, aInUb, xInUb, sInUb(0), tmpUb, shapeInfo);
        }
        FreeTensor();
    }
    Epilogue();
}

} // namespace GROUPED_GEMV
#endif