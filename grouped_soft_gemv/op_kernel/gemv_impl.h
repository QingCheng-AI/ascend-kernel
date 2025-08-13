#ifndef GEMV_IMPL_H
#define GEMV_IMPL_H
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tensor.h"

namespace GROUPED_GEMV {

using namespace AscendC;
constexpr uint32_t ONE_REPEAT_BLOCKS = 8;
constexpr uint32_t ONE_REPEAT_BYTES = 256;
constexpr uint32_t ONE_BLOCK_SIZE = 32;
constexpr uint32_t REPEAT_PLACE_HOLDER = 1;
struct DeQuantShapeInfo {
    uint32_t baseN{256};
    uint32_t baseK{64};
    uint32_t perGroupSize{16};
    uint32_t N{0};
    uint32_t K{0};
    uint32_t M{1};
};

template <typename SrcType, typename DstType>
__aicore__ inline void
preCastScale(const LocalTensor<SrcType> &scale,
             const LocalTensor<uint8_t> &sharedTmpBuffer) {

    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = ONE_REPEAT_BLOCKS / 2;
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(0, scale.GetSize());
    Cast<DstType, SrcType, false>(
        sharedTmpBuffer.template ReinterpretCast<DstType>(), scale,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, REPEAT_PLACE_HOLDER,
        unaryParams);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename SrcType, typename DstType>
__aicore__ inline void preCastFP4Weight(const LocalTensor<SrcType> &src,
                                        const LocalTensor<uint8_t> &tmpBuffer,
                                        const LocalTensor<int16_t> &dst) {

    UnaryRepeatParams s42f16unaryParams;
    s42f16unaryParams.srcRepStride =
        (ONE_REPEAT_BYTES / 2) / 2 / ONE_BLOCK_SIZE;
    UnaryRepeatParams f162i16unaryParams;
    BinaryRepeatParams f162f16binaryParams;
    f162f16binaryParams.src1RepStride = 0;
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(0, src.GetSize());
    Cast<half, SrcType, false>(dst.template ReinterpretCast<half>(), src,
                               RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                               REPEAT_PLACE_HOLDER, s42f16unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<int16_t, half, false>(dst, dst.template ReinterpretCast<half>(),
                               RoundMode::CAST_RINT, MASK_PLACEHOLDER,
                               REPEAT_PLACE_HOLDER, f162i16unaryParams);
    PipeBarrier<PIPE_V>();
    ShiftLeft<int16_t, false>(dst, dst, 9, MASK_PLACEHOLDER,
                              REPEAT_PLACE_HOLDER, f162i16unaryParams);
    PipeBarrier<PIPE_V>();
    Duplicate<uint16_t>(tmpBuffer.template ReinterpretCast<uint16_t>(), 0x8E00,
                        ONE_REPEAT_BYTES / 2);
    PipeBarrier<PIPE_V>();
    And<uint16_t, false>(dst.template ReinterpretCast<uint16_t>(),
                         dst.template ReinterpretCast<uint16_t>(),
                         tmpBuffer.template ReinterpretCast<uint16_t>(),
                         MASK_PLACEHOLDER, src.GetSize() / 128,
                         f162f16binaryParams);
    PipeBarrier<PIPE_V>();
    Duplicate<uint16_t>(tmpBuffer.template ReinterpretCast<uint16_t>(), 0x7400,
                        ONE_REPEAT_BYTES / 2);
    PipeBarrier<PIPE_V>();
    Mul<half, false>(dst.template ReinterpretCast<half>(),
                     dst.template ReinterpretCast<half>(),
                     tmpBuffer.template ReinterpretCast<half>(),
                     MASK_PLACEHOLDER, src.GetSize() / 128,
                     f162f16binaryParams);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename SrcType>
__aicore__ inline void preCastFP8Weight(const LocalTensor<SrcType> &src,
                                        const LocalTensor<uint8_t> &tmpBuff,
                                        const LocalTensor<int16_t> &dst,
                                        const uint32_t tensorSize) {
    UnaryRepeatParams s82f16unaryParam;
    s82f16unaryParam.srcRepStride = (ONE_REPEAT_BYTES / 2) / ONE_BLOCK_SIZE;
    UnaryRepeatParams f162i16unaryParam;
    BinaryRepeatParams f162f16binaryParam;
    f162f16binaryParam.src1RepStride = 0;
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(0, tensorSize);
    Cast<half, SrcType, false>(dst.template ReinterpretCast<half>(), src,
                               RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                               REPEAT_PLACE_HOLDER, s82f16unaryParam);
    PipeBarrier<PIPE_V>();
    Cast<int16_t, half, false>(dst, dst.template ReinterpretCast<half>(),
                               RoundMode::CAST_RINT, MASK_PLACEHOLDER,
                               REPEAT_PLACE_HOLDER, f162i16unaryParam);
    PipeBarrier<PIPE_V>();
    ShiftLeft<int16_t, false>(dst, dst, 7, MASK_PLACEHOLDER,
                              REPEAT_PLACE_HOLDER, f162i16unaryParam);
    PipeBarrier<PIPE_V>();
    Duplicate<uint16_t>(tmpBuff.template ReinterpretCast<uint16_t>(), 0xBFFF,
                        ONE_REPEAT_BYTES / 2);
    PipeBarrier<PIPE_V>();
    And<uint16_t, false>(dst.template ReinterpretCast<uint16_t>(),
                         dst.template ReinterpretCast<uint16_t>(),
                         tmpBuff.template ReinterpretCast<uint16_t>(),
                         MASK_PLACEHOLDER, tensorSize / 128,
                         f162f16binaryParam);
    PipeBarrier<PIPE_V>();
    Duplicate<uint16_t>(tmpBuff.template ReinterpretCast<uint16_t>(), 0x5C00,
                        ONE_REPEAT_BYTES / 2);
    PipeBarrier<PIPE_V>();
    Mul<half, false>(dst.template ReinterpretCast<half>(),
                     dst.template ReinterpretCast<half>(),
                     tmpBuff.template ReinterpretCast<half>(), MASK_PLACEHOLDER,
                     tensorSize / 128, f162f16binaryParam);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void
preCastHalfWeight(const LocalTensor<half> &src, const LocalTensor<uint8_t> &dst,
                  const UnaryRepeatParams &f162f32unaryParams,
                  const uint32_t tensorSize) {

    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(0, tensorSize);
    Cast<float, half, false>(dst.template ReinterpretCast<float>(), src,
                             RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                             REPEAT_PLACE_HOLDER, f162f32unaryParams);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename SrcType, typename DstType>
__aicore__ inline void dequantProcess(
    const LocalTensor<DstType> &dst, const LocalTensor<uint8_t> &scaleTmpBuffer,
    const LocalTensor<uint8_t> &weightTmpBuffer,
    const UnaryRepeatParams &f322f16unaryParams,
    const BinaryRepeatParams &binaryParams, const uint32_t tensorSize) {

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tensorSize);
    Mul<float, false>(weightTmpBuffer.template ReinterpretCast<float>(),
                      weightTmpBuffer.template ReinterpretCast<float>(),
                      scaleTmpBuffer.template ReinterpretCast<float>(),
                      MASK_PLACEHOLDER, REPEAT_PLACE_HOLDER, binaryParams);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename SrcType, typename DstType>
__aicore__ inline void
dequantProcess(const LocalTensor<DstType> &src, const float scale,
               const LocalTensor<uint8_t> &weightBuffer,
               const LocalTensor<uint8_t> &scaleTmpBuffer,
               const int32_t tensorSize) {

    UnaryRepeatParams f162f32unaryParam;
    f162f32unaryParam.srcRepStride = ONE_REPEAT_BLOCKS / 2;
    UnaryRepeatParams f322f16unaryParam;
    f322f16unaryParam.dstRepStride = ONE_REPEAT_BLOCKS / 2;
    BinaryRepeatParams f32binaryParam;
    f32binaryParam.src1RepStride = 0;
    // UnaryRepeatParams f32unaryParam;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tensorSize);
    Cast<float, half, false>(weightBuffer.template ReinterpretCast<float>(),
                             src.template ReinterpretCast<half>(),
                             RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                             REPEAT_PLACE_HOLDER, f162f32unaryParam);
    PipeBarrier<PIPE_V>();
    Duplicate<float>(scaleTmpBuffer.template ReinterpretCast<float>(), scale,
                     ONE_REPEAT_BYTES / 4);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tensorSize);
    Mul<float, false>(weightBuffer.template ReinterpretCast<float>(),
                      weightBuffer.template ReinterpretCast<float>(),
                      scaleTmpBuffer.template ReinterpretCast<float>(),
                      //   scale,
                      MASK_PLACEHOLDER, REPEAT_PLACE_HOLDER, f32binaryParam);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename DstType>
__aicore__ inline void InnerProductProcessFP4(const LocalTensor<float> &w,
                                              const LocalTensor<DstType> &x,
                                              const LocalTensor<float> &tmpBuff,
                                              const LocalTensor<float> &dst,
                                              const uint32_t tensorSize,
                                              const uint32_t weightOffset) {
    LocalTensor<float> sharedTmpBuffer = tmpBuff[64 * 8];
    LocalTensor<uint8_t> tmpSpace =
        sharedTmpBuffer.template ReinterpretCast<uint8_t>()[1024];
    uint32_t weightOffset_1_8 = weightOffset / 8;
    uint32_t Shape1[2], Shape2[2];
    Shape1[0] = Shape2[0] = 64;
    Shape1[1] = 1;
    Shape2[1] = 8;
    BinaryRepeatParams f162f16binaryParam;
    f162f16binaryParam.src0RepStride = 1;
    f162f16binaryParam.src1RepStride = 0;
    f162f16binaryParam.src0BlkStride = 130;
    f162f16binaryParam.dstRepStride = 1;
    f162f16binaryParam.dstBlkStride = 130;
    BinaryRepeatParams f32binaryParam;
    BroadCast<int16_t, 2, 1>(
        sharedTmpBuffer.template ReinterpretCast<int16_t>(),
        x.template ReinterpretCast<int16_t>(), Shape2, Shape1, tmpSpace);
    PipeBarrier<PIPE_V>();
    Cast<float, DstType>(tmpBuff,
                         sharedTmpBuffer.template ReinterpretCast<DstType>(),
                         RoundMode::CAST_NONE, 64 * 8);
    PipeBarrier<PIPE_V>();
    const LocalTensor<float> w_ = w[8320];
    const LocalTensor<float> x_ = tmpBuff[64];
    const LocalTensor<float> sharedTmpBuffer_ = sharedTmpBuffer[8320];
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tensorSize / 4);
    for (int i = 0; i < 4; i++) {
        Mul<float, false>(sharedTmpBuffer[256 * i], w[256 * i],
                          tmpBuff[128 * i], MASK_PLACEHOLDER,
                          REPEAT_PLACE_HOLDER, f162f16binaryParam);
    }
    for (int i = 0; i < 4; i++) {
        Mul<float, false>(sharedTmpBuffer_[256 * i], w_[256 * i], x_[128 * i],
                          MASK_PLACEHOLDER, REPEAT_PLACE_HOLDER,
                          f162f16binaryParam);
    }
    PipeBarrier<PIPE_V>();
    for (int i = 8; i > 0; i /= 2) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, weightOffset_1_8 * i);
        Add<float, false>(sharedTmpBuffer,
                          sharedTmpBuffer[weightOffset_1_8 * i],
                          sharedTmpBuffer, MASK_PLACEHOLDER,
                          REPEAT_PLACE_HOLDER, f32binaryParam);
        PipeBarrier<PIPE_V>();
    }
    for (int i = 2; i > 0; i /= 2) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, 256 * i);
        Add<float, false>(sharedTmpBuffer, sharedTmpBuffer[256 * i],
                          sharedTmpBuffer, MASK_PLACEHOLDER,
                          REPEAT_PLACE_HOLDER, f32binaryParam);
        PipeBarrier<PIPE_V>();
    }
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, 256);
    Add<float, false>(dst, sharedTmpBuffer, dst, MASK_PLACEHOLDER,
                      REPEAT_PLACE_HOLDER, f32binaryParam);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename DstType>
__aicore__ inline void InnerProductProcessFP8(const LocalTensor<float> &w,
                                              const LocalTensor<DstType> &x,
                                              const LocalTensor<float> &tmpBuff,
                                              const LocalTensor<float> &dst,
                                              const uint32_t weightOffset) {
    LocalTensor<float> sharedTmpBuffer = tmpBuff[128 * 8];
    LocalTensor<uint8_t> tmpSpace =
        sharedTmpBuffer.template ReinterpretCast<uint8_t>();
    uint32_t weightOffset_1_64 = weightOffset / 64;
    uint32_t Shape1[2], Shape2[2];
    Shape1[0] = Shape2[0] = 128;
    Shape1[1] = 1;
    Shape2[1] = 8;
    BinaryRepeatParams f322f32binaryParam;
    f322f32binaryParam.src0RepStride = 16;
    f322f32binaryParam.src1BlkStride = 0;
    f322f32binaryParam.src1RepStride = 1;
    f322f32binaryParam.dstRepStride = 16;
    BinaryRepeatParams f32binaryParam;
    BroadCast<int16_t, 2, 1>(
        sharedTmpBuffer.template ReinterpretCast<int16_t>(),
        x.template ReinterpretCast<int16_t>(), Shape2, Shape1, tmpSpace);
    PipeBarrier<PIPE_V>();
    Cast<float, DstType>(tmpBuff,
                         sharedTmpBuffer.template ReinterpretCast<DstType>(),
                         RoundMode::CAST_NONE, 128 * 8);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, weightOffset);
    // printf("FROM 323: %d \n", weightOffset);
    Mul<float, false>(sharedTmpBuffer, w, tmpBuff, MASK_PLACEHOLDER,
                      REPEAT_PLACE_HOLDER, f322f32binaryParam);
    Mul<float, false>(sharedTmpBuffer[64], w[64], tmpBuff, MASK_PLACEHOLDER,
                      REPEAT_PLACE_HOLDER, f322f32binaryParam);
    PipeBarrier<PIPE_V>();
    for (int i = 64; i > 0; i /= 2) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, weightOffset_1_64 * i);
        Add<float, false>(sharedTmpBuffer,
                          sharedTmpBuffer[weightOffset_1_64 * i],
                          sharedTmpBuffer, MASK_PLACEHOLDER,
                          REPEAT_PLACE_HOLDER, f32binaryParam);
        PipeBarrier<PIPE_V>();
    }
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, 128);
    Add<float, false>(dst, sharedTmpBuffer, dst, MASK_PLACEHOLDER,
                      REPEAT_PLACE_HOLDER, f32binaryParam);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename DstType, typename ScaleType>
__aicore__ inline void
SoftGEMV(const LocalTensor<int4b_t> &w, const LocalTensor<float> &acc,
         const LocalTensor<DstType> &x, const LocalTensor<ScaleType> &scale,
         const LocalTensor<uint8_t> &sharedTmpBuffer,
         const DeQuantShapeInfo &shapeInfo = {}) {

    const uint32_t tensorSize = shapeInfo.baseN * shapeInfo.baseK / 2;
    const uint32_t scaleSize =
        shapeInfo.baseN * shapeInfo.baseK / shapeInfo.perGroupSize;
    const uint32_t weightOffset = tensorSize / 64 * 65;
    auto scaleBuffer = sharedTmpBuffer;
    auto weightBuffer = sharedTmpBuffer[scale.GetSize() * sizeof(float)];
    auto tmpBuff =
        weightBuffer.template ReinterpretCast<float>()[weightOffset * 2];
    UnaryRepeatParams f162f32unaryParams;
    f162f32unaryParams.srcRepStride = ONE_REPEAT_BLOCKS / 2;
    f162f32unaryParams.dstRepStride = 1;
    f162f32unaryParams.dstBlkStride = 130;
    BinaryRepeatParams binaryParams;
    binaryParams.src0RepStride = 1;
    binaryParams.src1RepStride = 1;
    binaryParams.src0BlkStride = 130;
    binaryParams.src1BlkStride = 0;
    binaryParams.dstRepStride = 1;
    binaryParams.dstBlkStride = 130;
    UnaryRepeatParams f322f16unaryParams;
    f322f16unaryParams.dstRepStride = ONE_REPEAT_BLOCKS / 2;
    preCastScale<ScaleType, float>(scale, scaleBuffer);
    preCastFP4Weight<int4b_t, half>(
        w, weightBuffer, tmpBuff.template ReinterpretCast<int16_t>());
    preCastHalfWeight(tmpBuff.template ReinterpretCast<half>(), weightBuffer,
                      f162f32unaryParams, tensorSize);
    preCastHalfWeight(tmpBuff.template ReinterpretCast<half>()[tensorSize],
                      weightBuffer[weightOffset * sizeof(float)],
                      f162f32unaryParams, tensorSize);
    dequantProcess<half, DstType>(tmpBuff.template ReinterpretCast<DstType>(),
                                  scaleBuffer, weightBuffer, f322f16unaryParams,
                                  binaryParams, tensorSize);
    dequantProcess<half, DstType>(
        tmpBuff.template ReinterpretCast<DstType>()[weightOffset], scaleBuffer,
        weightBuffer[weightOffset * sizeof(float)], f322f16unaryParams,
        binaryParams, tensorSize);
    for (int i = 0; i < shapeInfo.M; i++) {
        InnerProductProcessFP4<DstType>(
            weightBuffer.template ReinterpretCast<float>(), x[64 * i], tmpBuff,
            acc[256 * i], tensorSize, weightOffset);
    }
}

template <typename DstType>
__aicore__ inline void
SoftGEMV(const LocalTensor<int8_t> &w, const LocalTensor<float> &acc,
         const LocalTensor<DstType> &x, const float scale,
         const LocalTensor<uint8_t> &sharedTmpBuffer,
         const DeQuantShapeInfo &shapeInfo = {}) {

    const int32_t tensorSize = w.GetSize();
    auto weightBuffer = sharedTmpBuffer.template ReinterpretCast<float>();
    auto tmpBuff = weightBuffer[tensorSize];
    preCastFP8Weight<int8_t>(w, sharedTmpBuffer,
                             tmpBuff.template ReinterpretCast<int16_t>(),
                             tensorSize);
    dequantProcess<float, bfloat16_t>(
        tmpBuff.template ReinterpretCast<DstType>(), scale,
        weightBuffer.template ReinterpretCast<uint8_t>(),
        tmpBuff.template ReinterpretCast<uint8_t>(), tensorSize);
    for (int i = 0; i < shapeInfo.M; i++) {
        InnerProductProcessFP8<DstType>(weightBuffer, x[128 * i], tmpBuff,
                                        acc[128 * i], tensorSize / 2);
    }
}
} // namespace GROUPED_GEMV
#endif