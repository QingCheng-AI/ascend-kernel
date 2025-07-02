/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

/* !
 * \file antiquant_impl.h
 * \brief
 */
#ifndef ASCENDC_ANTIQUANT_IMPL_H
#define ASCENDC_ANTIQUANT_IMPL_H
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tensor.h"

namespace GROUPED_MATMUL {

using namespace AscendC;
constexpr uint32_t ONE_REPEAT_BLOCKS = 8;
constexpr uint32_t ONE_REPEAT_BYTES = 256;
constexpr uint32_t ONE_BLOCK_SIZE = 32;
constexpr uint32_t REPEAT_PLACE_HOLDER = 1;
struct AntiQuantShapeInfo {
    uint32_t baseN{256};
    uint32_t baseK{64};
    uint32_t perGroupSize{16};
    uint32_t N{0};
    uint32_t K{0};
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
__aicore__ inline void antiquantProcess(
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
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tensorSize / 64 * 65);
    Cast<bfloat16_t, float, false>(
        dst.template ReinterpretCast<bfloat16_t>(),
        weightTmpBuffer.template ReinterpretCast<float>(),
        RoundMode::CAST_ROUND, MASK_PLACEHOLDER, REPEAT_PLACE_HOLDER,
        f322f16unaryParams);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename SrcType, typename DstType, typename ScaleType>
__aicore__ inline void AntiQuant2(const LocalTensor<SrcType> &src,
                                  const LocalTensor<DstType> &dst,
                                  const LocalTensor<ScaleType> &scale,
                                  const LocalTensor<uint8_t> &sharedTmpBuffer,
                                  const AntiQuantShapeInfo &shapeInfo = {}) {

    const uint32_t tensorSize = shapeInfo.baseN * shapeInfo.baseK / 2;
    const uint32_t scaleSize =
        shapeInfo.baseN * shapeInfo.baseK / shapeInfo.perGroupSize;
    const uint32_t weightOffset = tensorSize / 64 * 65;
    auto scaleBuffer = sharedTmpBuffer;
    auto weightBuffer = sharedTmpBuffer[scaleSize * sizeof(float)];
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
    preCastFP4Weight<SrcType, half>(src, weightBuffer,
                                    dst.template ReinterpretCast<int16_t>());
    preCastHalfWeight(dst.template ReinterpretCast<half>(), weightBuffer,
                      f162f32unaryParams, tensorSize);
    preCastHalfWeight(dst.template ReinterpretCast<half>()[tensorSize],
                      weightBuffer[weightOffset * sizeof(float)],
                      f162f32unaryParams, tensorSize);
    antiquantProcess<half, DstType>(dst, scaleBuffer, weightBuffer,
                                    f322f16unaryParams, binaryParams,
                                    tensorSize);
    antiquantProcess<half, DstType>(dst[weightOffset], scaleBuffer,
                                    weightBuffer[weightOffset * sizeof(float)],
                                    f322f16unaryParams, binaryParams,
                                    tensorSize);
}
} // namespace GROUPED_MATMUL
#endif