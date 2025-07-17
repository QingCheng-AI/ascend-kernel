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
 * \file grouped_matmul_utils.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_UTILS_H
#define ASCENDC_GROUPED_MATMUL_UTILS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

// #define GMM_ANTI_QUANT
#define DTYPE_X bfloat16_t
#define DTYPE_BIAS float
#define DTYPE_Y bfloat16_t
#define ORIG_DTYPE_X DT_BF16

#if ORIG_DTYPE_WEIGHT == DT_UINT8
#define DTYPE_WEIGHT int8_t
#else
#define DTYPE_WEIGHT int4b_t
#define ORIG_DTYPE_WEIGHT DT_INT4
#define DTYPE_ANTIQUANT_SCALE bfloat16_t
#endif

#define ORIG_DTYPE_Y DT_BF16
#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) &&                     \
    defined(ORIG_DTYPE_Y) && defined(DT_INT8) && defined(DT_BF16) &&           \
    defined(DT_INT4)
#if ORIG_DTYPE_X == ORIG_DTYPE_WEIGHT
#if ORIG_DTYPE_X == DT_INT8
#if ORIG_DTYPE_Y == DT_BF16
#define GMM_QUANT_BF16
#define MM_DTYPE_Y int32_t
#elif ORIG_DTYPE_Y == DT_FLOAT16
#define GMM_QUANT_FLOAT16
#define MM_DTYPE_Y int32_t
#elif ORIG_DTYPE_Y == DT_INT32
#define GMM_QUANT_INT32
#else
#define GMM_QUANT_INT8
#endif
#else
#define GMM_FLOAT
#endif
#else
#define GMM_ANTI_QUANT
#if ORIG_DTYPE_X == DT_INT8 && ORIG_DTYPE_WEIGHT == DT_INT4
#define GMM_ANTI_QUANT_A8W4_MSD
#if ORIG_DTYPE_Y == DT_BF16
#define GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
#else
#define GMM_ANTI_QUANT_A8W4_MSD_OUT_FP16
#endif
#define MM_DTYPE_Y int32_t
#else
#define GMM_ANTI_QUANT
#endif
#endif
#endif

#if defined(DTYPE_Y) && !defined(MM_DTYPE_Y)
#define MM_DTYPE_Y DTYPE_Y
#endif

#if defined(CONST_TILING)
#define TILING_TYPE const int32_t
#else
#define TILING_TYPE __gm__ int32_t
#endif

#if defined(CONST_TILING)
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling)           \
    GET_TILING_DATA_MEMBER(GMMTilingData, member, obj, tiling);                \
    const int32_t *(var) = (const int32_t *)((const uint8_t *)&obj);
#else
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling)           \
    size_t offset##var = (size_t)(&((tilingType *)0)->member);                 \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((tiling) + (offset##var));
#endif

namespace GROUPED_MATMUL {
using namespace AscendC;

constexpr uint32_t INT8_BITS = 8;           // a int8 number has 8 bits
constexpr int32_t MKN_LIST_LEN = 128;       // 128: predefined array legnth
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32; // 32: a block has 32 bytes data
constexpr uint32_t UB_BLOCK_DOUBLE_UNIT_SIZE =
    64; // 64: a block has 64 bytes data
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE =
    UB_BLOCK_UNIT_SIZE / 2; // 2: a float16 data has two bytes
constexpr MatmulConfig NZ_CFG_MDL =
    GetMDLConfig(false, false, 0, true, false, false, false);
constexpr MatmulConfig matmulCFGUnitFlag{
    false, false, true, 0, 0, 0, false, false, false, false,
    false, 0,     0,    0, 0, 0, 0,     0,     true};

constexpr uint64_t SYNC_AIV_AIC_FLAG = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG = 5;
constexpr uint64_t SYNC_MODE2 = 2;

template <class AT_, class BT_, class CT_, class BiasT_,
          const auto &MM_CFG = CFG_MDL>
struct MMType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::Matmul<AT, BT, CT, BiasT, MM_CFG>;
};

template <class AT_, class BT_, class CT_, class BiasT_,
          const auto &MM_CFG = CFG_MDL>
struct MMImplType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MM_CFG>;
};

template <typename T> __aicore__ inline T GreatestCommonDivisor(T a, T b) {
    T c = a;
    if (a < b) {
        a = b;
        b = c;
    }
    while (b != 0) {
        c = a;
        a = b;
        b = c % b;
    }
    return a;
}

template <typename T> __aicore__ inline T LeastCommonMultiple(T a, T b) {
    return a * b / GreatestCommonDivisor(a, b);
}

template <typename T> __aicore__ inline T Max(T a, T b) {
    return a > b ? a : b;
}

template <typename T> __aicore__ inline T Min(T a, T b) {
    return a > b ? b : a;
}

template <uint32_t base, typename T = uint32_t>
__aicore__ inline T AlignUp(T a) {
    return (a + base - 1) / base * base;
}

template <typename T> __aicore__ inline T AlignUp(T a, T base) {
    return (a + base - 1) / base * base;
}

template <typename T> __aicore__ inline T AlignDown(T a, T base) {
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

template <> __aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a) {
    // to be Multiple of 4, result should be in a format of b(xxxx,x100).
    // This means last two bits should be zero, requiring that
    // result = num & b(1111,1100) = num & (~3).
    // &(~3) operator may reduces num into the range [num, num - 3].
    // As the result should be no less than a (result >= a), it means num - 3 >=
    // a in the worst case. In this case, num >= a+3. On the other hand, num
    // should also be less then a+4, otherwise, the result will not be least
    // multiple of 4 for 3. In other cases like [num, num - 2], num = a + 3 also
    // satisfies the goal condition.
    return (a + 3) & ~3; // & ~3: set last two bits of (a+3) to be zero
}

template <> __aicore__ inline uint32_t AlignUp<8, uint32_t>(uint32_t a) {
    // In general, if we want to get the least multiple of b (b is the power of
    // 2) for a, it comes to a conclusion from the above comment: result = (a +
    // (b - 1)) & (~b)
    return (a + 7) & ~7; // & ~7: set last four bits of (a+7) to be zero
}

template <> __aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a) {
    // In general, if we want to get the least multiple of b (b is the power of
    // 2) for a, it comes to a conclusion from the above comment: result = (a +
    // (b - 1)) & (~b)
    return (a + 15) & ~15; // & ~15: set last four bits of (a+15) to be zero
}

template <> __aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a) {
    // refer to the above comments.
    return (a + 31) & ~31; // & ~31: set last five bits of (a+31) to be zero}
}

template <typename T>
__aicore__ inline __gm__ T *GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset =
        *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t *retPtr = dataAddr; // + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T *>((retPtr + index));
}

__aicore__ inline int32_t
GetSplitValueFromGroupList(uint32_t groupIdx, int32_t &preOffset,
                           const GMMBaseParams *__restrict &gmmBaseParams,
                           const GlobalTensor<int64_t> &groupListGm) {
    int32_t splitValue = 0;
    if (likely(gmmBaseParams->groupType != -1)) { // -1: no  need to split
        if (gmmBaseParams->groupListType == 0) {
            int32_t offset =
                static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        }
    }
    return splitValue;
}

template <typename T> __aicore__ inline constexpr uint32_t GetTypeBits() {
    if constexpr (IsSameType<T, int4b_t>::value) {
        return 4; // 4: int4 bits number
    }
    return sizeof(T) * INT8_BITS;
}

} // namespace GROUPED_MATMUL

#endif // ASCENDC_GROUPED_MATMUL_UTILS_H
