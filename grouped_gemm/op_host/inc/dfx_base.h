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

/*!
 * \file dfx_base.h
 * \brief 外部模块不应直接引用本头文件
 */

#pragma once

#include <cstdint>
#include <exe_graph/runtime/infer_datatype_context.h>
#include <exe_graph/runtime/infer_shape_context.h>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
#include <experiment/metadef/common/util/error_manager/error_manager.h>
#include <securec.h>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <toolchain/slog.h>
#include <unistd.h>

namespace ops {
namespace utils {

class LogBase {
  public:
    static constexpr const int MAX_LOG_LEN = 16000;
    static constexpr const int MSG_HDR_LEN = 200;

    static inline uint64_t GetTid() {
        return static_cast<uint64_t>(syscall(__NR_gettid));
    }

    static inline const char *GetStr(const std::string &str) {
        return str.c_str();
    }

    static inline const char *GetStr(const char *str) { return str; }

    static inline const std::string &GetOpInfo(const std::string &str) {
        return str;
    }

    static inline const char *GetOpInfo(const char *str) { return str; }

    static inline std::string GetOpInfo(const gert::TilingContext *context) {
        return GetOpInfoFromContext(context);
    }

    static inline std::string
    GetOpInfo(const gert::TilingParseContext *context) {
        return GetOpInfoFromContext(context);
    }

    static inline std::string
    GetOpInfo(const gert::InferShapeContext *context) {
        return GetOpInfoFromContext(context);
    }

    static inline std::string
    GetOpInfo(const gert::InferDataTypeContext *context) {
        return GetOpInfoFromContext(context);
    }

  private:
    template <class T>
    static inline std::string GetOpInfoFromContext(T context) {
        if (context == nullptr) {
            return "nil:nil";
        }
        std::string opInfo =
            context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
        opInfo += ":";
        opInfo +=
            context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
        return opInfo;
    }
};

} // namespace utils

template <typename T> std::string Shape2String(const T &shape) {
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}
} // namespace ops

// // 使用本宏前需预定义标识子模块名称的 OPS_UTILS_LOG_SUB_MOD_NAME
// // 如: #define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING" 或通过 CMake
// 传递预定义宏 #define OPS_UTILS_LOG_SUB_MOD_NAME "OP_TILING" #define
// OPS_UTILS_LOG_PACKAGE_TYPE "[Custom]"
// #define OPS_LOG_STUB(MOD_ID, LOG_LEVEL, OPS_DESC, FMT, ...) \
//     DlogSub(static_cast<int>(MOD_ID), (OPS_UTILS_LOG_SUB_MOD_NAME),
//     (LOG_LEVEL), "%s[%s][%lu] OpName:[%s] " #FMT,      \
//             (OPS_UTILS_LOG_PACKAGE_TYPE), __FUNCTION__,
//             ops::utils::LogBase::GetTid(),                                 \
//             ops::utils::LogBase::GetStr(ops::utils::LogBase::GetOpInfo(OPS_DESC)),
//             ##__VA_ARGS__)
#define OPS_LOG_STUB(MOD_ID, LOG_LEVEL, OPS_DESC, FMT, ...)                    \
    do {                                                                       \
        char msgbuf[ops::utils::LogBase::MAX_LOG_LEN];                         \
        snprintf(msgbuf, sizeof(msgbuf), FMT, ##__VA_ARGS__);                  \
        std::cerr << "[" << __FUNCTION__ << "]["                               \
                  << ops::utils::LogBase::GetTid() << "] OpName:["             \
                  << ops::utils::LogBase::GetStr(                              \
                         ops::utils::LogBase::GetOpInfo(OPS_DESC))             \
                  << "] " << msgbuf << std::endl;                              \
    } while (0)

#define OPS_LOG_STUB_IF(COND, LOG_FUNC, EXPR)                                  \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, \
                  "condition should be bool");                                 \
    do {                                                                       \
        if (__builtin_expect((COND), 0)) {                                     \
            LOG_FUNC;                                                          \
            EXPR;                                                              \
        }                                                                      \
    } while (0)

#define OPS_INNER_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)                   \
    do {                                                                       \
        OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__);            \
        REPORT_INNER_ERROR(ERR_CODE_STR, FMT, ##__VA_ARGS__);                  \
    } while (0)

#define OPS_CALL_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)                    \
    do {                                                                       \
        OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__);            \
        REPORT_CALL_ERROR(ERR_CODE_STR, FMT, ##__VA_ARGS__);                   \
    } while (0)

#define OPS_LOG_STUB_D(OPS_DESC, FMT, ...)                                     \
    OPS_LOG_STUB(OP, DLOG_DEBUG, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_I(OPS_DESC, FMT, ...)                                     \
    OPS_LOG_STUB(OP, DLOG_INFO, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_W(OPS_DESC, FMT, ...)                                     \
    OPS_LOG_STUB(OP, DLOG_WARN, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_E(OPS_DESC, FMT, ...)                                     \
    OPS_LOG_STUB(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__)
#define OPS_LOG_STUB_EVENT(OPS_DESC, FMT, ...)                                 \
    OPS_LOG_STUB(OP, DLOG_EVENT, OPS_DESC, FMT, ##__VA_ARGS__)

#define OPS_LOG_STUB_FULL(LEVEL, OPS_DESC, FMT, ...)                           \
    do {                                                                       \
        if (0 == CheckLogLevel(OP, (LEVEL))) {                                 \
            break;                                                             \
        }                                                                      \
        char msgbufxyz[ops::utils::LogBase::MAX_LOG_LEN];                      \
        size_t msgmaxlen = (MSG_LENGTH - ops::utils::LogBase::MSG_HDR_LEN);    \
        int rettmp = snprintf_s(msgbufxyz, sizeof(msgbufxyz),                  \
                                sizeof(msgbufxyz) - 1, FMT, ##__VA_ARGS__);    \
        if (rettmp == -1) {                                                    \
            msgbufxyz[sizeof(msgbufxyz) - 1] = '\0';                           \
        }                                                                      \
        size_t msglength = std::strlen(msgbufxyz);                             \
        if (msglength < msgmaxlen) {                                           \
            OPS_LOG_STUB(OP, (LEVEL), (OPS_DESC), "%s", msgbufxyz);            \
            break;                                                             \
        }                                                                      \
        char *msgchunkbegin = msgbufxyz;                                       \
        char *msgchunkend = nullptr;                                           \
        while (msgchunkbegin < msgbufxyz + msglength) {                        \
            if (msgchunkbegin[0] == '\n') {                                    \
                OPS_LOG_STUB(OP, (LEVEL), (OPS_DESC), "");                     \
                msgchunkbegin += 1;                                            \
                continue;                                                      \
            }                                                                  \
            msgchunkend = std::strchr(msgchunkbegin, '\n');                    \
            if (msgchunkend == nullptr) {                                      \
                msgchunkend = msgchunkbegin + std::strlen(msgchunkbegin);      \
            }                                                                  \
            while (msgchunkend > msgchunkbegin) {                              \
                std::string msgchunk(                                          \
                    msgchunkbegin,                                             \
                    std::min(msgmaxlen, static_cast<size_t>(msgchunkend -      \
                                                            msgchunkbegin)));  \
                OPS_LOG_STUB(OP, (LEVEL), (OPS_DESC), "%s", msgchunk.c_str()); \
                msgchunkbegin += msgchunk.size();                              \
            }                                                                  \
            msgchunkbegin += 1;                                                \
        }                                                                      \
    } while (0)
