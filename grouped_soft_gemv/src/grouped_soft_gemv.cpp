#include "grouped_soft_gemv.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>

namespace grouped_soft_gemv {

#define CHECK_RET(cond, return_expr)                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            return_expr;                                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define LOG_PRINT(message, ...)                                                \
    do {                                                                       \
        printf(message, ##__VA_ARGS__);                                        \
    } while (0)

void get_stride(at::Tensor t, int64_t *stride) {
    for (int i = 0; i < t.dim(); i++) {
        stride[i] = t.stride(i);
    }
}

void get_size(at::Tensor t, int64_t *sizes) {
    for (int i = 0; i < t.dim(); i++) {
        sizes[i] = t.size(i);
    }
}

// 用于Aten和Acl的类型转换
#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)                            \
    _(at::ScalarType::Byte, ACL_UINT8)                                         \
    _(at::ScalarType::Char, ACL_INT8)                                          \
    _(at::ScalarType::Short, ACL_INT16)                                        \
    _(at::ScalarType::Int, ACL_INT32)                                          \
    _(at::ScalarType::Long, ACL_INT64)                                         \
    _(at::ScalarType::Half, ACL_FLOAT16)                                       \
    _(at::ScalarType::Float, ACL_FLOAT)                                        \
    _(at::ScalarType::Double, ACL_DOUBLE)                                      \
    _(at::ScalarType::ComplexHalf, ACL_DT_UNDEFINED)                           \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)                             \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)                           \
    _(at::ScalarType::Bool, ACL_BOOL)                                          \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                                 \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                                \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                                \
    _(at::ScalarType::BFloat16, ACL_BF16)                                      \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                              \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                              \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)                             \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable
    [static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
        AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

aclTensor *createTensor(at::Tensor t, const int dtype) {
    int64_t dim = t.dim();
    int64_t stride[dim], sizes[dim];
    get_stride(t, stride);
    get_size(t, sizes);
    aclDataType acl_data_type =
        kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(
            t.scalar_type())];
    if (dtype == 0)
        return aclCreateTensor(sizes, dim, acl_data_type, stride, 0,
                               ACL_FORMAT_ND, sizes, dim, t.data_ptr());
    if (dtype == 1) {
        sizes[dim - 1] *= 2;
        for (int i = 0; i < dim - 1; i++) {
            stride[i] *= 2;
        }
        return aclCreateTensor(sizes, dim, ACL_INT4, stride, 0, ACL_FORMAT_ND,
                               sizes, dim, t.data_ptr());
    }
}

std::basic_string<char> itostring(uint64_t &int64) {
    uint32_t *tmp32 = reinterpret_cast<uint32_t *>(&int64);
    return std::to_string(tmp32[0]) + std::to_string(tmp32[1]);
}

class WorkspaceManager {
  private:
    static std::map<std::basic_string<char>, void *> workspaceManager;

  public:
    WorkspaceManager() {}
    static void addToMap(std::basic_string<char> id, void *value) {
        workspaceManager[id] = value;
    }

    static void *getFromMap(uint64_t workspaceSize) {
        std::basic_string<char> key = itostring(workspaceSize);
        auto it = workspaceManager.find(key);
        if (it != workspaceManager.end()) {
            return it->second;
        }
        void *workspaceAddr;
        if (workspaceSize > 0) {
            auto retn = aclrtMalloc(&workspaceAddr, workspaceSize,
                                    ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(
                retn == ACL_SUCCESS,
                LOG_PRINT("allocate workspace failed. ERROR: %d\n", retn));
        }
        addToMap(key, workspaceAddr);
        return workspaceAddr;
    }
};

std::map<std::basic_string<char>, void *> WorkspaceManager::workspaceManager;

void GroupedSoftGemv(at::Tensor x, at::Tensor weight, at::Tensor scale,
                     at::Tensor groupList, char *computeType,
                     at::Tensor output) {
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    aclTensor *x_ = createTensor(x, 0);
    CHECK_RET(
        !(strcmp(computeType, "fp4") && strcmp(computeType, "fp8")),
        LOG_PRINT(
            "Invalid compute type: %s. Only suppourt \"fp4\" or \"fp8\"\n",
            computeType));
    aclTensor *weight_ = !strcmp(computeType, "fp4") ? createTensor(weight, 1)
                                                     : createTensor(weight, 0);
    aclTensor *scale_ = createTensor(scale, 0);
    aclTensor *groupList_ = createTensor(groupList, 0);
    aclTensor *output_ = createTensor(output, 0);
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor;
    auto ret = aclnnGroupedSoftGemvGetWorkspaceSize(
        x_, weight_, scale_, groupList_, output_, &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedSoftGemvGetWorkspaceSize failed. ERROR: %d\n",
                  ret));
    // 根据第一段接口计算出的workspaceSize申请device内存并缓存
    workspaceAddr = WorkspaceManager().getFromMap(workspaceSize);
    ret = aclnnGroupedSoftGemv(workspaceAddr, workspaceSize, executor,
                               acl_stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGroupedSoftGemv failed. ERROR: %d\n", ret));

    return;
}
} // namespace grouped_soft_gemv