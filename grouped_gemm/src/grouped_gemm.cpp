#include "grouped_gemm.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>

namespace grouped_gemm {
#define CHECK_RET(cond, return_expr)                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            return_expr;                                                       \
        }                                                                      \
    } while (0)

#define LOG_PRINT(message, ...)                                                \
    do {                                                                       \
        printf(message, ##__VA_ARGS__);                                        \
    } while (0)

// int Init(int32_t deviceId, aclrtStream* stream) {
//   // 固定写法，AscendCL初始化
//   // auto ret = aclInit(nullptr);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n",
//   ret)); ret = aclrtSetDevice(deviceId); CHECK_RET(ret == ACL_SUCCESS,
//   LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret)); ret =
//   aclrtCreateStream(stream); CHECK_RET(ret == ACL_SUCCESS,
//   LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret)); return 0;
// }

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
    int64_t stride[t.dim()], sizes[t.dim()];
    get_stride(t, stride);
    get_size(t, sizes);
    aclDataType acl_data_type =
        kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(
            t.scalar_type())];
    if (dtype == 0)
        return aclCreateTensor(sizes, t.dim(), acl_data_type, stride, 0,
                               ACL_FORMAT_ND, sizes, t.dim(), t.data_ptr());
    if (dtype == 1) {
        sizes[2] *= 2;
        for (int i = 0; i < t.dim() - 1; i++) {
            stride[i] *= 2;
        }
        return aclCreateTensor(sizes, t.dim(), ACL_INT4, stride, 0,
                               ACL_FORMAT_ND, sizes, t.dim(), t.data_ptr());
    }
}

std::basic_string<char> get_id(at::Tensor input_1, at::Tensor input_2) {
    std::basic_string<char> id;
    for (int i = 0; i < input_1.dim(); i++) {
        id += std::to_string(input_1.size(i));
    }
    for (int i = 0; i < input_2.dim(); i++) {
        id += std::to_string(input_2.size(i));
    }
    return id;
}

// struct GEMMOpWorkspace{
//     aclOpExecutor* executor;
//     void *workspaceAddr;
//     uint64_t workspaceSize;
//     OpWorkspace(aclOpExecutor* opexc, void *spaddr, uint64_t spsize){
//     }
//     OpWorkspace(){
//       this->executor = nullptr;
//       this->workspaceAddr = nullptr;
//       this->workspaceSize = -1;
//     }
// };
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

void GroupedMatmul(at::Tensor x, at::Tensor weight,
                   at::Tensor antiquantScaleOptional,
                   at::Tensor antiquantOffsetOptional,
                   at::Tensor groupListOptional, GroupedGemmType type,
                   at::Tensor output
                   // int64_t splitItem,
                   // int64_t groupType,
                   // int64_t groupListType
) {
    // auto ret = Init(deviceId, &stream);
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    // check根据自己的需要处理
    // aclrtStream acl_stream;
    // aclrtCreateStream(&acl_stream);
    aclTensor *x_ = createTensor(x, 0);
    aclTensor *weight_ = type == GroupedGemmType::FP4 ? createTensor(weight, 1)
                                                      : createTensor(weight, 0);
    aclTensor *antiquantScaleOptional_ =
        createTensor(antiquantScaleOptional, 0);
    aclTensor *antiquantOffsetOptional_ =
        createTensor(antiquantOffsetOptional, 0);
    aclTensor *groupListOptional_ = createTensor(groupListOptional, 0);
    aclTensor *output_ = createTensor(output, 0);
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor;
    // std::string input_id;
    // input_id = get_id(x, weight);
    auto ret = aclnnGroupedMatmulAntiquantGetWorkspaceSize(
        x_, weight_, antiquantScaleOptional_, antiquantOffsetOptional_,
        groupListOptional_, output_, &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT(
            "aclnnGroupedMatmulAntiquantGetWorkspaceSize failed. ERROR: %d\n",
            ret));
    // 根据第一段接口计算出的workspaceSize申请device内存
    workspaceAddr = WorkspaceManager().getFromMap(workspaceSize);
    // std::chrono::time_point<std::chrono::system_clock> start, end;
    // start = std::chrono::system_clock::now();
    ret = aclnnGroupedMatmulAntiquant(workspaceAddr, workspaceSize, executor,
                                      acl_stream);
    // ret = aclrtSynchronizeStream(acl_stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed.
    //   ERROR: %d\n", ret));
    // ret = aclrtSynchronizeStream(acl_stream);
    // end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    // std::cout << "finished computation at " << std::ctime(&end_time)
    // << "elapsed time: " << elapsed_seconds.count() << "s\n";
    // // 4. （固定写法）同步等待任务执行结束
    // if (workspaceSize > 0) {
    //   aclrtFree(workspaceAddr);
    // }
    return;
}
} // namespace grouped_gemm