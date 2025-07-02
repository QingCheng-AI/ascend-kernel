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
 */

/*!
 * \file grouped_matmul_utils.h
 * \brief
 */

#ifndef EXAMPLE_GROUPED_MATMUL_UTILS_H
#define EXAMPLE_GROUPED_MATMUL_UTILS_H

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace grouped_matmul_example {
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

int64_t GetShapeSize(const std::vector<int64_t> &shape);

template <typename T>
void SaveOutResult(std::string &fileName, std::vector<int64_t> &shape,
                   void **deviceAddr, aclDataType dataType) {
    auto size = GetShapeSize(shape);
    auto dtypeSize = aclDataTypeSize(dataType);
    std::vector<T> resultData(size, 0);
    auto ret =
        aclrtMemcpy(resultData.data(), resultData.size() * dtypeSize,
                    *deviceAddr, size * dtypeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return);
    std::ofstream file(fileName, std::ios::binary);
    // Save data to file
    file.write(static_cast<char *>((void *)resultData.data()),
               size * dtypeSize);
    file.close();
}

int Init(int32_t deviceId, aclrtStream *stream);

int ReadBinFileNNop(const std::string &filePath, void *buffer,
                    size_t bufferSize);

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the
    // device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);

    // Compute the strides of the contiguous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostData,
                        const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType,
                        aclTensorList **tensor) {
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i,
                                     dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

int CreateAclTensor(const std::string &filePath,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor);

int CreateAclTensorList(const std::string &filePath,
                        const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType,
                        aclTensorList **tensor);

struct GroupedMatmulParams {
    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;
    aclTensor *antiquantScale = nullptr;
    aclTensor *antiquantOffset = nullptr;
    aclTensor *perTokenScale = nullptr;
    aclIntArray *groupList = nullptr;
    aclTensor *groupListTensor = nullptr;
    aclTensor *y = nullptr;

    // only support nullptr
    aclTensorList *activationInput = nullptr;
    aclTensorList *activationQuantScale = nullptr;
    aclTensorList *activationQuantOffset = nullptr;
    aclTensorList *activationFeatureOut = nullptr;
    aclTensorList *dynQuantScaleOut = nullptr;
};

// constexpr uint16_t TENSOR_SIZE = 2;
struct GroupedMatmulDevAddr {
    // void *x[TENSOR_SIZE] = {nullptr, nullptr};
    // void *weight[TENSOR_SIZE] = {nullptr, nullptr};
    // void *bias[TENSOR_SIZE] = {nullptr, nullptr};
    // void *scale[TENSOR_SIZE] = {nullptr, nullptr};
    // void *offset[TENSOR_SIZE] = {nullptr, nullptr};
    // void *antiquantScale[TENSOR_SIZE] = {nullptr, nullptr};
    // void *antiquantOffset[TENSOR_SIZE] = {nullptr, nullptr};
    // void *perTokenScale[TENSOR_SIZE] = {nullptr, nullptr};
    // void *groupList[TENSOR_SIZE] = {nullptr, nullptr};
    // void *groupListTensor[TENSOR_SIZE] = {nullptr, nullptr};
    // void *y[TENSOR_SIZE] = {nullptr, nullptr};
    // void *workspaceAddr = nullptr;
    void *x = nullptr;
    void *weight = nullptr;
    void *bias = nullptr;
    void *scale = nullptr;
    void *offset = nullptr;
    void *antiquantScale = nullptr;
    void *antiquantOffset = nullptr;
    void *perTokenScale = nullptr;
    void *groupList = nullptr;
    void *groupListTensor = nullptr;
    void *y = nullptr;
    void *workspaceAddr = nullptr;
};

void FreeParam(GroupedMatmulParams &params);

void FreeAddr(GroupedMatmulDevAddr &addrs);

void FreeResource(GroupedMatmulParams &params, GroupedMatmulDevAddr &addrs,
                  int32_t deviceId, aclrtStream *stream);
} // namespace grouped_matmul_example

#endif