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
 * \file test_grouped_matmul_v4.cpp
 * \brief
 */

#include "aclnn_grouped_matmul_antiquant.h"
#include "grouped_gemm_utils.h"

namespace grouped_matmul_example {
std::vector<int64_t> yShape;

int PrepareData(const std::string &currentPath, GroupedMatmulParams &params,
                GroupedMatmulDevAddr &addrs) {
    // The shape value must be the same with value in file
    // ./grouped_matmul_generate_data.py.
    std::vector<int64_t> xShape = {48, 7168};
    std::vector<int64_t> weightShape = {256, 7168, 512};
    std::vector<int64_t> antiquantOffsetShape = {256, 448, 512};
    std::vector<int64_t> antiquantScaleShape = {256, 448, 512};
    // std::vector<int64_t> offsetShape = {{2, 10}};
    std::vector<int64_t> groupListShape = {256};
    // std::vector<int64_t> groupShape = {2};
    // std::vector<int64_t> groupList = {16, 16};
    yShape = {48, 512};
    std::vector<int16_t> yData = std::vector<int16_t>(GetShapeSize(yShape), 0);
    // for (auto &i : yShape) {
    //     yData.push_back(std::vector<int16_t>(GetShapeSize(i), 0));
    // }

    std::string xPath = currentPath + "x.bin";
    auto ret = CreateAclTensor(xPath, xShape, &addrs.x, aclDataType::ACL_BF16,
                               &params.x);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor x failed\n");
              return ret);
    std::string weightPath = currentPath + "weight.bin";
    ret = CreateAclTensor(weightPath, weightShape, &addrs.weight,
                          aclDataType::ACL_INT4, &params.weight);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor weight failed\n");
              return ret);
    std::string antiquantOffsetPath = currentPath + "scale_offset.bin";
    ret = CreateAclTensor(antiquantOffsetPath, antiquantOffsetShape,
                          &addrs.antiquantOffset, aclDataType::ACL_BF16,
                          &params.antiquantOffset);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("Prepare Tensor scale_offset failed\n");
              return ret);
    std::string antiquantScalePath = currentPath + "scale.bin";
    ret = CreateAclTensor(antiquantScalePath, antiquantScaleShape,
                          &addrs.antiquantScale, aclDataType::ACL_BF16,
                          &params.antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor scale failed\n");
              return ret);
    std::string groupListPath = currentPath + "group_list.bin";
    ret = CreateAclTensor(groupListPath, groupListShape, &addrs.groupListTensor,
                          aclDataType::ACL_INT64, &params.groupListTensor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("Prepare Tensor group_list failed\n");
              return ret);
    // ret = CreateAclTensor(groupList, groupShape, addrs.groupListTensor,
    // aclDataType::ACL_INT64, &params.groupListTensor); CHECK_RET(ret ==
    // ACL_SUCCESS, LOG_PRINT("Prepare Tensor groupList failed\n"); return ret);
    ret = CreateAclTensor(yData, yShape, &addrs.y, aclDataType::ACL_BF16,
                          &params.y);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare Tensor y failed\n");
              return ret);

    return ACL_SUCCESS;
}
} // namespace grouped_matmul_example
using namespace grouped_matmul_example;

int main(int argc, char **argv) {
    if (argv == nullptr || argc < 2) { // 2: Input num, exeFile and testCase.
        LOG_PRINT(
            "Number of input parameter error, except 2 but got %d inputs.\n",
            argc);
        return 0;
    }
    std::string exeFile(argv[0]);
    // std::string currentPath = std::string(exeFile.substr(0,
    // exeFile.rfind('/')) + "/");
    std::string dataPath(argv[1]);
    dataPath += "/";
    // 1. (Fixed writing) Initialize the device and stream. For details, see the
    // list of external AscendCL APIs. Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Use CHECK as required.
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
              return ret);

    // 2. Construct the input and output based on the API.
    // int64_t splitItem = 3;
    // int64_t groupType = 0;
    // int64_t groupListType = 1;
    // int64_t actType = 0;
    GroupedMatmulParams params;
    GroupedMatmulDevAddr addrs;
    ret = PrepareData(dataPath, params, addrs);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Prepare failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &stream); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmulAntiquantGetWorkspaceSize.
    ret = aclnnGroupedMatmulAntiquantGetWorkspaceSize(
        params.x, params.weight, params.antiquantScale, params.antiquantOffset,
        params.groupListTensor, params.y, &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT(
            "aclnnGroupedMatmulAntiquantGetWorkspaceSize failed. ERROR: %d\n",
            ret);
        FreeResource(params, addrs, deviceId, &stream); return ret);

    // Malloc device memory for workspace based on the workspaceSize calculated
    // from the first interface
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&addrs.workspaceAddr, workspaceSize,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(params, addrs, deviceId, &stream); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmulAntiquant.
    ret = aclnnGroupedMatmulAntiquant(addrs.workspaceAddr, workspaceSize,
                                      executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGroupedMatmulAntiquant failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &stream); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &stream); return ret);

    // 5. Copy output from device to host and then save to file
    // for (int i = 0; i < yShape.size(); ++i) {
    std::string outFile = dataPath + "y.bin";
    SaveOutResult<float>(outFile, yShape, &addrs.y, aclDataType::ACL_BF16);
    // }

    // 6. Release aclTensor and device resource.
    FreeResource(params, addrs, deviceId, &stream);
    return 0;
}