/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclrtlaunch_fp4_to_bf16_custom.h"
#include "aclrtlaunch_fp4_to_bf16_multi_tensor.h"
#include "aclrtlaunch_fp8_to_bf16_custom.h"
#include "aclrtlaunch_fp8_to_bf16_custom2.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <assert.h>
#include <pybind11/pybind11.h>
#include <string>
#include <torch/extension.h>

namespace ascend_custom {
void run_fp8_to_bf16_ascend(const at::Tensor &x, const at::Tensor &scale,
                            const at::Tensor &output, const int BLOCK_SIZE) {
    assert(BLOCK_SIZE == 128);
    assert(x.is_contiguous() && scale.is_contiguous() &&
           output.is_contiguous());
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

    if (x.dim() == 2) {
        int M = x.sizes()[0], N = x.sizes()[1];
        assert(N % 128 == 0);
        uint32_t blockDim = (M + BLOCK_SIZE - 1) / 128;

        ACLRT_LAUNCH_KERNEL(fp8_to_bf16_custom)
        (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
         const_cast<void *>(scale.storage().data()),
         const_cast<void *>(output.storage().data()), M, N, BLOCK_SIZE);
    } else if (x.dim() == 3) {
        int G = x.sizes()[0], M = x.sizes()[1], N = x.sizes()[2];
        assert(N % 128 == 0);
        uint32_t blockDim = (M + BLOCK_SIZE - 1) / 128;
        blockDim = G * blockDim;

        ACLRT_LAUNCH_KERNEL(fp8_to_bf16_custom2)
        (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
         const_cast<void *>(scale.storage().data()),
         const_cast<void *>(output.storage().data()), M, N, BLOCK_SIZE);
    }
}

void run_fp4_to_bf16_ascend(const at::Tensor &x, const at::Tensor &z,
                            const at::Tensor &scale1, const float scale2,
                            const int BLOCK_SIZE) {
    int M = z.sizes()[0], N = z.sizes()[1];
    assert(x.sizes()[0] == M && x.sizes()[1] == N / 2 &&
           x.sizes().size() == z.sizes().size());
    assert(x.is_contiguous() && z.is_contiguous());
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    uint32_t size = M * N;
    uint32_t blockDim = 24;
    // 256是单个核一次处理的x的元素个数，相当于512个实际元素
    while (blockDim > 0 && size % (2 * 256 * blockDim) != 0) {
        blockDim--;
    }
    if (blockDim == 0) {
        printf("Invalid Tensor size: %d.\n", size);
        assert(blockDim > 0);
    }

    ACLRT_LAUNCH_KERNEL(fp4_to_bf16_custom)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(scale1.storage().data()), scale2, M, N, BLOCK_SIZE);
}

void run_fp4_to_bf16_multi_tensor_ascend(
    const at::Tensor &x, const at::Tensor &z, const at::Tensor &scale1,
    const at::Tensor &scale2, const int tensorNum, const int BLOCK_SIZE) {
    // assert(BLOCK_SIZE == 16);
    assert(z.sizes().size() == 2);
    int M = z.sizes()[0], N = z.sizes()[1];
    assert(x.sizes()[0] == M && x.sizes()[1] == N / 2 && x.sizes().size() == 2);
    assert(scale1.sizes().size() == 2 && scale1.sizes()[0] == M &&
           scale1.sizes()[1] == N / BLOCK_SIZE);
    assert(x.is_contiguous() && z.is_contiguous());
    assert(scale2.sizes()[0] == tensorNum);
    assert(M % tensorNum == 0);
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    int singleTensorM = M / tensorNum;
    uint32_t singleTensorSize = singleTensorM * N;
    uint32_t blockDim = 24;
    if (tensorNum < blockDim) {
        blockDim = tensorNum;
    }
    uint32_t tileLength = 4096;
    // while (tileLength >= 512 && singleTensorSize % tileLength != 0) {
    //     tileLength /= 2;
    // }
    // 一个scale1为uint8_t，1B，单次最少读取32B，对应到输出矩阵就是最小为32*BLOCK_SIZE
    if (singleTensorSize % (32 * BLOCK_SIZE) != 0) {
        printf("Invalid Tensor size: %d.\n", singleTensorSize);
        assert(singleTensorSize % 512 == 0);
    }

    ACLRT_LAUNCH_KERNEL(fp4_to_bf16_multi_tensor)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(scale1.storage().data()),
     const_cast<void *>(scale2.storage().data()), M, N, tensorNum, tileLength,
     BLOCK_SIZE);
}

} // namespace ascend_custom

PYBIND11_MODULE(ascend_adapter, m) {
    m.doc() =
        "fp8_to_fp16_custom pybind11 interfaces"; // optional module docstring
    m.def("run_fp8_to_bf16_ascend", &ascend_custom::run_fp8_to_bf16_ascend, "");
    m.def("run_fp4_to_bf16_ascend", &ascend_custom::run_fp4_to_bf16_ascend, "");
    m.def("run_fp4_to_bf16_multi_tensor_ascend",
          &ascend_custom::run_fp4_to_bf16_multi_tensor_ascend, "");
}
