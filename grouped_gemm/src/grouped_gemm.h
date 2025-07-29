#pragma once

#include "aclnn_grouped_matmul_antiquant.h"
#include "aclnn_grouped_soft_gemv.h"
#include "aclnnop/aclnn_copy.h"
#include <ATen/Tensor.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace grouped_gemm {
enum class GroupedGemmType { FP4 = 0, FP8 };
void GroupedGEMM(at::Tensor x, at::Tensor weight,
                 at::Tensor antiquantScaleOptional,
                 at::Tensor antiquantOffsetOptional,
                 at::Tensor groupListOptional, GroupedGemmType type,
                 at::Tensor output);

void GroupedGEMV(at::Tensor x, at::Tensor weight, at::Tensor scale,
                 at::Tensor groupList, GroupedGemmType type, at::Tensor output);
} // namespace grouped_gemm