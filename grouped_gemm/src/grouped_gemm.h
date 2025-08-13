#pragma once

#include "aclnn_grouped_matmul_antiquant.h"
#include "aclnnop/aclnn_copy.h"
#include <ATen/Tensor.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace grouped_gemm {
void GroupedGemm(at::Tensor x, at::Tensor weight,
                 at::Tensor antiquantScaleOptional,
                 at::Tensor antiquantOffsetOptional,
                 at::Tensor groupListOptional, char *computeType,
                 at::Tensor output);
} // namespace grouped_gemm