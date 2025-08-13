#pragma once

#include "aclnn_grouped_soft_gemv.h"
#include "aclnnop/aclnn_copy.h"
#include <ATen/Tensor.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace grouped_soft_gemv {
void GroupedSoftGemv(at::Tensor x, at::Tensor weight, at::Tensor scale,
                     at::Tensor groupList, char *computeType,
                     at::Tensor output);
} // namespace grouped_soft_gemv