#pragma once

#include <iostream>
// #include "acl/acl.h"
// #include "acl/acl_base.h"
// #include "aclnn_grouped_matmul_v4.h"
#include "aclnn_grouped_matmul_antiquant.h"
#include "aclnnop/aclnn_copy.h"
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace grouped_gemm {
void GroupedMatmul(at::Tensor x, at::Tensor weight,
                   at::Tensor antiquantScaleOptional,
                   at::Tensor antiquantOffsetOptional,
                   at::Tensor groupListOptional, at::Tensor output
                   // int64_t splitItem,
                   // int64_t groupType,
                   // int64_t groupListType
);
// void GroupedMatmul();
} // namespace grouped_gemm