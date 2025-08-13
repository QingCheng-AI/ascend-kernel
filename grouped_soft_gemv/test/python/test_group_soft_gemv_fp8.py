import torch
import cinfer_ascendc
import torch_npu
import sys


def anti_quant_fp8(weight, scale):
    shape_w = weight.shape
    shape_nw = list(shape_w)
    new_weight = torch.empty(shape_nw, dtype=torch.bfloat16, device="npu")
    scale_fp8_to_32 = torch.tensor(0x7B80, dtype=torch.uint16)
    weight = weight.to(torch.int16)
    new_weight = ((weight & 0x0080) << 8) | ((weight & 0x007F) << 4)
    new_weight = new_weight.view(torch.bfloat16) * scale_fp8_to_32.view(torch.bfloat16)
    new_weight = new_weight.to(torch.float32)
    new_weight = (
        new_weight.reshape(
            shape_w[-3], shape_w[-2] // 128, 128, shape_w[-1] // 128, 128
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    new_weight = new_weight.reshape(shape_w[-3], -1, 128 * 128) * scale.reshape(
        shape_w[-3], -1
    ).unsqueeze(-1)
    return (
        # new_weight.to(torch.bfloat16)
        new_weight.reshape(
            shape_w[-3], shape_w[-2] // 128, shape_w[-1] // 128, 128, 128
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(shape_w)
    )


def check_get_fusion_group_matmul_():
    print("===== 运行 FP8 量化测试 =====")
    G = 4
    M = 4
    K = 512
    N = 1536
    scale_K = (K + 127) // 128
    scale_N = (N + 127) // 128
    w1w3 = torch.randint(0, 255, [G, K, N], device="npu", dtype=torch.uint8)
    scale = torch.randn([G, scale_K, scale_N], device="npu", dtype=torch.float32)
    # w1w3.fill_(0)
    # w1w3[:, : 384, :] = 184
    # scale.fill_(1)
    # scale[:, 1 : 3, :] = 1
    new_w1_w3 = anti_quant_fp8(w1w3, scale)

    scale_off = torch.zeros_like(scale, dtype=torch.float32, device="npu")
    output = torch.zeros([M, N], device="npu", dtype=torch.bfloat16)
    x = torch.full([M, K], 1, device="npu", dtype=torch.bfloat16)
    expert_tokens = torch.tensor([1, 2, 3, 4], device="npu", dtype=torch.int64)

    exp_output = torch_npu.npu_grouped_matmul(
        [x.to(torch.float32)],
        [new_w1_w3],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )
    for i in range(1):
        cinfer_ascendc.grouped_soft_gemv(
            x,
            w1w3,
            output=output,
            scale=scale,
            groupList=expert_tokens,
            computeType="fp8",
        )
    torch.npu.synchronize()
    # import pdb
    # pdb.set_trace()
    # print(output)
    # print(exp_output[0])
    if torch.allclose(output, exp_output[0].to(torch.bfloat16), rtol=0.02):
        print("✅ FP8 GEMV 测试通过")
        sys.exit(0)
    else:
        print("❌ FP8 GEMV 测试失败")
        sys.exit(1)


check_get_fusion_group_matmul_()
