import torch
import grouped_gemm
from grouped_gemm import GroupedGemmType
import torch_npu
import sys


def get_ascend_custom_opp_path():
    import os
    import site

    site_packages_path = os.path.join(site.getsitepackages()[0], "vendors", "customize")
    return site_packages_path


import os

site_packages_path = get_ascend_custom_opp_path()
os.environ["ASCEND_CUSTOM_OPP_PATH"] = site_packages_path
os.environ["LD_LIBRARY_PATH"] += os.path.join(site_packages_path, "op_api", "lib")

FP4_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def to_fp4_e2m1_in_uint8(x: torch.Tensor) -> torch.Tensor:
    abs_x = x.abs()
    levels = FP4_E2M1_LEVELS.to(abs_x.device).view(*([1] * abs_x.dim()), -1)
    idx = (abs_x.unsqueeze(-1) == levels).to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    nibble = sign | idx
    return nibble


def from_fp4_e2m1_in_uint8(nibbles: torch.Tensor) -> torch.Tensor:
    n = nibbles.to(torch.uint8)
    sign = torch.where((n >> 3).bool(), -1.0, 1.0)
    idx = (n & 0x7).to(torch.long)
    levels = FP4_E2M1_LEVELS.to(n.device)
    val = sign * levels[idx]
    return val


def pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(w_nib: torch.Tensor) -> torch.Tensor:
    out, inp = w_nib.shape
    assert inp % 2 == 0
    high = w_nib[:, 0::2]
    low = w_nib[:, 1::2]
    packed = (low << 4) | high
    return packed


def unpack_every_uint8_to_two_fp4_e2m1_in_uint8(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    out, half_in = packed.shape
    high_nibble = packed & 0x0F
    low_nibble = packed >> 4
    return torch.stack([high_nibble, low_nibble], dim=2).view(out, half_in * 2)


def init_weight_and_scales(dim, block_size):
    assert dim % block_size == 0
    b = torch.randn(
        dim,
        dim // block_size,
        block_size,
        dtype=torch.float32,
        device="npu",
    )
    # b = torch.full([dim, dim // block_size,
    #      block_size,], 1, device='npu', dtype=torch.float32)
    b_s_2 = b.abs().max() / (448 * 6)
    b_s = torch.clamp((b.amax(dim=2, keepdim=True) / (6 * b_s_2)), -448, 448)
    b = b / (b_s * b_s_2)
    b = pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(
        to_fp4_e2m1_in_uint8(b.view(dim, dim))
    )
    b_s = b_s.view(dim, dim // block_size).to(torch.bfloat16)
    b_s_2 = b_s_2.view(1, 1).to(torch.float32)
    return b, b_s, b_s_2


def do_dequant_b(b, b_s, b_s_2, dim, block_size):
    return (
        from_fp4_e2m1_in_uint8(unpack_every_uint8_to_two_fp4_e2m1_in_uint8(b))
        .view(dim, dim // block_size, block_size)
        .to(torch.float32)
        * b_s.view(dim, dim // block_size, 1).to(torch.float32)
        * b_s_2.view(1, 1, 1)
    ).view(dim, dim)


def unpack_weight(weight):
    tmp_weight = weight.to(torch.int16)
    tmp_weight = ((tmp_weight & 0x00F0) << 4) | ((tmp_weight & 0x000F))
    new_weight = tmp_weight.view(torch.uint8)
    new_weight = new_weight.to(torch.bfloat16)
    return new_weight


def anti_quant_fp8_scale(scale, scale_2):
    shape_w = scale.shape
    shape_nw = list(shape_w)
    if scale_2.shape[-2] == 1:
        scale *= scale_2
    else:
        scale[..., : shape_nw[-2] // 2, :] *= scale_2[..., 0, :].unsqueeze(-1)
        scale[..., shape_nw[-2] // 2 :, :] *= scale_2[..., 1, :].unsqueeze(-1)
    return scale.to(torch.bfloat16)


def fp4_dequant(weight):
    weight = weight.to(torch.int16)
    weight = ((weight & 0x0008) << 12) | ((weight & 0x0007) << 6)
    scale_fp4_to_32 = torch.tensor(0x7E80, dtype=torch.uint16)
    weight = weight.view(torch.bfloat16) * scale_fp4_to_32.view(torch.bfloat16)
    return weight


def repack_weight(weight):
    tmp_weight = weight.to(torch.int16)
    tmp_weight = ((tmp_weight & 0x00F0) << 4) | (tmp_weight & 0x000F)
    shape = list(tmp_weight.shape)
    shape[-2] = shape[-2]
    shape[-1] = shape[-1] * 2
    new_weight = torch.empty(shape, dtype=torch.uint8, device="npu")
    new_weight = tmp_weight.view(torch.uint8)
    new_weight = new_weight.reshape(shape)
    new_weight = new_weight.transpose(-2, -1).contiguous()
    new_weight = new_weight.view(torch.int16)
    new_weight = ((new_weight & 0x0F00) >> 4) | (new_weight & 0x000F)
    return new_weight.to(torch.int8)


def repack_weight_step2_old(weight):
    weight_shape = weight.shape
    tmp_weight = weight.reshape(
        weight_shape[-3], weight_shape[-2] // 64, 4, 2, 8, weight_shape[-1] // 4, 4
    )
    new_weight = tmp_weight.permute(0, 1, 3, 2, 5, 4, 6).contiguous()
    return new_weight.reshape(weight_shape)


def repack_weight_step2(weight):
    weight_shape = weight.shape
    tmp_weight = weight.reshape(
        weight_shape[-3] * weight_shape[-2] // 64,
        4,
        2,
        8,
        weight_shape[-1] // 128,
        8,
        4,
        4,
    )
    new_weight = tmp_weight.permute(0, 2, 1, 5, 4, 6, 3, 7).contiguous()
    return new_weight.reshape(weight_shape)


def get_fusion_group_matmul_(
    x, w1w3_, w1w3_scale, w1w3_scale_2, expert_tokens, dim, block_size
):
    dequant_b = do_dequant_b(w1w3_, w1w3_scale, w1w3_scale_2, dim, block_size).to(
        torch.bfloat16
    )
    dequant_b = dequant_b.transpose(-2, -1).contiguous()
    output = torch_npu.npu_grouped_matmul(
        [x],
        [dequant_b],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )
    torch.npu.synchronize()
    return output[0]


def test_fp4_quantization():
    """测试 FP4 量化与反量化"""
    print("===== 运行 FP4 量化测试 =====")
    torch.set_default_dtype(torch.bfloat16)
    dim = 256
    block_size = 16

    G = 4
    expert_tokens = torch.tensor([2, 3, 4, 256], device="npu", dtype=torch.int64)
    x = torch.randn(dim, dim, dtype=torch.bfloat16, device="npu")
    # x = torch.full([dim, dim], 1, device='npu', dtype=torch.bfloat16)
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)
    b = b.unsqueeze(0).repeat(G, 1, 1)
    b_s = b_s.unsqueeze(0).repeat(G, 1, 1)
    b_s_2 = b_s_2.unsqueeze(0).repeat(G, 1, 1)

    w1w3 = repack_weight(b)
    w1w3 = repack_weight_step2(w1w3)
    output = torch.zeros(
        [x.shape[0], w1w3.shape[-1] * 2], dtype=x.dtype, device=x.device
    )
    scale = anti_quant_fp8_scale(b_s, b_s_2)
    new_scale = scale.transpose(-2, -1).contiguous()
    scale_off = torch.zeros_like(new_scale, dtype=scale.dtype, device=scale.device)
    grouped_gemm.grouped_gemm(
        x,
        w1w3,
        output=output,
        antiquantOffsetOptional=scale_off,
        antiquantScaleOptional=new_scale,
        groupListOptional=expert_tokens,
        type=GroupedGemmType.FP4,
        # splitItem=3,
        # groupType=0,
        # groupListType=0
    )
    torch.npu.synchronize()
    # exp out
    w1w3_2 = unpack_weight(b)

    exp_output = torch.zeros(
        [x.shape[0], w1w3_2.shape[-1] * 2], dtype=x.dtype, device=x.device
    )  # [256, 512]

    w1w3_2 = fp4_dequant(w1w3_2)
    for i in range(scale.shape[-1]):
        w1w3_2[:, :, i * 16 : (i + 1) * 16] *= scale[:, :, i].unsqueeze(-1)
    w1w3_2 = w1w3_2.transpose(-2, -1).contiguous()

    exp_output = torch_npu.npu_grouped_matmul(
        [x],
        [w1w3_2],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )
    torch.npu.synchronize()

    # 数值验证
    diff = (exp_output[0] - output).abs().max()
    print(f"最大数值差异: {diff.item():.6f}")

    # 简单比较
    if torch.allclose(exp_output[0], output, atol=0.15):
        print("✅ FP4 GEMM 测试通过")
        sys.exit(0)
    else:
        print("❌ FP4 GEMM 测试失败")
        sys.exit(1)


if __name__ == "__main__":
    test_fp4_quantization()
