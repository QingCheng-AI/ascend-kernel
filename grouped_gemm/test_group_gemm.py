import torch
import grouped_gemm
import torch_npu


def repack_weight(weight):
    tmp_weight = weight.to(torch.int16)
    tmp_weight = ((tmp_weight & 0x00F0) << 4) | (tmp_weight & 0x000F)
    shape = list(tmp_weight.shape)
    shape[-2] = shape[-2]
    shape[-1] = shape[-1] * 2
    new_weight = torch.empty(shape, dtype=torch.uint8, device="npu")
    new_weight = tmp_weight.view(torch.uint8)
    new_weight = new_weight.transpose(-2, -1).contiguous()
    new_weight = new_weight.view(torch.int16)
    new_weight = ((new_weight & 0x0F00) >> 4) | (new_weight & 0x000F)
    return new_weight.to(torch.uint8)


def anti_quant_fp8(weight, scale):
    shape_w = weight.shape
    shape_nw = list(shape_w)
    new_weight = torch.empty(shape_nw, dtype=torch.bfloat16, device="npu")
    scale_fp8_to_32 = torch.tensor(0x7B80, dtype=torch.uint16)
    weight = weight.to(torch.int16)
    new_weight = ((weight & 0x0080) << 8) | ((weight & 0x007F) << 4)
    new_weight = new_weight.view(torch.bfloat16) * scale_fp8_to_32.view(torch.bfloat16)
    new_weight = new_weight.to(torch.float32)
    if scale.shape[-2] == 1:
        new_weight *= scale
    else:
        new_weight[..., : shape_nw[-2] // 2, :] *= scale[..., 0, :].unsqueeze(-1)
        new_weight[..., shape_nw[-2] // 2 :, :] *= scale[..., 1, :].unsqueeze(-1)
    return new_weight.to(torch.bfloat16)


def get_fusion_group_matmul(x, w1w3_, w1w3_scale, w1w3_scale_2, expert_tokens):
    # �~\~_�~^�~S�~E��~U��~M� shape
    # x shape: [48, 7168]
    # w1w3 weight: [256, 512, 3584]
    # w1w3 scale: [256, 512, 448]
    # w1w3 scale_2: [256, 2, 1]
    # output shape: [48, 512]
    # logger.info("�~@�~K fusion group matmul 计�~W")
    # logger.info(f"x shape: {x.shape}, w1w3 shape: {w1w3.shape}, w1w3_scale shape: {w1w3_scale.shape}, w1w3_scale_2 shape: {w1w3_scale_2.shape}, expert_tokens shape: {expert_tokens.shape}")

    # reshape �~K�~I~M�~E~H对 w1w3 �~A~Z repack
    w1w3 = repack_weight(w1w3_)
    # w1w3 = w1w3.fill_(0)
    # w1w3[:, 1111, 128] = 2
    # B = torch.zeros(size=(48, 7168), dtype=torch.bfloat16, device='npu')
    # B[:, 1111] = 2
    # w1w3 = unpack_weight(w1w3)
    # import pdb
    # pdb.set_trace()

    # �~C�~T� grouped_gemm �~K�~I~M�~\~@�~A�~C�~T� reshape�~G��~U��~L�~F w1w3 �~Z~D形�~J��~N (256, 512, 3584) 转�~M�为 (256, 256, 7168) �~F~M转�~H~P (256, 7168, 256)
    # w1w3 = self.reshape_w1w3(w1w3)

    output = torch.zeros(
        [x.shape[0], w1w3.shape[-1] * 2], dtype=x.dtype, device=x.device
    )  # [256, 512]

    # �~\~@�~A�~E~H�~J~J w1w3 scale 转�~M�为 fp32�~L�~Xw1w3 scale 2�~L�~W�~H��~\~@�~H�~Z~D scale
    # s_shape = w1w3_scale.shape
    # scale_pad = torch.zeros((s_shape[0], s_shape[1], 512), dtype=torch.uint8, device='npu')
    # scale_pad[:, :, : 448] = w1w3_scale
    # scale_2_ = torch.zeros((s_shape[0], s_shape[1] // 128, 512 // 128), dtype=torch.float32, device='npu')
    # scale_2_[:, : scale_2_.shape[-2] // 2, :] = w1w3_scale_2[:, 0, :].unsqueeze(-1)
    # scale_2_[:, scale_2_.shape[-2] // 2 : , :] = w1w3_scale_2[:, 1, :].unsqueeze(-1)
    # scale = torch.empty(scale_pad.shape, dtype=torch.bfloat16, device='npu')
    # w1w3_scale = run_fp8_to_bf16_ascend(scale_pad, scale_2_, scale, 128)
    # scale = scale[:, :, 448]
    # w1w3_scale_fp32 = w1w3_scale.to(torch.float32) # [256, 512, 448]
    # w1w3_scale_fp32_reshape = w1w3_scale_fp32.view(256, 2, 256, 448)
    scale = anti_quant_fp8(w1w3_scale, w1w3_scale_2)
    scale = scale.transpose(-2, -1).contiguous()
    # scale = torch.ones(scale.shape, dtype=torch.bfloat16, device='npu')
    # new_scale = scale[:, : scale.shape[-2] // 2, :]
    # scale = torch.randn((256, 448, 512), dtype=torch.bfloat16, device='npu')
    # scale = torch.ones_like(scale_)

    # w1w3_scale_2_expend = w1w3_scale_2.expand(-1, -1, 448) # [256, 2, 448]

    # use nn.functional.linear to do the gemm
    # scale = w1w3_scale_fp32_reshape * w1w3_scale_2_expend.unsqueeze(2) # [256, 2, 256, 448]
    # scale = scale.reshape(256, 512, 448).to(torch.bfloat16)
    # scale = scale.transpose(1, 2)

    # FIXME: 补�~E� scale_off�~Z~D�~K�~@�
    scale_off = torch.zeros_like(
        scale, dtype=scale.dtype, device=scale.device
    )  # [256, 512, 448]
    # for i in range(scale.shape[-2]):
    #     w1w3[:, i * 16 : (i + 1) * 16, :] *= scale[:, i, :].unsqueeze(1)

    # import pdb
    # pdb.set_trace()
    # print all input shape
    # logger.info(f"x shape: {x.shape}, w1w3 shape: {w1w3.shape}, output shape: {output.shape}, scale shape: {scale.shape}, scale_off shape: {scale_off.shape}")

    # w1w3 = w1w3.fill_(34)

    # print grouped_gemm path
    # x.fill_(1)
    # w1w3[:, 1:, :] = 0
    for i in range(1):
        grouped_gemm.grouped_gemm(
            x,
            w1w3.view(torch.int8),
            antiquantOffsetOptional=scale_off,
            antiquantScaleOptional=scale,
            groupListOptional=expert_tokens,
            output=output,
            # splitItem=3,
            # groupType=0,
            # groupListType=0
        )
    torch.npu.synchronize()
    # import time
    # time1 = time.perf_counter()
    # for i in range(10):
    #     grouped_gemm.grouped_gemm(
    #         x,
    #         w1w3,
    #         output,
    #         antiquantOffsetOptional=scale_off,
    #         antiquantScaleOptional=scale,
    #         groupListOptional=expert_tokens,
    #         splitItem=3,
    #         groupType=0,
    #         groupListType=0
    #     )
    # torch.npu.synchronize()
    # time2 = time.perf_counter()
    # import pdb
    # pdb.set_trace()
    # output = torch_npu.npu_grouped_matmul(
    #     [x],
    #     [w1w3.view(torch_npu.int4)],
    #     antiquant_scale=[scale],
    #     antiquant_offset=[scale_off],
    #     group_list=expert_tokens,
    #     split_item=2,
    #     group_type=0,
    #     group_list_type=0,
    # )
    return output, w1w3, 0


def unpack_weight(weight):
    tmp_weight = weight.to(torch.int16)
    tmp_weight = ((tmp_weight & 0x00F0) << 4) | ((tmp_weight & 0x000F))
    new_weight = tmp_weight.view(torch.uint8)
    new_weight = new_weight.to(torch.bfloat16)
    return new_weight


def fp4_dequant(weight):
    weight = weight.to(torch.int16)
    weight = ((weight & 0x0008) << 12) | ((weight & 0x0007) << 6)
    scale_fp4_to_32 = torch.tensor(0x7E80, dtype=torch.uint16)
    weight = weight.view(torch.bfloat16) * scale_fp4_to_32.view(torch.bfloat16)
    return weight


def get_fusion_group_matmul_(x, w1w3_, w1w3_scale, w1w3_scale_2, expert_tokens):
    # �~\~_�~^�~S�~E��~U��~M� shape
    # x shape: [48, 7168]
    # w1w3 weight: [256, 512, 3584]
    # w1w3 scale: [256, 512, 448]
    # w1w3 scale_2: [256, 2, 1]
    # output shape: [48, 512]
    # logger.info("�~@�~K fusion group matmul 计�~W")
    # logger.info(f"x shape: {x.shape}, w1w3 shape: {w1w3.shape}, w1w3_scale shape: {w1w3_scale.shape}, w1w3_scale_2 shape: {w1w3_scale_2.shape}, expert_tokens shape: {expert_tokens.shape}")

    # reshape �~K�~I~M�~E~H对 w1w3 �~A~Z repack
    w1w3 = unpack_weight(w1w3_)

    # �~C�~T� grouped_gemm �~K�~I~M�~\~@�~A�~C�~T� reshape�~G��~U��~L�~F w1w3 �~Z~D形�~J��~N (256, 512, 3584) 转�~M�为 (256, 256, 7168) �~F~M转�~H~P (256, 7168, 256)
    # w1w3 = self.reshape_w1w3(w1w3)

    output = torch.zeros(
        [x.shape[0], w1w3.shape[-1] * 2], dtype=x.dtype, device=x.device
    )  # [256, 512]

    # �~\~@�~A�~E~H�~J~J w1w3 scale 转�~M�为 fp32�~L�~Xw1w3 scale 2�~L�~W�~H��~\~@�~H�~Z~D scale
    # s_shape = w1w3_scale.shape
    # scale_pad = torch.zeros((s_shape[0], s_shape[1], 512), dtype=torch.uint8, device='npu')
    # scale_pad[:, :, : 448] = w1w3_scale
    # scale_2_ = torch.zeros((s_shape[0], s_shape[1] // 128, 512 // 128), dtype=torch.float32, device='npu')
    # scale_2_[:, : scale_2_.shape[-2] // 2, :] = w1w3_scale_2[:, 0, :].unsqueeze(-1)
    # scale_2_[:, scale_2_.shape[-2] // 2 : , :] = w1w3_scale_2[:, 1, :].unsqueeze(-1)
    # scale = torch.empty(scale_pad.shape, dtype=torch.bfloat16, device='npu')
    # w1w3_scale = run_fp8_to_bf16_ascend(scale_pad, scale_2_, scale, 128)
    # scale = scale[:, :, 448]
    # w1w3_scale_fp32 = w1w3_scale.to(torch.float32) # [256, 512, 448]
    # w1w3_scale_fp32_reshape = w1w3_scale_fp32.view(256, 2, 256, 448)
    scale = anti_quant_fp8(w1w3_scale, w1w3_scale_2)
    # # scale = torch.ones_like(scale)
    w1w3 = fp4_dequant(w1w3)
    for i in range(scale.shape[-1]):
        w1w3[:, :, i * 16 : (i + 1) * 16] *= scale[:, :, i].unsqueeze(-1)
    # scale = scale.transpose(-2, -1).contiguous()

    # w1w3_scale_2_expend = w1w3_scale_2.expand(-1, -1, 448) # [256, 2, 448]

    # use nn.functional.linear to do the gemm
    # scale = w1w3_scale_fp32_reshape * w1w3_scale_2_expend.unsqueeze(2) # [256, 2, 256, 448]
    # scale = scale.reshape(256, 512, 448).to(torch.bfloat16)
    # scale = scale.transpose(1, 2)

    # FIXME: 补�~E� scale_off�~Z~D�~K�~@�
    # scale_off = torch.zeros_like(scale, dtype=scale.dtype, device=scale.device) # [256, 512, 448]

    # print all input shape
    # logger.info(f"x shape: {x.shape}, w1w3 shape: {w1w3.shape}, output shape: {output.shape}, scale shape: {scale.shape}, scale_off shape: {scale_off.shape}")

    # w1w3 = w1w3.fill_(34)

    # print grouped_gemm path

    # w1w3[:, :, 1 :] = 0
    w1w3 = w1w3.transpose(-2, -1).contiguous()
    # w1w3 = w1w3.fill_(0)
    # w1w3[:, 1111, 256] = 2
    # B = torch.zeros(size=(48, 7168), dtype=torch.bfloat16, device='npu')
    # B[:, 1111] = 2
    # import pdb
    # pdb.set_trace()
    # x.fill_(1)
    for i in range(100):
        output = torch_npu.npu_grouped_matmul(
            [x],
            [w1w3],
            group_list=expert_tokens,
            split_item=2,
            group_type=0,
            group_list_type=0,
        )
    torch.npu.synchronize()
    import time

    time1 = time.perf_counter()
    for i in range(100):
        output = torch_npu.npu_grouped_matmul(
            [x],
            [w1w3],
            group_list=expert_tokens,
            split_item=2,
            group_type=0,
            group_list_type=0,
        )
    torch.npu.synchronize()
    time2 = time.perf_counter()
    return output, w1w3, (time2 - time1)


def load_data():
    pt_file_path = "./group_matmul_step_1_deepseek-v3_fp4_m0_c0_l0_d0_1.pt"
    pt_file_path2 = "./golden/group_matmul_step_1_deepseek-v3_fp4_m0_c0_l0_d0_1.pt"
    expert_d = torch.load(pt_file_path2)
    # print(expert_d.shape)
    # print(expert_d.dtype)
    expert_data = torch.load(pt_file_path)
    # print(expert_data.shape)
    # print(expert_data.dtype)
    # 检查必要的键是否存在
    required_keys = [
        "expanded_x",
        "expanded_row_idx",
        "expanded_expert_idx",
        "w1_dequant",
        "expert_tokens",
        "w1_fp4",  # 原始权重
        "w2_fp4",  # 原始权重
        "w1_fp4_scale",
        "w1_fp4_scale_2",
        "w2_fp4_scale",
        "w2_fp4_scale_2",
        "func_return",
    ]
    w1w3 = expert_data["w1_fp4"]
    w1w3_ = expert_d["w1_fp4"]
    w1_s = expert_data["w1_fp4_scale"]
    w1_s2 = expert_data["w1_fp4_scale_2"]
    # expert_tokens = torch.tensor([24, 48], dtype=torch.int64, device='npu')
    expert_tokens = expert_d["expert_tokens"]
    expert_tokens = expert_tokens
    x = expert_d["expanded_x"]
    exp_out = expert_d["func_return"]
    print(x.shape)
    print(w1w3.shape)
    print(w1_s.shape)
    print(w1_s2.shape)
    out, w1w3_1, time1 = get_fusion_group_matmul(x, w1w3, w1_s, w1_s2, expert_tokens)
    # import pdb
    # pdb.set_trace()
    print(out)
    # import pdb
    # pdb.set_trace()
    # exp, w1w3_2, time2 = get_fusion_group_matmul_(x, w1w3, w1_s, w1_s2, expert_tokens)
    # print(out)
    # print(time1 / 100)
    # print(time2 / 100)
    # import numpy as np
    # torch.set_printoptions(threshold=np.inf)
    # print(out[0])
    # print(exp[0])
    # print(w1w3_2[0])


load_data()
