#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os

from chitu.ops import anti_quant_fp4, reverse_preprocess

sys.path.append(os.getcwd())
import ascend_adapter

torch.npu.config.allow_internal_format = False


class TestCustomAdd(TestCase):

    # def test_fp8_to_bf16_custom_ops_2d(self):
    #     BLOCK_SIZE = 128
    #     shape_pairs = [
    #         ([2112, 7168], [17, 56]),
    #         ([3072, 1536], [24, 12]),
    #         ([4096, 512],  [32, 4]),
    #         ([7168, 2048], [56, 16]),
    #         ([4608, 7168], [36, 56]),
    #         ([7168, 2304], [56, 18]),
    #         ([512, 7168],  [4, 56]),
    #         ([7168, 256],  [56, 2])
    #     ]

    #     for (M, N), (scale_M, scale_N) in shape_pairs:
    #         # 生成三维输入数据
    #         input_fp8_x = torch.full([M, N], 0b11100100, device='npu', dtype=torch.uint8)
    #         input_float_scale = torch.full([scale_M, scale_N], 1.0, device='npu', dtype=torch.float32)
    #         output = torch.zeros([M, N], device='npu', dtype=torch.bfloat16)
    #         torch.npu.synchronize()

    #         # 调用 run_fp8_to_float_ascend
    #         ascend_adapter.run_fp8_to_bf16_ascend(
    #                 input_fp8_x,
    #                 input_float_scale,
    #                 output,
    #                 128
    #             )
    #         print(f"output: {output}")

    #         # 调用torch等价计算
    #         input_np_x = np.full([M, N], 0b11100100, dtype=np.uint8)
    #         x_uint_32 = input_np_x.astype(np.uint32)
    #         y_uint_32 = ((x_uint_32 & 0x80) << 24) | ((x_uint_32 & 0x7F) << 20)
    #         y_float_32 = y_uint_32.view(dtype=np.float32)
    #         result_x = (y_float_32 * (2 ** 120))
    #         golden = torch.from_numpy(result_x.astype(np.float32) * 1.0).to(torch.bfloat16)
    #         print(f"golden: {golden}")

    #         # 验证结果
    #         try:
    #             self.assertRtolEqual(output, golden)
    #             print(f"测试通过: M={M}, N={N}")
    #         except AssertionError as e:
    #             print(f"测试失败: M={M}, N={N}")
    #             print(f"错误信息: {str(e)}")
    #             raise e

    # def test_fp8_to_bf16_custom_ops_3d(self):
    #     shape_pairs = [
    #         (4, [2112, 7168], [17, 56]),
    #         # (8, [3072, 1536], [24, 12]),
    #         # (16, [4096, 512], [32, 4]),
    #         # (32, [7168, 2048], [56, 16]),
    #         # (64, [4608, 7168], [36, 56]),
    #         # (128, [7168, 2304], [56, 18]),
    #         # (256, [512, 7168], [4, 56]),
    #         # (512, [7168, 256], [56, 2])
    #     ]

    #     for G, (M, N), (scale_M, scale_N) in shape_pairs:
    #         print(f"\n测试 shape: G={G}, M={M}, N={N}, scale_M={scale_M}, scale_N={scale_N}")

    #         # 生成三维输入数据
    #         input_fp8_x = torch.full([G, M, N], 0b11100100, device='npu', dtype=torch.uint8)
    #         input_float_scale = torch.full([G, scale_M, scale_N], 1.0, device='npu', dtype=torch.float32)
    #         output = torch.zeros([G, M, N], device='npu', dtype=torch.bfloat16)
    #         torch.npu.synchronize()

    #         # 遍历 G 维度，对每个二维切片调用算子
    #         ascend_adapter.run_fp8_to_bf16_ascend(
    #                 input_fp8_x,
    #                 input_float_scale,
    #                 output,
    #                 128
    #             )
    #         print(f"output: {output}")

    #         # 调用torch等价计算
    #         input_np_x = np.full([G, M, N], 0b11100100, dtype=np.uint8)
    #         x_uint_32 = input_np_x.astype(np.uint32)
    #         y_uint_32 = ((x_uint_32 & 0x80) << 24) | ((x_uint_32 & 0x7F) << 20)
    #         y_float_32 = y_uint_32.view(dtype=np.float32)
    #         result_x = (y_float_32 * (2 ** 120))

    #         golden = torch.from_numpy(result_x.astype(np.float32) * 1.0).to(torch.bfloat16)

    #         # 验证结果
    #         try:
    #             self.assertRtolEqual(output, golden)
    #             print(f"测试通过: G={G}, M={M}, N={N}")
    #         except AssertionError as e:
    #             print(f"测试失败: G={G}, M={M}, N={N}")
    #             print(f"错误信息: {str(e)}")
    #             raise e

    # # def test_fp4_to_bf16_custom_ops(self):
    # #     M = 128
    # #     N = 128
    # #     BLOCK_SIZE = 16

    # #     # 【生成输入数据】
    # #     input_fp8_x = torch.full([M, N//2], 0b00010001, device='npu', dtype=torch.uint8)
    # #     input_fp8_x[0][0] = 0b00100010
    # #     input_fp8_x[16] = 0b00100010
    # #     input_fp8_scale = torch.full([M, N//BLOCK_SIZE], 0x30, device='npu', dtype=torch.uint8)
    # #     input_fp32_scale = torch.full([1], 2.0, device='npu', dtype=torch.float32)
    # #     output_bf16 = torch.zeros([M, N], device='npu', dtype=torch.bfloat16)
    # #     torch.npu.synchronize()

    # #     # 【调用自定义算子】
    # #     ascend_adapter.run_fp4_to_bf16_ascend(input_fp8_x, output_bf16, input_fp8_scale, input_fp32_scale[0], BLOCK_SIZE)
    # #     #########################################################################################

    # #     # 【调用torch等价计算】
    # #     input_np_x = np.full([M, N//2], 0b00010001, dtype=np.uint8)
    # #     input_np_x[0][0] = 0b00100010
    # #     input_np_x[16] = 0b00100010
    # #     input_np_scale1 = np.full([M, N//BLOCK_SIZE], 0x30, dtype=np.uint8)
    # #     input_np_scale2 = np.full([1], 2.0, dtype=np.float32)
    # #     scale1_uint_32 = input_np_scale1.astype(np.uint32)
    # #     scale1_uint_32 = ((scale1_uint_32 & 0x80) << 24) | ((scale1_uint_32 & 0x7F) << 20)
    # #     scale1_float_32 = scale1_uint_32.view(dtype=np.float32)
    # #     scale1_float_32 = (scale1_float_32 * (2 ** 120))
    # #     tmp = np.ones([M,N//BLOCK_SIZE,BLOCK_SIZE],dtype=np.float32)
    # #     scale1_float_32 = (scale1_float_32.reshape(M, N//BLOCK_SIZE, 1)*tmp).reshape(M, N)

    # #     x_uint_32_left = ((input_np_x & 0xF0) >> 4).reshape(M*N//2).astype(np.uint32)
    # #     x_uint_32_right = (input_np_x & 0x0F).reshape(M*N//2).astype(np.uint32)

    # #     result_x_32 = np.zeros((M*N), dtype=np.uint32)
    # #     for i in range(M * N // 2 // BLOCK_SIZE):
    # #         for j in range(BLOCK_SIZE):
    # #             result_x_32[2*BLOCK_SIZE*i+j] = x_uint_32_left[BLOCK_SIZE*i+j]
    # #             result_x_32[2*BLOCK_SIZE*i+BLOCK_SIZE+j] = x_uint_32_right[BLOCK_SIZE*i+j]
    # #     result_x_32 = result_x_32.reshape(M,N)
    # #     # print("result_x_32", result_x_32)

    # #     y_uint_32 = ((result_x_32 & 0x08) << 28) | ((result_x_32 & 0x07) << 22)
    # #     y_float_32 = y_uint_32.view(dtype=np.float32)
    # #     result_x = (y_float_32 * (2 ** 126))
    # #     # 分步进行转换，确保数据类型正确
    # #     result_x = result_x.astype(np.float32)
    # #     scale1_float_32 = scale1_float_32.astype(np.float32)
    # #     final_result = result_x * scale1_float_32 * input_np_scale2[0]
    # #     golden = torch.from_numpy(final_result.astype(np.float32)).to(torch.bfloat16)

    # #     # torch.set_printoptions(profile='full')
    # #     print("output_bf16 = ", output_bf16)
    # #     print("golden = ", golden)

    # #     self.assertRtolEqual(output_bf16, golden)

    # def test_fp4_to_bf16_multi_tensor(self):
    #     tensorNum = 514
    #     M = 32 * tensorNum
    #     N = 128
    #     BLOCK_SIZE = 16

    #     # 【生成输入数据】
    #     input_fp8_x = torch.full([M, N//2], 0b00010001, device='npu', dtype=torch.uint8)
    #     input_fp8_x[0][0] = 0b00100010
    #     input_fp8_x[16] = 0b00100010
    #     input_fp8_scale = torch.full([M, N//BLOCK_SIZE], 0x30, device='npu', dtype=torch.uint8)
    #     input_fp32_scale = torch.full([tensorNum], 2.0, device='npu', dtype=torch.float32)
    #     input_fp32_scale[1] = 3.0
    #     output_bf16 = torch.zeros([M, N], device='npu', dtype=torch.bfloat16)
    #     torch.npu.synchronize()

    #     # 【调用自定义算子】
    #     ascend_adapter.run_fp4_to_bf16_multi_tensor_ascend(input_fp8_x, output_bf16, input_fp8_scale, input_fp32_scale, tensorNum, BLOCK_SIZE)
    #     #########################################################################################

    #     # 【调用torch等价计算】
    #     input_np_x = np.full([M, N//2], 0b00010001, dtype=np.uint8)
    #     input_np_x[0][0] = 0b00100010
    #     input_np_x[16] = 0b00100010
    #     input_np_scale1 = np.full([M, N//BLOCK_SIZE], 0x30, dtype=np.uint8)
    #     input_np_scale2 = np.full([tensorNum], 2.0, dtype=np.float32)
    #     input_np_scale2[1] = 3.0
    #     scale1_uint_32 = input_np_scale1.astype(np.uint32)
    #     scale1_uint_32 = ((scale1_uint_32 & 0x80) << 24) | ((scale1_uint_32 & 0x7F) << 20)
    #     scale1_float_32 = scale1_uint_32.view(dtype=np.float32)
    #     scale1_float_32 = (scale1_float_32 * (2 ** 120))
    #     tmp = np.ones([M,N//BLOCK_SIZE,BLOCK_SIZE],dtype=np.float32)
    #     scale1_float_32 = (scale1_float_32.reshape(M, N//BLOCK_SIZE, 1)*tmp).reshape(M, N)

    #     x_uint_32_left = ((input_np_x & 0xF0) >> 4).reshape(M*N//2).astype(np.uint32)
    #     x_uint_32_right = (input_np_x & 0x0F).reshape(M*N//2).astype(np.uint32)

    #     result_x_32 = np.zeros((M*N), dtype=np.uint32)
    #     for i in range(M * N // 2 // BLOCK_SIZE):
    #         for j in range(BLOCK_SIZE):
    #             result_x_32[2*BLOCK_SIZE*i+j] = x_uint_32_left[BLOCK_SIZE*i+j]
    #             result_x_32[2*BLOCK_SIZE*i+BLOCK_SIZE+j] = x_uint_32_right[BLOCK_SIZE*i+j]
    #     result_x_32 = result_x_32.reshape(M,N)
    #     # print("result_x_32", result_x_32)

    #     y_uint_32 = ((result_x_32 & 0x08) << 28) | ((result_x_32 & 0x07) << 22)
    #     y_float_32 = y_uint_32.view(dtype=np.float32)
    #     result_x = (y_float_32 * (2 ** 126))
    #     # 分步进行转换，确保数据类型正确
    #     result_x = result_x.astype(np.float32)
    #     scale1_float_32 = scale1_float_32.astype(np.float32)
    #     final_result = ((result_x * scale1_float_32).reshape(tensorNum, M//tensorNum*N) * input_np_scale2.reshape(tensorNum,1)).reshape(M,N)
    #     golden = torch.from_numpy(final_result.astype(np.float32)).to(torch.bfloat16)

    #     # torch.set_printoptions(profile='full')
    #     print("output_bf16 = ", output_bf16)
    #     print("golden = ", golden)
    #     tmp=output_bf16.cpu().reshape(tensorNum, M//tensorNum, N)
    #     print(tmp[1])
    #     tmp=golden.reshape(tensorNum, M//tensorNum, N)
    #     print(tmp[1])
    #     self.assertRtolEqual(output_bf16, golden)

    def reshape_3d_to_2d(self, weight_3d, scale_3d, scale_2_3d):
        """将三维权重和 scale 转换为二维"""
        # 确保内存连续性
        weight_3d = weight_3d.contiguous()
        if scale_3d is not None:
            scale_3d = scale_3d.contiguous()

        # 转换权重
        weight_2d = weight_3d.view(-1, weight_3d.shape[-1])

        # 转换 scale
        if scale_3d is not None:
            if len(scale_3d.shape) > 1:
                scale_2d = scale_3d.view(-1, scale_3d.shape[-1])
            else:
                scale_2d = scale_3d.view(-1)
        else:
            scale_2d = None
        if scale_2_3d is not None:
            scale_2_2d = scale_2_3d.view(-1, scale_2_3d.shape[-1])
        else:
            scale_2_2d = None

        return weight_2d, scale_2d, scale_2_2d

    def reshape_2d_to_3d(self, weight_2d, num_experts):
        """将二维权重转换回三维"""
        # 确保内存连续性
        weight_2d = weight_2d.contiguous()

        # 计算中间维度
        # weight_2d.shape[0] = num_experts * middle_dim
        middle_dim = weight_2d.shape[0] // num_experts

        # 转换回三维
        weight_3d = weight_2d.view(num_experts, middle_dim, weight_2d.shape[-1])

        return weight_3d

    def test_fp4_to_bf16_torch_2d(self):
        BLOCK_SIZE = 16
        print(f"so:{ascend_adapter.__file__}")

        w1w3_s = torch.randint(0, 255, (7168, 256), dtype=torch.uint8, device="npu")
        w1w3_s_s = torch.randint(
            0, 255, (7168, 256 * 2 // 16), dtype=torch.uint8, device="npu"
        )
        b_s_2_2 = torch.randn((1, 1), dtype=torch.float32, device="npu")
        b_s_2_2[0][0] = 0.0002
        # b_s_2 = torch.tensor([0.0002, 0.0002], dtype=torch.float32, device="npu")
        w1w3_ = anti_quant_fp4(w1w3_s, w1w3_s_s, b_s_2_2)
        print(f"torch_dequant_weight: {w1w3_} shape: {w1w3_.shape}")

        # 【调用自定义算子】
        # cal elapsed time of ascend_adapter.run_fp4_to_bf16_multi_tensor_ascend
        import time

        start_time = time.time()

        shape_w = w1w3_s.shape
        shape_nw = list(shape_w)
        shape_nw[-1] *= 2
        output_bf16 = torch.zeros(shape_nw, device="npu", dtype=torch.bfloat16)
        ascend_adapter.run_fp4_to_bf16_ascend(
            reverse_preprocess(w1w3_s), output_bf16, w1w3_s_s, float(b_s_2_2), 16
        )
        torch.npu.synchronize()
        print(f"output_bf16 shape: {output_bf16.shape}, output_bf16: {output_bf16}")

        end_time = time.time()
        print(
            f"Elapsed time of ascend_adapter.run_fp4_to_bf16_multi_tensor_ascend: {end_time - start_time} seconds"
        )

        print(f"close:{torch.sum(torch.isclose(output_bf16, w1w3_)==True)}")

    # def test_fp4_to_bf16_torch_3d(self):
    #     BLOCK_SIZE = 16
    #     print(f"so:{ascend_adapter.__file__}")
    #     w1w3_s = torch.randint(0, 255, (256, 7168, 256), dtype=torch.uint8, device='npu')
    #     w1w3_s_s = torch.randint(0, 255, (256, 7168, 256 * 2 // 16), dtype=torch.uint8, device='npu')
    #     # create b_s_2 as 256, 2, 1 shape tensor
    #     b_s_2_2 = torch.randn((256, 2, 1), dtype=torch.float32, device='npu')
    #     # b_s_2 = torch.tensor([0.0002, 0.0002], dtype=torch.float32, device="npu")
    #     w1w3_ = anti_quant_fp4(w1w3_s, w1w3_s_s, b_s_2_2)
    #     print(f"torch_dequant_weight: {w1w3_}")

    #     # 【调用自定义算子】
    #     # cal elapsed time of ascend_adapter.run_fp4_to_bf16_multi_tensor_ascend
    #     import time
    #     start_time = time.time()

    #     w1w3_2d, w1w3_s_2d, b_s_2_2_2d = self.reshape_3d_to_2d(w1w3_s, w1w3_s_s, b_s_2_2)
    #     shape_w = w1w3_2d.shape
    #     shape_nw = list(shape_w)
    #     shape_nw[-1] *= 2
    #     output_bf16_2 = torch.zeros(shape_nw, device='npu', dtype=torch.bfloat16)
    #     ascend_adapter.run_fp4_to_bf16_multi_tensor_ascend(w1w3_2d, output_bf16_2, w1w3_s_2d, b_s_2_2_2d, 512, BLOCK_SIZE)
    #     #reshape output_bf16_2 to 3d
    #     output_bf16_3d = self.reshape_2d_to_3d(output_bf16_2, 256)
    #     torch.npu.synchronize()
    #     print(f"output_bf16_3d shape: {output_bf16_3d.shape}, output_bf16_3d: {output_bf16_3d}")

    #     # ascend_adapter.run_fp4_to_bf16_ascend(w1w3_s, output_bf16, w1w3_s_s, float(b_s_2), 16)
    #     end_time = time.time()
    #     print(f"Elapsed time of ascend_adapter.run_fp4_to_bf16_multi_tensor_ascend: {end_time - start_time} seconds")


if __name__ == "__main__":
    run_tests()
