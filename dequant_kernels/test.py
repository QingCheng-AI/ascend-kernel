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

sys.path.append(os.getcwd())
import ascend_adapter

torch.npu.config.allow_internal_format = False


class TestCustomAdd(TestCase):

    def test_fp8_to_bf16_custom_ops_2d(self):
        BLOCK_SIZE = 128
        shape_pairs = [
            ([2112, 7168], [17, 56]),
            ([3072, 1536], [24, 12]),
            ([4096, 512], [32, 4]),
            ([7168, 2048], [56, 16]),
            ([4608, 7168], [36, 56]),
            ([7168, 2304], [56, 18]),
            ([512, 7168], [4, 56]),
            ([7168, 256], [56, 2]),
        ]

        for (M, N), (scale_M, scale_N) in shape_pairs:
            # 生成三维输入数据
            input_fp8_x = torch.full(
                [M, N], 0b11100100, device="npu", dtype=torch.uint8
            )
            input_float_scale = torch.full(
                [scale_M, scale_N], 1.0, device="npu", dtype=torch.float32
            )
            output = torch.zeros([M, N], device="npu", dtype=torch.bfloat16)
            torch.npu.synchronize()

            # 调用 run_fp8_to_float_ascend
            ascend_adapter.run_fp8_to_bf16_ascend(
                input_fp8_x, input_float_scale, output, 128
            )
            print(f"output: {output}")

            # 调用torch等价计算
            input_np_x = np.full([M, N], 0b11100100, dtype=np.uint8)
            x_uint_32 = input_np_x.astype(np.uint32)
            y_uint_32 = ((x_uint_32 & 0x80) << 24) | ((x_uint_32 & 0x7F) << 20)
            y_float_32 = y_uint_32.view(dtype=np.float32)
            result_x = y_float_32 * (2**120)
            golden = torch.from_numpy(result_x.astype(np.float32) * 1.0).to(
                torch.bfloat16
            )
            print(f"golden: {golden}")

            # 验证结果
            try:
                self.assertRtolEqual(output, golden)
                print(f"测试通过: M={M}, N={N}")
            except AssertionError as e:
                print(f"测试失败: M={M}, N={N}")
                print(f"错误信息: {str(e)}")
                raise e

    def test_fp8_to_bf16_custom_ops_3d(self):
        shape_pairs = [
            (4, [2112, 7168], [17, 56]),
            # (8, [3072, 1536], [24, 12]),
            # (16, [4096, 512], [32, 4]),
            # (32, [7168, 2048], [56, 16]),
            # (64, [4608, 7168], [36, 56]),
            # (128, [7168, 2304], [56, 18]),
            # (256, [512, 7168], [4, 56]),
            # (512, [7168, 256], [56, 2])
        ]

        for G, (M, N), (scale_M, scale_N) in shape_pairs:
            print(
                f"\n测试 shape: G={G}, M={M}, N={N}, scale_M={scale_M}, scale_N={scale_N}"
            )

            # 生成三维输入数据
            input_fp8_x = torch.full(
                [G, M, N], 0b11100100, device="npu", dtype=torch.uint8
            )
            input_float_scale = torch.full(
                [G, scale_M, scale_N], 1.0, device="npu", dtype=torch.float32
            )
            output = torch.zeros([G, M, N], device="npu", dtype=torch.bfloat16)
            torch.npu.synchronize()

            # 遍历 G 维度，对每个二维切片调用算子
            ascend_adapter.run_fp8_to_bf16_ascend(
                input_fp8_x, input_float_scale, output, 128
            )
            print(f"output: {output}")

            # 调用torch等价计算
            input_np_x = np.full([G, M, N], 0b11100100, dtype=np.uint8)
            x_uint_32 = input_np_x.astype(np.uint32)
            y_uint_32 = ((x_uint_32 & 0x80) << 24) | ((x_uint_32 & 0x7F) << 20)
            y_float_32 = y_uint_32.view(dtype=np.float32)
            result_x = y_float_32 * (2**120)

            golden = torch.from_numpy(result_x.astype(np.float32) * 1.0).to(
                torch.bfloat16
            )

            # 验证结果
            try:
                self.assertRtolEqual(output, golden)
                print(f"测试通过: G={G}, M={M}, N={N}")
            except AssertionError as e:
                print(f"测试失败: G={G}, M={M}, N={N}")
                print(f"错误信息: {str(e)}")
                raise e

    # def test_fp4_to_bf16_custom_ops(self):
    #     M = 128
    #     N = 128
    #     BLOCK_SIZE = 16

    #     # 【生成输入数据】
    #     input_fp8_x = torch.full([M, N//2], 0b00010001, device='npu', dtype=torch.uint8)
    #     input_fp8_x[0][0] = 0b00100010
    #     input_fp8_x[16] = 0b00100010
    #     input_fp8_scale = torch.full([M, N//BLOCK_SIZE], 0x30, device='npu', dtype=torch.uint8)
    #     input_fp32_scale = torch.full([1], 2.0, device='npu', dtype=torch.float32)
    #     output_bf16 = torch.zeros([M, N], device='npu', dtype=torch.bfloat16)
    #     torch.npu.synchronize()

    #     # 【调用自定义算子】
    #     ascend_adapter.run_fp4_to_bf16_ascend(input_fp8_x, output_bf16, input_fp8_scale, input_fp32_scale[0], BLOCK_SIZE)
    #     #########################################################################################

    #     # 【调用torch等价计算】
    #     input_np_x = np.full([M, N//2], 0b00010001, dtype=np.uint8)
    #     input_np_x[0][0] = 0b00100010
    #     input_np_x[16] = 0b00100010
    #     input_np_scale1 = np.full([M, N//BLOCK_SIZE], 0x30, dtype=np.uint8)
    #     input_np_scale2 = np.full([1], 2.0, dtype=np.float32)
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
    #     final_result = result_x * scale1_float_32 * input_np_scale2[0]
    #     golden = torch.from_numpy(final_result.astype(np.float32)).to(torch.bfloat16)

    #     # torch.set_printoptions(profile='full')
    #     print("output_bf16 = ", output_bf16)
    #     print("golden = ", golden)

    #     self.assertRtolEqual(output_bf16, golden)

    def test_fp4_to_bf16_multi_tensor(self):
        tensorNum = 25
        M = 1 * tensorNum
        N = 512
        BLOCK_SIZE = 16

        # 【生成输入数据】
        # input_fp8_x = torch.full([M, N//2], 0b00010001, device='npu', dtype=torch.uint8)
        # input_fp8_x[0][0] = 0b00100110
        # input_fp8_x[16] = 0b00100010
        # input_fp8_scale = torch.full([M, N//BLOCK_SIZE], 0x30, device='npu', dtype=torch.uint8)
        # input_fp8_scale[0][0] = 0x220
        input_fp8_x = torch.randint(
            0, 255, [M, N // 2], device="npu", dtype=torch.uint8
        )
        input_fp8_scale = torch.randint(
            0, 255, [M, N // BLOCK_SIZE], device="npu", dtype=torch.uint8
        )
        input_fp32_scale = torch.full(
            [tensorNum], 2.0, device="npu", dtype=torch.float32
        )
        input_fp32_scale[0] = 0.0002
        # input_fp32_scale[1] = 0.0002
        output_bf16 = torch.zeros([M, N], device="npu", dtype=torch.bfloat16)
        torch.npu.synchronize()

        # 【调用自定义算子】
        ascend_adapter.run_fp4_to_bf16_multi_tensor_ascend(
            input_fp8_x,
            output_bf16,
            input_fp8_scale,
            input_fp32_scale,
            tensorNum,
            BLOCK_SIZE,
        )
        #########################################################################################

        # 【调用torch等价计算】
        input_np_x = input_fp8_x.cpu().numpy()
        input_np_scale1 = (
            input_fp8_scale.cpu().numpy()
        )  # np.full([M, N//BLOCK_SIZE], 0x30, dtype=np.uint8)
        input_np_scale2 = (
            input_fp32_scale.cpu().numpy()
        )  # np.full([tensorNum], 2.0, dtype=np.float32)
        # input_np_scale2[1] = 3.0
        scale1_uint_32 = input_np_scale1.astype(np.uint32)
        scale1_uint_32 = ((scale1_uint_32 & 0x80) << 24) | (
            (scale1_uint_32 & 0x7F) << 20
        )
        scale1_float_32 = scale1_uint_32.view(dtype=np.float32)
        scale1_float_32 = scale1_float_32 * (2**120)
        tmp = np.ones([M, N // BLOCK_SIZE, BLOCK_SIZE], dtype=np.float32)
        scale1_float_32 = (
            scale1_float_32.reshape(M, N // BLOCK_SIZE, 1) * tmp
        ).reshape(M, N)

        x_uint_32_left = (
            ((input_np_x & 0xF0) >> 4).reshape(M * N // 2).astype(np.uint32)
        )
        x_uint_32_right = (input_np_x & 0x0F).reshape(M * N // 2).astype(np.uint32)

        result_x_32 = np.zeros((M * N), dtype=np.uint32)
        for i in range(M * N // 2 // BLOCK_SIZE):
            for j in range(BLOCK_SIZE):
                result_x_32[2 * BLOCK_SIZE * i + j] = x_uint_32_left[BLOCK_SIZE * i + j]
                result_x_32[2 * BLOCK_SIZE * i + BLOCK_SIZE + j] = x_uint_32_right[
                    BLOCK_SIZE * i + j
                ]
        result_x_32 = result_x_32.reshape(M, N)
        # torch.set_printoptions(profile='full')
        # np.set_printoptions(threshold=22222222)
        # print("result_x_32", result_x_32)

        y_uint_32 = ((result_x_32 & 0x08) << 28) | ((result_x_32 & 0x07) << 22)
        y_float_32 = y_uint_32.view(dtype=np.float32)
        result_x = y_float_32 * (2**126)
        # 分步进行转换，确保数据类型正确
        result_x = result_x.astype(np.float32)
        scale1_float_32 = scale1_float_32.astype(np.float32)
        final_result = (
            (result_x * scale1_float_32).reshape(tensorNum, M // tensorNum * N)
            * input_np_scale2.reshape(tensorNum, 1)
        ).reshape(M, N)
        golden = torch.from_numpy(final_result.astype(np.float32)).to(torch.bfloat16)

        # torch.set_printoptions(profile='full')
        print("output_bf16 = ", output_bf16)
        print("golden = ", golden)

        # device_res=output_bf16.cpu()
        # for i in range(M):
        #     for j in range(N):
        #         if (device_res[i][j] != golden[i][j]):
        #             print("i j:",i,j)
        #             print("device_res[i][j]:", device_res[i][j])
        #             print("golden[i][j]:", golden[i][j])
        #             print("input_np_x[i][j//2]:",input_np_x[i][j//2])
        #             print(input_np_scale1[i][j//BLOCK_SIZE], scale1_float_32[i][j])
        #             print("input_np_scale2[i//(M//tensorNum)]:", input_np_scale2[i//(M//tensorNum)])
        #             print("result_x[i][j]:",result_x[i][j])
        #             assert(False)

        self.assertRtolEqual(output_bf16, golden)


if __name__ == "__main__":
    run_tests()
