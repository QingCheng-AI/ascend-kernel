/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelFp8ToFloat2 {
  public:
    __aicore__ inline KernelFp8ToFloat2() {}
    __aicore__ inline ~KernelFp8ToFloat2() {
        inQueueY.FreeTensor(this->yLocal);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t M,
                                uint32_t N, uint32_t BLOCK_SIZE) {
        uint32_t blockDim = (M + BLOCK_SIZE - 1) / 128;
        uint32_t threadId = AscendC::GetBlockIdx();
        threadId_i = threadId / blockDim;
        threadId_j = threadId % blockDim;
        // AscendC::printf("threadId=%d threadId_i=%d threadId_j=%d\n",
        // threadId, threadId_i, threadId_j);

        this->BLOCK_SIZE = BLOCK_SIZE;
        this->sub_m = BLOCK_SIZE;
        this->sub_n = N;
        this->tileLength = BLOCK_SIZE;

        this->scale_m = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
        this->scale_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        this->scale_size = this->scale_m * this->scale_n;
        uint32_t scale_size_64 = ((this->scale_size + 63) / 64) * 64;

        uint32_t m_start =
            threadId_i * (M * N) + threadId_j * (this->sub_m * this->sub_n);
        uint32_t scale_start = threadId_i * this->scale_size;

        uint32_t tail_m = M - threadId_j * this->sub_m;
        this->sub_m = tail_m < this->sub_m ? tail_m : this->sub_m;
        this->blockLength = this->sub_m * this->sub_n;
        this->tileNum = this->blockLength / this->tileLength;

        // AscendC::printf(" -------- blockId=%d, sub_m=%d, sub_n=%d, tileNum=%d
        // tileLength=%d, scale_m=%d, scale_n=%d, scale_size=%d m_start=%d\n",
        //         AscendC::GetBlockIdx(), this->sub_m, this->sub_n,
        //         this->tileNum, this->tileLength, this->scale_m,
        //         this->scale_n, this->scale_size, m_start);

        xGm.SetGlobalBuffer((__gm__ uint8_t *)x + m_start, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + scale_start, scale_size_64);
        zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + m_start,
                            this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM,
                        this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(inQueueY, 1, scale_size_64 * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM,
                        this->tileLength * sizeof(bfloat16_t));

        pipe.InitBuffer(tmpBuf_half, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf_float, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(uint32_t));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(uint32_t));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(uint32_t));

        // 将本block需要的scale一次全部加载上来
        this->yLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(this->yLocal, yGm[0], scale_size_64);
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<uint8_t> xLocal = inQueueX.AllocTensor<uint8_t>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength],
                          this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<uint8_t> xLocal = inQueueX.DeQue<uint8_t>();
        AscendC::LocalTensor<bfloat16_t> zLocal =
            outQueueZ.AllocTensor<bfloat16_t>();

        // 【step1:将uint8数值转成uint32数值】
        AscendC::LocalTensor<half> tmpTensor_half = tmpBuf_half.Get<half>();
        AscendC::Cast(tmpTensor_half, xLocal, AscendC::RoundMode::CAST_NONE,
                      this->tileLength);
        AscendC::LocalTensor<uint32_t> tmpTensor = tmpBuf.Get<uint32_t>();
        AscendC::Cast(tmpTensor.ReinterpretCast<int32_t>(), tmpTensor_half,
                      AscendC::RoundMode::CAST_RINT, this->tileLength);

        // 【step2:通过移位和与，将uint32数值的二进制 转成 浮点数格式】
        AscendC::LocalTensor<uint32_t> tmpTensor0 = tmpBuf0.Get<uint32_t>();
        AscendC::Duplicate(tmpTensor0, (uint32_t)0x80, this->tileLength);
        tmpTensor0 = (tmpTensor & tmpTensor0);
        AscendC::ShiftLeft(tmpTensor0, tmpTensor0, uint32_t(24),
                           this->tileLength);

        AscendC::LocalTensor<uint32_t> tmpTensor1 = tmpBuf1.Get<uint32_t>();
        AscendC::Duplicate(tmpTensor1, (uint32_t)0x7F, this->tileLength);
        tmpTensor1 = (tmpTensor & tmpTensor1);
        AscendC::ShiftLeft(tmpTensor1, tmpTensor1, uint32_t(20),
                           this->tileLength);

        tmpTensor = tmpTensor0 | tmpTensor1;

        // 【step3:将上面的转换结果，通过ReinterpretCast<float>读取出来,并乘以
        // 2**120】
        AscendC::Duplicate(tmpTensor0, (uint32_t)0x7b800000, this->tileLength);
        AscendC::LocalTensor<float> tmpTensor_float = tmpBuf_float.Get<float>();
        tmpTensor_float = tmpTensor.ReinterpretCast<float>() *
                          tmpTensor0.ReinterpretCast<float>();

        // 【step4: 将结果乘以 scale】
        int32_t index =
            (threadId_j + progress / (this->BLOCK_SIZE * this->scale_n)) *
                this->scale_n +
            (progress % this->scale_n);
        AscendC::Duplicate(tmpTensor0.ReinterpretCast<float>(),
                           (float)yLocal.GetValue(index), this->tileLength);
        // AscendC::Duplicate(tmpTensor0.ReinterpretCast<float>(), (float)1.0,
        // this->tileLength);
        tmpTensor_float = tmpTensor_float * tmpTensor0.ReinterpretCast<float>();

        // 【写回结果】
        AscendC::Cast(zLocal, tmpTensor_float, AscendC::RoundMode::CAST_ROUND,
                      this->tileLength);
        outQueueZ.EnQue<bfloat16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.DeQue<bfloat16_t>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal,
                          this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

  private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<uint8_t> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<bfloat16_t> zGm;

    AscendC::LocalTensor<float> yLocal;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_half;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_float;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;

    uint32_t BLOCK_SIZE;
    uint32_t threadId_i;
    uint32_t threadId_j;
    uint32_t sub_m;
    uint32_t sub_n;
    uint32_t scale_m;
    uint32_t scale_n;
    uint32_t scale_size;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void
fp8_to_bf16_custom2(GM_ADDR x, GM_ADDR scale, GM_ADDR z, uint32_t M, uint32_t N,
                    uint32_t BLOCK_SIZE) {
    KernelFp8ToFloat2 op;
    op.Init(x, scale, z, M, N, BLOCK_SIZE);
    op.Process();
}
