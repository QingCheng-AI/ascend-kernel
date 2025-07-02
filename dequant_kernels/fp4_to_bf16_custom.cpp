#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr int32_t TILE_SIZE = 256;

class KernelFp4ToBf16 {
  public:
    __aicore__ inline KernelFp4ToBf16() {}
    __aicore__ inline ~KernelFp4ToBf16() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR scale1,
                                float scale2, uint32_t M, uint32_t N,
                                uint32_t BLOCK_SIZE) {
        this->BLOCK_SIZE = BLOCK_SIZE;
        this->size = M * N;
        this->blockLength = this->size / 2 / AscendC::GetBlockNum();
        this->tileLength = TILE_SIZE;
        this->tileNum = this->blockLength / this->tileLength;
        this->blockPerTile = this->tileLength / BLOCK_SIZE;

        this->scale_size = this->tileLength * 2 / BLOCK_SIZE;
        this->scaleBlockLength =
            this->size / BLOCK_SIZE / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ uint8_t *)x +
                                this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        scaleGm.SetGlobalBuffer((__gm__ uint8_t *)scale1 +
                                    this->scaleBlockLength *
                                        AscendC::GetBlockIdx(),
                                this->scaleBlockLength);
        zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + (2 * this->blockLength) *
                                                         AscendC::GetBlockIdx(),
                            2 * this->blockLength);

        pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(inQueueScale, 1, this->scale_size * sizeof(uint8_t));
        pipe.InitBuffer(outQueueZ, 1,
                        2 * this->tileLength * sizeof(bfloat16_t));
        this->scale2 = scale2;

        pipe.InitBuffer(tmpBuf, 32 * this->tileLength);
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
        // DumpTensor(xLocal,5 ,this->tileLength);
        inQueueX.EnQue(xLocal);
        AscendC::LocalTensor<uint8_t> scaleLocal =
            inQueueScale.AllocTensor<uint8_t>();
        AscendC::DataCopy(scaleLocal, scaleGm[progress * this->scale_size],
                          this->scale_size);
        inQueueScale.EnQue(scaleLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<uint8_t> xLocal = inQueueX.DeQue<uint8_t>();
        AscendC::LocalTensor<uint8_t> scaleLocal =
            inQueueScale.DeQue<uint8_t>();
        AscendC::LocalTensor<bfloat16_t> zLocal =
            outQueueZ.AllocTensor<bfloat16_t>();
        // 从tmpBuf分配中间变量的空间
        AscendC::LocalTensor<uint8_t> tmpBufHeader = tmpBuf.Get<uint8_t>();
        AscendC::LocalTensor<float> scale1 =
            tmpBufHeader.ReinterpretCast<float>();
        tmpBufHeader =
            tmpBufHeader[this->scale_size * this->BLOCK_SIZE * sizeof(float)];
        AscendC::LocalTensor<uint32_t> tmpConst0 =
            tmpBufHeader.ReinterpretCast<uint32_t>();
        tmpBufHeader = tmpBufHeader[2 * this->tileLength * sizeof(uint32_t)];
        AscendC::LocalTensor<uint32_t> tmpConst1 =
            tmpBufHeader.ReinterpretCast<uint32_t>();
        tmpBufHeader = tmpBufHeader[2 * this->tileLength * sizeof(uint32_t)];
        AscendC::LocalTensor<uint32_t> xUint32 =
            tmpBufHeader.ReinterpretCast<uint32_t>();
        // 处理scale
        // 先将uint8_t存储的fp8格式的scale转成float
        // scaleHalf, scaleUint32, scaleFloat共用一块内存，都暂时用zLocal存放
        AscendC::LocalTensor<half> scaleHalf = zLocal.ReinterpretCast<half>();
        AscendC::Cast(scaleHalf, scaleLocal, AscendC::RoundMode::CAST_NONE,
                      this->scale_size);
        AscendC::LocalTensor<uint32_t> scaleUint32 =
            scaleHalf.ReinterpretCast<uint32_t>();
        AscendC::Cast(scaleUint32.ReinterpretCast<int32_t>(), scaleHalf,
                      AscendC::RoundMode::CAST_RINT, this->scale_size);
        // ((scale1_uint_32 & 0x80) << 24) | ((scale1_uint_32 & 0x7F) << 20)
        AscendC::Duplicate(tmpConst0, (uint32_t)0x80, this->scale_size);
        AscendC::And(tmpConst0.ReinterpretCast<int16_t>(),
                     scaleUint32.ReinterpretCast<int16_t>(),
                     tmpConst0.ReinterpretCast<int16_t>(),
                     2 * this->scale_size);
        AscendC::ShiftLeft(tmpConst0, tmpConst0, uint32_t(24),
                           this->scale_size);
        AscendC::Duplicate(tmpConst1, uint32_t(0x7F), this->scale_size);
        AscendC::And(tmpConst1.ReinterpretCast<int16_t>(),
                     scaleUint32.ReinterpretCast<int16_t>(),
                     tmpConst1.ReinterpretCast<int16_t>(),
                     2 * this->scale_size);
        AscendC::ShiftLeft(tmpConst1, tmpConst1, uint32_t(20),
                           this->scale_size);
        AscendC::Or(scaleUint32.ReinterpretCast<int16_t>(),
                    tmpConst0.ReinterpretCast<int16_t>(),
                    tmpConst1.ReinterpretCast<int16_t>(), 2 * this->scale_size);
        union U const_value;
        const_value.i = 0x7b800000;
        AscendC::LocalTensor<float> scaleFloat =
            scaleUint32.ReinterpretCast<float>();
        AscendC::Muls(scaleFloat, scaleFloat, this->scale2 * const_value.f,
                      this->scale_size);
        uint32_t srcShape[2] = {this->scale_size, 1};
        uint32_t dstShape[2] = {this->scale_size, this->BLOCK_SIZE};
        AscendC::BroadCast<float, 2, 1>(scale1, scaleFloat, dstShape, srcShape);
        // 【将uint8数值转成uint32数值】
        // xHalf刚好可以放进zLocal中
        AscendC::LocalTensor<half> xHalf = zLocal.ReinterpretCast<half>();
        AscendC::Cast(xHalf, xLocal, AscendC::RoundMode::CAST_NONE,
                      this->tileLength);
        AscendC::LocalTensor<uint32_t> xUint32Left = xUint32;
        AscendC::LocalTensor<uint32_t> xUint32Right =
            xUint32Left[this->tileLength];
        AscendC::Cast(xUint32Left.ReinterpretCast<int32_t>(), xHalf,
                      AscendC::RoundMode::CAST_RINT, this->tileLength);
        AscendC::Cast(xUint32Right.ReinterpretCast<int32_t>(), xHalf,
                      AscendC::RoundMode::CAST_RINT, this->tileLength);
        // 将由两个half组成个int8分隔开来，每个元素占一个uint32
        // 用tmpConst而非xUint32存放中间结果，能更节省空间
        AscendC::Duplicate(tmpConst0, uint32_t(0xF0), this->tileLength);
        AscendC::And(tmpConst0.ReinterpretCast<int16_t>(),
                     xUint32Left.ReinterpretCast<int16_t>(),
                     tmpConst0.ReinterpretCast<int16_t>(),
                     2 * this->tileLength);
        AscendC::ShiftRight(tmpConst0, tmpConst0, uint32_t(4),
                            this->tileLength);
        AscendC::Duplicate(tmpConst1, uint32_t(0x0F), this->tileLength);
        AscendC::And(tmpConst1.ReinterpretCast<int16_t>(),
                     xUint32Right.ReinterpretCast<int16_t>(),
                     tmpConst1.ReinterpretCast<int16_t>(),
                     2 * this->tileLength);
        AscendC::DataCopy(xUint32Left, tmpConst0,
                          {uint16_t(this->blockPerTile),
                           uint16_t(this->BLOCK_SIZE * sizeof(uint32_t) / 32),
                           0,
                           uint16_t(this->BLOCK_SIZE * sizeof(uint32_t) / 32)});
        AscendC::DataCopy(xUint32Left[this->BLOCK_SIZE], tmpConst1,
                          {uint16_t(this->blockPerTile),
                           uint16_t(this->BLOCK_SIZE * sizeof(uint32_t) / 32),
                           0,
                           uint16_t(this->BLOCK_SIZE * sizeof(uint32_t) / 32)});

        // 【step2:通过移位和与，将uint32数值的二进制 转成 浮点数格式】
        // ((result_x_32 & 0x08) << 28) | ((result_x_32 & 0x07) << 22)
        AscendC::Duplicate(tmpConst0, uint32_t(0x08), 2 * this->tileLength);
        AscendC::And(tmpConst0.ReinterpretCast<int16_t>(),
                     xUint32.ReinterpretCast<int16_t>(),
                     tmpConst0.ReinterpretCast<int16_t>(),
                     4 * this->tileLength);
        AscendC::ShiftLeft(tmpConst0, tmpConst0, uint32_t(28),
                           2 * this->tileLength);
        AscendC::Duplicate(tmpConst1, uint32_t(0x07), 2 * this->tileLength);
        AscendC::And(tmpConst1.ReinterpretCast<int16_t>(),
                     xUint32.ReinterpretCast<int16_t>(),
                     tmpConst1.ReinterpretCast<int16_t>(),
                     4 * this->tileLength);
        AscendC::ShiftLeft(tmpConst1, tmpConst1, uint32_t(22),
                           2 * this->tileLength);
        AscendC::Or(xUint32.ReinterpretCast<int16_t>(),
                    tmpConst0.ReinterpretCast<int16_t>(),
                    tmpConst1.ReinterpretCast<int16_t>(), 4 * this->tileLength);

        // 【step3:将上面的转换结果，通过ReinterpretCast<float>读取出来,并乘以
        // 2**120】
        AscendC::LocalTensor<float> xFloat = xUint32.ReinterpretCast<float>();
        const_value.i = 0x7E800000;
        AscendC::Muls(xFloat, xFloat, const_value.f, 2 * this->tileLength);

        // 【step4: 将结果乘以 scale】
        AscendC::Mul(xFloat, xFloat, scale1, 2 * this->tileLength);

        // 【写回结果】
        AscendC::Cast(zLocal, xFloat, AscendC::RoundMode::CAST_ROUND,
                      2 * this->tileLength);
        outQueueZ.EnQue<bfloat16_t>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueScale.FreeTensor(scaleLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.DeQue<bfloat16_t>();
        AscendC::DataCopy(zGm[2 * progress * this->tileLength], zLocal,
                          2 * this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

  private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX, inQueueScale,
        inQueueScale2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<uint8_t> xGm;
    AscendC::GlobalTensor<uint8_t> scaleGm;
    AscendC::GlobalTensor<bfloat16_t> zGm;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    union U {
        uint32_t i;
        float f;
    };
    float scale2;
    // uint8_t scale1[128];
    uint32_t BLOCK_SIZE;
    uint32_t sub_m;
    uint32_t sub_n;
    uint32_t size;
    uint32_t scale_m;
    uint32_t scale_n;
    uint32_t scale_size;
    uint32_t blockLength;
    uint32_t scaleBlockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t blockPerTile;
};

extern "C" __global__ __aicore__ void
fp4_to_bf16_custom(GM_ADDR x, GM_ADDR z, GM_ADDR scale1, float scale2,
                   uint32_t M, uint32_t N, uint32_t BLOCK_SIZE) {
    KernelFp4ToBf16 op;
    op.Init(x, z, scale1, scale2, M, N, BLOCK_SIZE);
    op.Process();
}
