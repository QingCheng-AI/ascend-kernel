#include "grouped_soft_gemv.h"
#include "kernel_operator.h"

using namespace AscendC;
using namespace GROUPED_GEMV;

#if ORIG_DTYPE_WEIGHT == DT_UINT8
#define DTYPE_WEIGHT int8_t
#define DTYPE_SCALE float
#else
#define DTYPE_WEIGHT int4b_t
#define ORIG_DTYPE_WEIGHT DT_INT4
#define DTYPE_SCALE bfloat16_t
#endif

using xType = DTYPE_X;

using weightType = DTYPE_WEIGHT;

using yType = DTYPE_X;

using scaleType = DTYPE_SCALE;

extern "C" __global__ __aicore__ void
grouped_soft_gemv(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR group_list,
                  GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);
    GET_TILING_DATA(gmvBaseParams, tiling);
    using mvType = MVType<xType, weightType, xType, yType, scaleType>;
    GMVCompute<mvType> computeOp;
    computeOp.Init(x, weight, scale, group_list, y, workspace, &gmvBaseParams,
                   &tPipe, tiling);
    computeOp.GEMVProcess();
}
