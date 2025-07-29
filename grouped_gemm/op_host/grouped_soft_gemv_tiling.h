#include "inc/error/ops_error.h"
#include "inc/log/ops_log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include <climits>
#include <graph/utils/type_utils.h>

using namespace ge;
using namespace AscendC;

namespace optiling {
BEGIN_TILING_DATA_DEF(GMVBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseK);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseN);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseM);
TILING_DATA_FIELD_DEF(uint32_t, ubXInSize);
TILING_DATA_FIELD_DEF(uint32_t, ubWInSize);
TILING_DATA_FIELD_DEF(uint32_t, ubOutSize);
TILING_DATA_FIELD_DEF(uint32_t, ubSInSize);
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint64_t, workspaceSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupedSoftGemv, GMVBaseParams)
extern "C" ge::graphStatus TilingGMV(gert::TilingContext *context);
} // namespace optiling