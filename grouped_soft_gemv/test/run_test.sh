#!/bin/bash
set -e

export ASCEND_CUSTOM_OPP_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")/vendors/customize

python3 python/test_group_soft_gemv_fp4.py
python3 python/test_group_soft_gemv_fp8.py
