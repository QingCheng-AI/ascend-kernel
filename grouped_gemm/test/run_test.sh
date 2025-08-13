#!/bin/bash
set -e

export ASCEND_CUSTOM_OPP_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")/vendors/customize

MODE="python"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [-m|--mode cpp|python]"
            exit 1
            ;;
    esac
done

if [[ "$MODE" == "cpp" ]]; then
    cmake -B build
    cmake --build build -j
    mkdir -p outputs
    echo "==== generating data ===="
    python3 python/gen_data.py outputs
    echo "==== generating data done ===="
    echo "==== executing test ===="
    ./bin/execute_grouped_gemm outputs
    echo "==== executing test done ===="
    python3 python/print_result.py outputs
elif [[ "$MODE" == "python" ]]; then
    python3 python/test_group_gemm_fp4.py
    python3 python/test_group_gemm_fp8.py
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [-m|--mode cpp|python]"
    exit 1
fi