
set -e
export ASCEND_CUSTOM_OPP_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")/vendors/customize
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

echo "==== start clean... ===="
rm -rf bin build outputs ../build ../grouped_gemm.egg-info ../kernel_meta