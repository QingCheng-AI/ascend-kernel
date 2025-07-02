# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)

# project information
project(acl_execute_grouped_gemm)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../bin")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

# set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
set(CUST_PKG_PATH "$ENV{ASCEND_CUSTOM_OPP_PATH}/op_api")
if (NOT DEFINED ENV{ASCEND_CUSTOM_OPP_PATH})
    set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
    message(STATUS "set default CUST_PKG_PATH: ${CUST_PKG_PATH}")
else ()
    message(STATUS "env CUST_PKG_PATH: ${CUST_PKG_PATH}")
endif()


set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/devlib")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ../inc
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_grouped_gemm
    src/test_grouped_gemm.cpp
    src/grouped_gemm_utils.cpp
)

target_link_libraries(execute_grouped_gemm
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_grouped_gemm DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})