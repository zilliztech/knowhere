# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(GTest REQUIRED)
set(unittest_libs
        GTest::gtest
        GTest::gtest_main)

set(depend_libs
        knowhere
        ${HDF5_LIBRARIES}
        )

if(WITH_PROFILER)
    set(depend_libs
        ${depend_libs}
        tcmalloc_and_profiler
    )
endif()

#==============================================================================
macro(benchmark_test target file)
    set(FILE_SRCS ${file})
    add_executable(${target} ${FILE_SRCS})
    target_link_libraries(${target} ${depend_libs} ${unittest_libs})

    # this is needed for clang in Debug compilation mode
    target_link_libraries(${target} atomic)

    install(TARGETS ${target} DESTINATION unittest)
endmacro()

benchmark_test(benchmark_binary                hdf5/benchmark_binary.cpp)
benchmark_test(benchmark_binary_range          hdf5/benchmark_binary_range.cpp)
benchmark_test(benchmark_float                 hdf5/benchmark_float.cpp)
benchmark_test(benchmark_float_bitset          hdf5/benchmark_float_bitset.cpp)
benchmark_test(benchmark_float_qps             hdf5/benchmark_float_qps.cpp)
benchmark_test(benchmark_float_range           hdf5/benchmark_float_range.cpp)
benchmark_test(benchmark_float_range_bitset    hdf5/benchmark_float_range_bitset.cpp)
benchmark_test(benchmark_simd_qps              hdf5/benchmark_simd_qps.cpp)

benchmark_test(gen_hdf5_file hdf5/gen_hdf5_file.cpp)
benchmark_test(gen_fbin_file hdf5/gen_fbin_file.cpp)

# Sparse algorithm benchmark (standalone, no HDF5 required)
# Compares DAAT_MAXSCORE vs DAAT_MAXSCORE_V2 using MSMARCO/SPLADE CSR data
add_executable(benchmark_sparse_algo benchmark_sparse_algo.cpp)
target_link_libraries(benchmark_sparse_algo knowhere)
install(TARGETS benchmark_sparse_algo DESTINATION unittest)

# Sparse SIMD benchmark (x86_64 only, standalone, no HDF5 required)
# Only build on x86_64/AMD64, skip on ARM/aarch64/arm64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|X86_64)$")
    message(STATUS "Building sparse SIMD benchmark for ${CMAKE_SYSTEM_PROCESSOR}")
    add_executable(benchmark_sparse_simd benchmark_sparse_simd.cpp)
    target_link_libraries(benchmark_sparse_simd knowhere)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(benchmark_sparse_simd PRIVATE -mavx512f -mavx512dq)
    endif()
    install(TARGETS benchmark_sparse_simd DESTINATION unittest)
else()
    message(STATUS "Skipping sparse SIMD benchmark on ${CMAKE_SYSTEM_PROCESSOR} (x86_64 only)")
endif()
