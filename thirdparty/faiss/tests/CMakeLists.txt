# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

###############################################################################
# copy from knowhere/utils/CMakeLists.txt
set(UTILS_SRC
        ../../../knowhere/utils/distances_simd.cpp
        ../../../knowhere/utils/FaissHookFvec.cpp
        )

set(UTILS_SSE_SRC
        ../../../knowhere/utils/distances_simd_sse.cpp
        )
set(UTILS_AVX_SRC
        ../../../knowhere/utils/distances_simd_avx.cpp
        )
set(UTILS_AVX512_SRC
        ../../../knowhere/utils/distances_simd_avx512.cpp
        )

add_library(utils_sse OBJECT
        ${UTILS_SSE_SRC}
        )
add_library(utils_avx OBJECT
        ${UTILS_AVX_SRC}
        )
add_library(utils_avx512 OBJECT
        ${UTILS_AVX512_SRC}
        )

target_compile_options(utils_sse PUBLIC "-msse4.2")
target_compile_options(utils_avx PUBLIC "-mf16c;-mavx2")
target_compile_options(utils_avx512 PUBLIC "-mf16c;-mavx512f;-mavx512dq;-mavx512bw")

add_library(knowhere_utils STATIC
        ${UTILS_SRC}
        $<TARGET_OBJECTS:utils_sse>
        $<TARGET_OBJECTS:utils_avx>
        $<TARGET_OBJECTS:utils_avx512>
        )

target_include_directories(knowhere_utils PUBLIC ${KNOWHERE_SOURCE_DIR}/knowhere/utils)
###############################################################################

set(FAISS_TEST_SRC
  test_binary_flat.cpp
  test_dealloc_invlists.cpp
  test_ivfpq_codec.cpp
  test_ivfpq_indexing.cpp
  test_lowlevel_ivf.cpp
  test_merge.cpp
  test_omp_threads.cpp
  test_ondisk_ivf.cpp
  test_pairs_decoding.cpp
  test_params_override.cpp
  test_pq_encoding.cpp
  test_sliding_ivf.cpp
  test_threaded_index.cpp
  test_transfer_invlists.cpp
  # test_mem_leak.cpp
)

add_executable(faiss_test ${FAISS_TEST_SRC})

if(FAISS_OPT_LEVEL STREQUAL "avx2")
  target_link_libraries(faiss_test PRIVATE faiss_avx2)
else()
  target_link_libraries(faiss_test PRIVATE faiss)
endif()

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

target_link_libraries(faiss_test PRIVATE
  OpenMP::OpenMP_CXX
  gtest_main
  knowhere_utils
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(faiss_test)

install(TARGETS faiss_test DESTINATION unittest)