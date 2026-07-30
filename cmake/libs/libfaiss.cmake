include(CheckCXXCompilerFlag)

include_directories(thirdparty/faiss)

# all .cpp files
knowhere_file_glob(
  GLOB
  FAISS_SRCS
  thirdparty/faiss/faiss/*.cpp
  thirdparty/faiss/faiss/impl/*.cpp
  thirdparty/faiss/faiss/impl/fast_scan/*.cpp
  thirdparty/faiss/faiss/impl/hnsw/*.cpp
  thirdparty/faiss/faiss/impl/pq_code_distance/*.cpp
  thirdparty/faiss/faiss/impl/scalar_quantizer/*.cpp
  thirdparty/faiss/faiss/invlists/*.cpp
  thirdparty/faiss/faiss/utils/*.cpp
  thirdparty/faiss/faiss/utils/distances_fused/*.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/*.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/invlists/*.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/utils/*.cpp
)

if(WITH_SVS)
  knowhere_file_glob(
    GLOB
    FAISS_SVS_SRCS
    thirdparty/faiss/faiss/svs/*.cpp
    thirdparty/faiss/faiss/impl/svs_io.cpp
  )
  list(APPEND FAISS_SRCS ${FAISS_SVS_SRCS})
endif()

# AVX512 files
knowhere_file_glob(
  GLOB
  FAISS_AVX512_SRCS
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*avx512.cpp
)
# AVX512 vanilla Faiss dynamic dispatch related files. Baseline
# sq-avx512.cpp is replaced by a knowhere-local prelude file that declares
# a fast DCTemplate specialization for QT_4bit_uniform + L2 and then
# textually #includes the baseline sq-avx512.cpp \u2014 see
# cppcontrib/knowhere/impl/sq-avx512-fastpath.cpp for the full design note.
knowhere_file_glob(
  GLOB
  FAISS_DD_AVX512_SRCS
  thirdparty/faiss/faiss/impl/fast_scan/impl-avx512.cpp
  thirdparty/faiss/faiss/impl/hnsw/avx512.cpp
  thirdparty/faiss/faiss/impl/pq_code_distance/avx512.cpp
  thirdparty/faiss/faiss/impl/binary_hamming/avx512.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/sq-avx512-fastpath.cpp
  thirdparty/faiss/faiss/utils/distances_fused/avx512.cpp
  thirdparty/faiss/faiss/utils/hamming_distance/hamming_avx512.cpp
  thirdparty/faiss/faiss/utils/simd_impl/distances_avx512.cpp
  thirdparty/faiss/faiss/utils/simd_impl/rabitq_avx512.cpp
  thirdparty/faiss/faiss/utils/simd_impl/super_kmeans_kernels_avx512.cpp
)
# Baseline sq-avx512.cpp is pulled in textually by the prelude file, not
# compiled directly. Remove it from the generic list so it is not picked
# up as a stand-alone TU (which would duplicate symbols).
knowhere_file_glob(
  GLOB
  FAISS_SQ_AVX512_EXCLUDE
  thirdparty/faiss/faiss/impl/scalar_quantizer/sq-avx512.cpp
)
list(REMOVE_ITEM FAISS_SRCS ${FAISS_SQ_AVX512_EXCLUDE})
# combine files
list(APPEND FAISS_AVX512_SRCS ${FAISS_DD_AVX512_SRCS})
# remove platform files from general files
list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX512_SRCS})

# AVX512 Sapphire Rapids files
knowhere_file_glob(
  GLOB
  FAISS_AVX512_SPR_SRCS
  thirdparty/faiss/faiss/impl/scalar_quantizer/sq-avx512-spr.cpp
  thirdparty/faiss/faiss/utils/hamming_distance/hamming_avx512_spr.cpp
  thirdparty/faiss/faiss/utils/simd_impl/rabitq_avx512_spr.cpp
)
# remove platform files from general files
list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX512_SPR_SRCS})


# AVX2 files
knowhere_file_glob(
  GLOB
  FAISS_AVX2_SRCS
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*avx.cpp
)
# AVX2 vanilla Faiss dynamic dispatch related files. sq-avx2.cpp is
# textually wrapped by sq-avx2-fastpath.cpp (see design note there).
knowhere_file_glob(
  GLOB
  FAISS_DD_AVX2_SRCS
  thirdparty/faiss/faiss/impl/approx_topk/avx2.cpp
  thirdparty/faiss/faiss/impl/fast_scan/impl-avx2.cpp
  thirdparty/faiss/faiss/impl/hnsw/avx2.cpp
  thirdparty/faiss/faiss/impl/pq_code_distance/avx2.cpp
  thirdparty/faiss/faiss/impl/binary_hamming/avx2.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/sq-avx2-fastpath.cpp
  thirdparty/faiss/faiss/utils/distances_fused/simdlib_based.cpp
  thirdparty/faiss/faiss/utils/hamming_distance/hamming_avx2.cpp
  thirdparty/faiss/faiss/utils/simd_impl/distances_avx2.cpp
  thirdparty/faiss/faiss/utils/simd_impl/partitioning_avx2.cpp
  thirdparty/faiss/faiss/utils/simd_impl/rabitq_avx2.cpp
  thirdparty/faiss/faiss/utils/simd_impl/super_kmeans_kernels_avx2.cpp
)
knowhere_file_glob(
  GLOB
  FAISS_SQ_AVX2_EXCLUDE
  thirdparty/faiss/faiss/impl/scalar_quantizer/sq-avx2.cpp
)
list(REMOVE_ITEM FAISS_SRCS ${FAISS_SQ_AVX2_EXCLUDE})
# combine files
list(APPEND FAISS_AVX2_SRCS ${FAISS_DD_AVX2_SRCS})
# remove platform files from general files
list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX2_SRCS})


# FastScan files, which are supported on machine with a lookup
#   instructions
knowhere_file_glob(
  GLOB
  FAISS_FASTSCAN_SRCS
  thirdparty/faiss/faiss/impl/pq4_fast_scan_search_1.cpp
  thirdparty/faiss/faiss/impl/pq4_fast_scan_search_qbs.cpp
  thirdparty/faiss/faiss/IndexPQFastScan.cpp
  thirdparty/faiss/faiss/IndexIVFFastScan.cpp
  thirdparty/faiss/faiss/IndexIVFPQFastScan.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/IndexIVFPQFastScan.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/IVFFastScanIteratorWorkspace.cpp
)
# remove fastscan files from general files (they need special flags on x86)
list(REMOVE_ITEM FAISS_SRCS ${FAISS_FASTSCAN_SRCS})


# NEON files
knowhere_file_glob(
  GLOB
  FAISS_NEON_SRCS
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*neon.cpp
)
# NEON vanilla Faiss dynamic dispatch related files. sq-neon.cpp is
# textually wrapped by sq-neon-fastpath.cpp (see design note there).
knowhere_file_glob(
  GLOB
  FAISS_DD_NEON_SRCS
  thirdparty/faiss/faiss/impl/approx_topk/neon.cpp
  thirdparty/faiss/faiss/impl/fast_scan/impl-neon.cpp
  thirdparty/faiss/faiss/impl/binary_hamming/neon.cpp
  thirdparty/faiss/faiss/impl/pq_code_distance/neon.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/sq-neon-fastpath.cpp
  thirdparty/faiss/faiss/utils/distances_fused/simdlib_based_neon.cpp
  thirdparty/faiss/faiss/utils/hamming_distance/hamming_neon.cpp
  thirdparty/faiss/faiss/utils/simd_impl/distances_aarch64.cpp
  thirdparty/faiss/faiss/utils/simd_impl/partitioning_neon.cpp
  thirdparty/faiss/faiss/utils/simd_impl/rabitq_neon.cpp
)
knowhere_file_glob(
  GLOB
  FAISS_SQ_NEON_EXCLUDE
  thirdparty/faiss/faiss/impl/scalar_quantizer/sq-neon.cpp
)
list(REMOVE_ITEM FAISS_SRCS ${FAISS_SQ_NEON_EXCLUDE})
# combine files
list(APPEND FAISS_NEON_SRCS ${FAISS_DD_NEON_SRCS})
# remove platform files from general files
list(REMOVE_ITEM FAISS_SRCS ${FAISS_NEON_SRCS})


# SVE files
knowhere_file_glob(
  GLOB
  FAISS_SVE_SRCS
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*sve.cpp
)
# SVE vanilla Faiss dynamic dispatch related files
knowhere_file_glob(
  GLOB
  FAISS_DD_SVE_SRCS
  thirdparty/faiss/faiss/impl/pq_code_distance/pq_code_distance-sve.cpp
  thirdparty/faiss/faiss/utils/simd_impl/distances_arm_sve.cpp
)
# combine files
list(APPEND FAISS_SVE_SRCS ${FAISS_DD_SVE_SRCS})
# remove platform files from general files
list(REMOVE_ITEM FAISS_SRCS ${FAISS_SVE_SRCS})


# RVV files
knowhere_file_glob(
  GLOB
  FAISS_RVV_SRCS
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*rvv.cpp
)
# RVV vanilla Faiss dynamic dispatch related files
knowhere_file_glob(
  GLOB
  FAISS_DD_RVV_SRCS
  thirdparty/faiss/faiss/impl/fast_scan/impl-riscv.cpp
  thirdparty/faiss/faiss/impl/pq_code_distance/rvv.cpp
  thirdparty/faiss/faiss/impl/scalar_quantizer/sq-rvv.cpp
  thirdparty/faiss/faiss/impl/binary_hamming/rvv.cpp
  thirdparty/faiss/faiss/utils/simd_impl/distances_rvv.cpp
  thirdparty/faiss/faiss/utils/hamming_distance/hamming_rvv.cpp
  thirdparty/faiss/faiss/utils/simd_impl/rabitq_rvv.cpp
)
# combine files
list(APPEND FAISS_RVV_SRCS ${FAISS_DD_RVV_SRCS})
# remove platform files from general files
list(REMOVE_ITEM FAISS_SRCS ${FAISS_RVV_SRCS})


# disable RHNSW
knowhere_file_glob(GLOB FAISS_RHNSW_SRCS thirdparty/faiss/faiss/impl/RHNSW.cpp)
list(REMOVE_ITEM FAISS_SRCS ${FAISS_RHNSW_SRCS})

# generate `knowhere_utils` library for x86
if(__X86_64)
  set(UTILS_SRC src/simd/distances_ref.cc src/simd/hook.cc)
  set(UTILS_SSE_SRC src/simd/distances_sse.cc)
  set(UTILS_AVX_SRC src/simd/distances_avx.cc)
  set(UTILS_AVX512_SRC src/simd/distances_avx512.cc)
  set(UTILS_AVX512ICX_SRC src/simd/distances_avx512icx.cc)
  set(SPARSE_SIMD_AVX512_SRC src/simd/sparse_simd_avx512.cc)

  add_library(utils_sse OBJECT ${UTILS_SSE_SRC})
  add_library(utils_avx OBJECT ${UTILS_AVX_SRC})
  add_library(utils_avx512 OBJECT ${UTILS_AVX512_SRC})
  add_library(utils_avx512icx OBJECT ${UTILS_AVX512ICX_SRC})
  add_library(sparse_simd_avx512 OBJECT ${SPARSE_SIMD_AVX512_SRC})

  target_compile_options(utils_sse PRIVATE -msse4.2 -mpopcnt)
  target_compile_options(utils_avx PRIVATE -mfma -mf16c -mavx2 -mpopcnt)
  target_compile_options(utils_avx512 PRIVATE -mfma -mf16c -mavx512f -mavx512dq
                                              -mavx512bw -mpopcnt -mavx512vl)
  target_compile_options(utils_avx512icx PRIVATE -mfma -mf16c -mavx512f -mavx512dq
                                              -mavx512bw -mpopcnt -mavx512vl -mavx512vpopcntdq)
  target_compile_options(sparse_simd_avx512 PRIVATE -mavx512f -mavx512dq)
  target_include_directories(sparse_simd_avx512 PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(sparse_simd_avx512 PRIVATE milvus-common::milvus-common)

  add_library(
    knowhere_utils STATIC
    ${UTILS_SRC} $<TARGET_OBJECTS:utils_sse> $<TARGET_OBJECTS:utils_avx>
    $<TARGET_OBJECTS:utils_avx512> $<TARGET_OBJECTS:utils_avx512icx>
    $<TARGET_OBJECTS:sparse_simd_avx512>)
  target_link_libraries(knowhere_utils PUBLIC glog::glog)
  target_link_libraries(knowhere_utils PUBLIC xxHash::xxhash)
  target_link_libraries(knowhere_utils PUBLIC milvus-common::milvus-common)
endif()

# generate `knowhere_utils` library for AARCH64
if(__AARCH64)
  set(UTILS_SRC src/simd/distances_ref.cc src/simd/distances_neon.cc src/simd/hook.cc)
  set(UTILS_SVE_SRC src/simd/distances_sve.cc)
  set(ALL_UTILS_SRC ${UTILS_SRC} ${UTILS_SVE_SRC})

  add_library(
    knowhere_utils STATIC
    ${ALL_UTILS_SRC}
  )

  # Check for different ARM architecture and extension support
  check_cxx_compiler_flag("-march=armv9-a+sve+bf16" HAS_ARMV9_SVE_BF16)
  if (HAS_ARMV9_SVE_BF16)
    message(STATUS "SVE with BF16 for ARMv9: Found")
  else()
    message(STATUS "SVE with BF16 for ARMv9: Not Found")
  endif()

  check_cxx_compiler_flag("-march=armv9-a+sve" HAS_ARMV9_SVE)
  if (HAS_ARMV9_SVE)
    message(STATUS "SVE for ARMv9: Found")
  else()
    message(STATUS "SVE for ARMv9: Not Found")
  endif()

  check_cxx_compiler_flag("-march=armv8-a+sve+bf16" HAS_ARMV8_SVE_BF16)
  if (HAS_ARMV8_SVE_BF16)
    message(STATUS "SVE with BF16 for ARMv8: Found")
  else()
    message(STATUS "SVE with BF16 for ARMv8: Not Found")
  endif()

  check_cxx_compiler_flag("-march=armv8-a+sve" HAS_ARMV8_SVE)
  if (HAS_ARMV8_SVE)
    message(STATUS "SVE for ARMv8: Found")
  else()
    message(STATUS "SVE for ARMv8: Not Found")
  endif()

  if (APPLE)
    set(HAS_ARMV9_SVE_BF16 FALSE)
    set(HAS_ARMV9_SVE FALSE)
    set(HAS_ARMV8_SVE_BF16 FALSE)
    set(HAS_ARMV8_SVE FALSE)
    message(STATUS "Disable SVE for Apple")
  endif()

  set(KNOWHERE_SVE_COMPILE_OPTIONS "")
  if (HAS_ARMV9_SVE_BF16)
    set(KNOWHERE_SVE_COMPILE_OPTIONS "-march=armv9-a+sve+bf16")
  elseif (HAS_ARMV8_SVE_BF16)
    set(KNOWHERE_SVE_COMPILE_OPTIONS "-march=armv8-a+sve+bf16")
  elseif (HAS_ARMV9_SVE)
    set(KNOWHERE_SVE_COMPILE_OPTIONS "-march=armv9-a+sve")
  elseif (HAS_ARMV8_SVE)
    set(KNOWHERE_SVE_COMPILE_OPTIONS "-march=armv8-a+sve")
  endif()

  if (KNOWHERE_SVE_COMPILE_OPTIONS)
    foreach(SVE_FILE ${UTILS_SVE_SRC})
      set_source_files_properties(${SVE_FILE} PROPERTIES COMPILE_OPTIONS "${KNOWHERE_SVE_COMPILE_OPTIONS}")
    endforeach()
    set_source_files_properties(src/simd/hook.cc PROPERTIES COMPILE_DEFINITIONS KNOWHERE_USE_SVE)
    target_compile_options(knowhere_utils PRIVATE -march=armv8-a)
  else()
    message(WARNING "SVE not supported on this platform.")
    target_compile_options(knowhere_utils PRIVATE -march=armv8-a)
  endif()

  target_link_libraries(knowhere_utils PUBLIC glog::glog)
  target_link_libraries(knowhere_utils PUBLIC xxHash::xxhash)
  target_link_libraries(knowhere_utils PUBLIC milvus-common::milvus-common)
endif()

# generate `knowhere_utils` library for RISCV64
if(__RISCV64)
  set(UTILS_SRC src/simd/hook.cc src/simd/distances_ref.cc src/simd/distances_rvv.cc)
  add_library(knowhere_utils STATIC ${UTILS_SRC})
  target_link_libraries(knowhere_utils PUBLIC glog::glog)
  target_link_libraries(knowhere_utils PUBLIC xxHash::xxhash)
  target_link_libraries(knowhere_utils PUBLIC milvus-common::milvus-common)
  target_compile_options(knowhere_utils PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=rv64gcv_zvfhmin -mabi=lp64d>
    $<$<COMPILE_LANGUAGE:C>:-march=rv64gcv_zvfhmin -mabi=lp64d>
  )
endif()

# generate `knowhere_utils` library for PPC64
# ToDo: Add distances_vsx.cc for powerpc64 SIMD acceleration
if(__PPC64)
  set(UTILS_SRC src/simd/hook.cc src/simd/distances_ref.cc src/simd/distances_powerpc.cc)
  add_library(knowhere_utils STATIC ${UTILS_SRC})
  target_link_libraries(knowhere_utils PUBLIC glog::glog)
  target_link_libraries(knowhere_utils PUBLIC xxHash::xxhash)
  target_link_libraries(knowhere_utils PUBLIC milvus-common::milvus-common)
endif()


if(APPLE)
  set(BLA_VENDOR Apple)
  find_package(LAPACK REQUIRED)
  find_package(BLAS REQUIRED)
else()
  find_package(OpenBLAS CONFIG REQUIRED)
  set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
  set(LAPACK_LIBRARIES OpenBLAS::OpenBLAS)
endif()

find_package(xxHash REQUIRED)
include_directories(${xxHash_INCLUDE_DIRS})

# generate `faiss` library for x86
if(__X86_64)
  check_cxx_compiler_flag("-mavx512vpopcntdq" FAISS_COMPILER_SUPPORTS_AVX512VPOPCNTDQ)
  check_cxx_compiler_flag("-mavx512vnni" FAISS_COMPILER_SUPPORTS_AVX512VNNI)
  check_cxx_compiler_flag("-mavx512fp16" FAISS_COMPILER_SUPPORTS_AVX512FP16)
  check_cxx_compiler_flag("-mavx512bf16" FAISS_COMPILER_SUPPORTS_AVX512BF16)
  set(FAISS_ENABLE_AVX512_SPR FALSE)
  if(FAISS_COMPILER_SUPPORTS_AVX512VPOPCNTDQ
     AND FAISS_COMPILER_SUPPORTS_AVX512VNNI
     AND FAISS_COMPILER_SUPPORTS_AVX512FP16
     AND FAISS_COMPILER_SUPPORTS_AVX512BF16)
    set(FAISS_ENABLE_AVX512_SPR TRUE)
  else()
    message(STATUS "Skip Faiss AVX512_SPR: compiler does not support required SPR flags")
  endif()

  add_library(faiss_avx2 OBJECT ${FAISS_AVX2_SRCS} ${FAISS_FASTSCAN_SRCS})
  target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -msse4.2
                                            -mavx2 -mfma -mf16c -mpopcnt>)
  target_compile_definitions(faiss_avx2 PRIVATE COMPILE_SIMD_AVX2)
  target_include_directories(faiss_avx2 PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(faiss_avx2 PRIVATE milvus-common::milvus-common)
  add_library(faiss_avx512 OBJECT ${FAISS_AVX512_SRCS})
  target_compile_options(
    faiss_avx512
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -msse4.2
            -mavx2
            -mfma
            -mf16c
            -mavx512f
            -mavx512dq
            -mavx512bw
            -mavx512vl
            -mpopcnt>)
  target_compile_definitions(faiss_avx512 PRIVATE COMPILE_SIMD_AVX2 COMPILE_SIMD_AVX512)
  target_include_directories(faiss_avx512 PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(faiss_avx512 PRIVATE milvus-common::milvus-common)

  if(FAISS_ENABLE_AVX512_SPR)
    add_library(faiss_avx512_spr OBJECT ${FAISS_AVX512_SPR_SRCS})
    target_compile_options(
      faiss_avx512_spr
      PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
              -msse4.2
              -mavx2
              -mfma
              -mf16c
              -mavx512f
              -mavx512cd
              -mavx512vl
              -mavx512dq
              -mavx512bw
              -mavx512vpopcntdq
              -mavx512vnni
              -mavx512fp16
              -mavx512bf16
              -mpopcnt>)
    target_compile_definitions(faiss_avx512_spr PRIVATE
                               COMPILE_SIMD_AVX2 COMPILE_SIMD_AVX512 COMPILE_SIMD_AVX512_SPR)
    target_include_directories(faiss_avx512_spr PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(faiss_avx512_spr PRIVATE milvus-common::milvus-common)
  endif()

  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})

  add_dependencies(faiss faiss_avx2 faiss_avx512 knowhere_utils)
  if(FAISS_ENABLE_AVX512_SPR)
    add_dependencies(faiss faiss_avx512_spr)
  endif()
  target_compile_options(
    faiss
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -msse4.2
            -mpopcnt
            -mno-avx
            -mno-avx2
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)
  target_link_libraries(
    faiss PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                 faiss_avx2 faiss_avx512 knowhere_utils)
  if(FAISS_ENABLE_AVX512_SPR)
    target_link_libraries(faiss PUBLIC faiss_avx512_spr)
  endif()
  target_compile_definitions(faiss PRIVATE
                             FINTEGER=int FAISS_ENABLE_DD COMPILE_SIMD_AVX2 COMPILE_SIMD_AVX512)
  if(FAISS_ENABLE_AVX512_SPR)
    target_compile_definitions(faiss PRIVATE COMPILE_SIMD_AVX512_SPR)
  endif()

  if(WITH_SVS)
    # Use pre-built SVS runtime bindings (like baseline Faiss does).
    # The tarball ships libsvs_runtime.so with all deps (fmt, spdlog) baked in,
    # plus CMake config files and runtime API headers.
    # Override: set svs_runtime_DIR to a local unpacked tarball to skip download.
    find_package(svs_runtime 0.3.0 QUIET)
    if(NOT svs_runtime_FOUND)
      include(FetchContent)
      set(SVS_TARBALL_URL
          "https://github.com/intel/ScalableVectorSearch/releases/download/nightly/svs-cpp-runtime-bindings-nightly-2026-05-18-1299.tar.gz"
          CACHE STRING "URL for pre-built SVS runtime bindings tarball")
      FetchContent_Declare(svs URL "${SVS_TARBALL_URL}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
      FetchContent_MakeAvailable(svs)
      set(svs_runtime_DIR "${svs_SOURCE_DIR}/lib/cmake/svs_runtime")
      find_package(svs_runtime 0.3.0 REQUIRED)
    endif()

    target_link_libraries(faiss PUBLIC svs::svs_runtime)
    target_link_libraries(faiss_avx2 PUBLIC svs::svs_runtime)
    target_link_libraries(faiss_avx512 PUBLIC svs::svs_runtime)
    target_compile_definitions(faiss PUBLIC FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=v0)
    target_compile_definitions(faiss_avx2 PUBLIC FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=v0)
    target_compile_definitions(faiss_avx512 PUBLIC FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=v0)
    if(FAISS_ENABLE_AVX512_SPR)
      target_link_libraries(faiss_avx512_spr PUBLIC svs::svs_runtime)
      target_compile_definitions(faiss_avx512_spr PUBLIC FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=v0)
    endif()
  endif()
endif()

# generate `faiss` library for AARCH64
if(__AARCH64)
  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})
  target_sources(faiss PRIVATE ${FAISS_NEON_SRCS} ${FAISS_FASTSCAN_SRCS})

  target_compile_options(
    faiss
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)

  # SVE object library (compiled with SVE flags when available)
  set(SVE_AVAILABLE FALSE)
  if(HAS_ARMV9_SVE_BF16 OR HAS_ARMV8_SVE_BF16 OR HAS_ARMV9_SVE OR HAS_ARMV8_SVE)
    set(SVE_AVAILABLE TRUE)
  endif()

  if(SVE_AVAILABLE)
    add_library(faiss_sve OBJECT ${FAISS_SVE_SRCS})
    target_include_directories(faiss_sve PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(faiss_sve PRIVATE milvus-common::milvus-common)
    target_compile_definitions(faiss_sve PRIVATE
      FINTEGER=int FAISS_ENABLE_DD COMPILE_SIMD_ARM_NEON COMPILE_SIMD_ARM_SVE)
    target_compile_options(
      faiss_sve
      PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
              -Wno-sign-compare
              -Wno-unused-variable
              -Wno-reorder
              -Wno-unused-local-typedefs
              -Wno-unused-function
              -Wno-strict-aliasing>)
    if(HAS_ARMV9_SVE_BF16)
      target_compile_options(faiss_sve PRIVATE "-march=armv9-a+sve+bf16")
    elseif(HAS_ARMV8_SVE_BF16)
      target_compile_options(faiss_sve PRIVATE "-march=armv8-a+sve+bf16")
    elseif(HAS_ARMV9_SVE)
      target_compile_options(faiss_sve PRIVATE "-march=armv9-a+sve")
    elseif(HAS_ARMV8_SVE)
      target_compile_options(faiss_sve PRIVATE "-march=armv8-a+sve")
    endif()
  endif()

  add_dependencies(faiss knowhere_utils)
  target_link_libraries(faiss PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                                     knowhere_utils)
  if(SVE_AVAILABLE)
    target_link_libraries(faiss PUBLIC faiss_sve)
  endif()
  target_compile_definitions(faiss PRIVATE FINTEGER=int FAISS_ENABLE_DD COMPILE_SIMD_ARM_NEON)
endif()

# generate `faiss` library for RISCV64
if(__RISCV64)
  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})
  target_sources(faiss PRIVATE ${FAISS_RVV_SRCS} ${FAISS_FASTSCAN_SRCS})

  target_compile_options(
    faiss
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -march=rv64gcv_zvfhmin
            -mabi=lp64d
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)

  add_dependencies(faiss knowhere_utils)
  target_link_libraries(faiss PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES}
                                     ${LAPACK_LIBRARIES} knowhere_utils)
  target_compile_definitions(faiss PRIVATE FINTEGER=int COMPILE_SIMD_RISCV_RVV)
endif()

# generate `faiss` library for PPC64
if(__PPC64)
  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})
  target_sources(faiss PRIVATE ${FAISS_FASTSCAN_SRCS})

  target_compile_options(
    faiss
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -mcpu=native
            -mvsx
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)

  add_dependencies(faiss knowhere_utils)
  target_link_libraries(faiss PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                                      knowhere_utils)
  target_compile_definitions(faiss PRIVATE FINTEGER=int)
endif()

# GPU HNSW CUDA sources — compiled when WITH_CUVS is enabled.
#
# This `faiss_gpu_hnsw` OBJECT library is the ONLY place the faiss GPU sources
# are compiled in the knowhere build:
#   * The main `faiss` STATIC target above is built from FAISS_SRCS, which globs
#     only thirdparty/faiss/faiss/{*.cpp,impl,utils,...} — it deliberately does
#     NOT include thirdparty/faiss/faiss/gpu/*, so none of the files below are
#     also compiled into `faiss`.
#   * Upstream faiss ships its own GPU build file
#     (thirdparty/faiss/faiss/gpu/CMakeLists.txt → the `faiss_gpu_objs` target
#     over FAISS_GPU_SRC). Knowhere assembles faiss manually via this .cmake and
#     never `add_subdirectory()`s the vendored faiss tree, so that file (and
#     `faiss_gpu_objs`) is never part of the knowhere build graph.
# Therefore the two GPU build paths are mutually exclusive here — there is no
# duplicate compilation or ODR hazard. Keep it that way: add GPU sources to the
# list below, NOT by pulling in the vendored gpu/CMakeLists.txt.
if(WITH_CUVS)
  set(FAISS_GPU_HNSW_SRCS
    thirdparty/faiss/faiss/gpu/GpuIndexHNSW.cu
    thirdparty/faiss/faiss/gpu/GpuIndex.cu
    thirdparty/faiss/faiss/gpu/GpuResources.cpp
    thirdparty/faiss/faiss/gpu/StandardGpuResources.cpp
    thirdparty/faiss/faiss/gpu/impl/GpuHnswTypes.cu
    thirdparty/faiss/faiss/gpu/impl/IndexUtils.cu
    thirdparty/faiss/faiss/gpu/utils/DeviceUtils.cu
    thirdparty/faiss/faiss/gpu/utils/StackDeviceMemory.cpp
    thirdparty/faiss/faiss/gpu/utils/Timer.cpp
  )
  add_library(faiss_gpu_hnsw OBJECT ${FAISS_GPU_HNSW_SRCS})
  # These objects are linked into the STATIC `faiss` target which is in turn
  # linked into the shared libknowhere.so. The global -fPIC in
  # cmake/utils/compile_flags.cmake only reaches the C++ host compiler, not the
  # CUDA (.cu) sources built by nvcc, so their objects are non-PIC and trigger
  # `relocation R_X86_64_PC32 ... can not be used when making a shared object`
  # (surfaces in clean/Debug test builds). Mirror upstream faiss_gpu_objs and
  # mark the target PIC so CMake passes -Xcompiler -fPIC to nvcc.
  set_target_properties(faiss_gpu_hnsw PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(faiss_gpu_hnsw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/faiss
    ${Boost_INCLUDE_DIRS}
  )
  target_compile_definitions(faiss_gpu_hnsw PRIVATE FINTEGER=int)
  target_link_libraries(faiss_gpu_hnsw PRIVATE CUDA::cudart)
  target_link_libraries(faiss PUBLIC faiss_gpu_hnsw)
endif()
