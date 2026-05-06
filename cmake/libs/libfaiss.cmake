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
# textually #includes the baseline sq-avx512.cpp — see
# cppcontrib/knowhere/impl/sq-avx512-fastpath.cpp for the full design note.
knowhere_file_glob(
  GLOB
  FAISS_DD_AVX512_SRCS
  thirdparty/faiss/faiss/impl/fast_scan/impl-avx512.cpp
  thirdparty/faiss/faiss/impl/hnsw/avx512.cpp
  thirdparty/faiss/faiss/impl/pq_code_distance/pq_code_distance-avx512.cpp
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
  thirdparty/faiss/faiss/impl/pq_code_distance/pq_code_distance-avx2.cpp
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

  add_library(
    knowhere_utils STATIC
    ${UTILS_SRC} $<TARGET_OBJECTS:utils_sse> $<TARGET_OBJECTS:utils_avx>
    $<TARGET_OBJECTS:utils_avx512> $<TARGET_OBJECTS:utils_avx512icx>
    $<TARGET_OBJECTS:sparse_simd_avx512>)
  target_link_libraries(knowhere_utils PUBLIC glog::glog)
  target_link_libraries(knowhere_utils PUBLIC xxHash::xxhash)
endif()

# generate `knowhere_utils` library for AARCH64
if(__AARCH64)
  set(UTILS_SRC src/simd/distances_ref.cc src/simd/distances_neon.cc)
  set(UTILS_SVE_SRC src/simd/hook.cc src/simd/distances_sve.cc)
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

  if (HAS_ARMV9_SVE_BF16)
    foreach(SVE_FILE ${UTILS_SVE_SRC})
      set_source_files_properties(${SVE_FILE} PROPERTIES COMPILE_OPTIONS "-march=armv9-a+sve+bf16")
      target_compile_options(knowhere_utils PRIVATE -march=armv8-a)
    endforeach()
  elseif (HAS_ARMV8_SVE_BF16)
    foreach(SVE_FILE ${UTILS_SVE_SRC})
      set_source_files_properties(${SVE_FILE} PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sve+bf16")
      target_compile_options(knowhere_utils PRIVATE -march=armv8-a)
    endforeach()
  elseif (HAS_ARMV9_SVE)
    foreach(SVE_FILE ${UTILS_SVE_SRC})
      set_source_files_properties(${SVE_FILE} PROPERTIES COMPILE_OPTIONS "-march=armv9-a+sve")
      target_compile_options(knowhere_utils PRIVATE -march=armv8-a)
    endforeach()
  elseif (HAS_ARMV8_SVE)
    foreach(SVE_FILE ${UTILS_SVE_SRC})
      set_source_files_properties(${SVE_FILE} PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sve")
      target_compile_options(knowhere_utils PRIVATE -march=armv8-a)
    endforeach()
  else()
    message(WARNING "SVE not supported on this platform.")
    target_compile_options(knowhere_utils PRIVATE -march=armv8-a)
  endif()

  target_link_libraries(knowhere_utils PUBLIC glog::glog)
  target_link_libraries(knowhere_utils PUBLIC xxHash::xxhash)
endif()

# generate `knowhere_utils` library for RISCV64
if(__RISCV64)
  set(UTILS_SRC src/simd/hook.cc src/simd/distances_ref.cc src/simd/distances_rvv.cc)
  add_library(knowhere_utils STATIC ${UTILS_SRC})
  target_link_libraries(knowhere_utils PUBLIC glog::glog)
  target_link_libraries(knowhere_utils PUBLIC xxHash::xxhash)
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
  add_library(faiss_avx2 OBJECT ${FAISS_AVX2_SRCS} ${FAISS_FASTSCAN_SRCS})
  target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -msse4.2
                                            -mavx2 -mfma -mf16c -mpopcnt>)
  target_compile_definitions(faiss_avx2 PRIVATE COMPILE_SIMD_AVX2)
  target_include_directories(faiss_avx2 PRIVATE ${Boost_INCLUDE_DIRS})
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

  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})

  add_dependencies(faiss faiss_avx2 faiss_avx512 knowhere_utils)
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
  target_compile_definitions(faiss PRIVATE FINTEGER=int FAISS_ENABLE_DD COMPILE_SIMD_AVX2 COMPILE_SIMD_AVX512)

  if(WITH_SVS)
    # Use pre-built SVS runtime bindings (like baseline Faiss does).
    # The tarball ships libsvs_runtime.so with all deps (fmt, spdlog) baked in,
    # plus CMake config files and runtime API headers.
    # Override: set svs_runtime_DIR to a local unpacked tarball to skip download.
    find_package(svs_runtime 0.3.0 QUIET)
    if(NOT svs_runtime_FOUND)
      include(FetchContent)
      set(SVS_TARBALL_URL
          "https://github.com/intel/ScalableVectorSearch/releases/download/v0.3.0/svs-cpp-runtime-bindings-0.3.0.tar.gz"
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
