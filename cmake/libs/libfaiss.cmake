include(CheckCXXCompilerFlag)

include_directories(thirdparty/faiss)

# all .cpp files
knowhere_file_glob(
  GLOB
  FAISS_SRCS
  thirdparty/faiss/faiss/*.cpp
  thirdparty/faiss/faiss/impl/*.cpp
  thirdparty/faiss/faiss/impl/fast_scan/*.cpp
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


# AVX512 files
knowhere_file_glob(
  GLOB
  FAISS_AVX512_SRCS
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*avx512.cpp
)
# AVX512 vanilla Faiss dynamic dispatch related files
knowhere_file_glob(
  GLOB
  FAISS_DD_AVX512_SRCS
  thirdparty/faiss/faiss/impl/fast_scan/impl-avx512.cpp
  thirdparty/faiss/faiss/impl/pq_code_distance/pq_code_distance-avx512.cpp
  thirdparty/faiss/faiss/impl/scalar_quantizer/sq-avx512.cpp
  # # temporarily disabled
  # thirdparty/faiss/faiss/utils/distances_fused/avx512.cpp
  thirdparty/faiss/faiss/utils/simd_impl/distances_avx512.cpp
)
# combine files
list(APPEND FAISS_AVX512_SRCS ${FAISS_DD_AVX512_SRCS})
# remove platform files from general files
list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX512_SRCS})


# AVX2 files
knowhere_file_glob(
  GLOB
  FAISS_AVX2_SRCS
  thirdparty/faiss/faiss/impl/pq4_fast_scan_search_1.cpp
  thirdparty/faiss/faiss/impl/pq4_fast_scan_search_qbs.cpp
  thirdparty/faiss/faiss/IndexPQFastScan.cpp
  thirdparty/faiss/faiss/IndexIVFFastScan.cpp
  thirdparty/faiss/faiss/IndexIVFPQFastScan.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*avx.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/IndexIVFFastScan.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/IndexIVFPQFastScan.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/IVFFastScanIteratorWorkspace.cpp
)
# AVX2 vanilla Faiss dynamic dispatch related files
knowhere_file_glob(
  GLOB
  FAISS_DD_AVX2_SRCS
  thirdparty/faiss/faiss/impl/approx_topk/avx2.cpp
  thirdparty/faiss/faiss/impl/fast_scan/impl-avx2.cpp
  thirdparty/faiss/faiss/impl/pq_code_distance/pq_code_distance-avx2.cpp
  thirdparty/faiss/faiss/impl/scalar_quantizer/sq-avx2.cpp
  thirdparty/faiss/faiss/utils/simd_impl/distances_avx2.cpp
)
# combine files
list(APPEND FAISS_AVX2_SRCS ${FAISS_DD_AVX2_SRCS})
# remove platform files from general files
list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX2_SRCS})


# NEON files
knowhere_file_glob(
  GLOB
  FAISS_NEON_SRCS
  thirdparty/faiss/faiss/cppcontrib/knowhere/impl/*neon.cpp
)
# NEON vanilla Faiss dynamic dispatch related files
knowhere_file_glob(
  GLOB
  FAISS_DD_NEON_SRCS
  thirdparty/faiss/faiss/impl/approx_topk/neon.cpp
  thirdparty/faiss/faiss/impl/fast_scan/impl-neon.cpp
  thirdparty/faiss/faiss/impl/scalar_quantizer/sq-neon.cpp
  thirdparty/faiss/faiss/utils/simd_impl/distances_aarch64.cpp
)
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
# # RVE vanilla Faiss dynamic dispatch related files are not there yet
# knowhere_file_glob(
#   GLOB
#   FAISS_DD_RVV_SRCS
# )
# # combine files
# list(APPEND FAISS_RVV_SRCS ${FAISS_DD_RVV_SRCS})
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


if(LINUX)
  set(BLA_VENDOR OpenBLAS)
endif()

if(APPLE)
  set(BLA_VENDOR Apple)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  find_package(OpenBLAS REQUIRED)
  set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
else()
  find_package(LAPACK REQUIRED)
  find_package(BLAS REQUIRED)
endif()

find_package(xxHash REQUIRED)
include_directories(${xxHash_INCLUDE_DIRS})

# generate `faiss` library for x86
if(__X86_64)
  add_library(faiss_avx2 OBJECT ${FAISS_AVX2_SRCS})
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
endif()

# generate `faiss` library for AARCH64
if(__AARCH64)
  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})
  target_sources(faiss PRIVATE ${FAISS_NEON_SRCS})

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
  target_sources(faiss PRIVATE ${FAISS_RVV_SRCS})

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
  target_compile_definitions(faiss PRIVATE FINTEGER=int)
endif()

# generate `faiss` library for PPC64
if(__PPC64)
  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})

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
