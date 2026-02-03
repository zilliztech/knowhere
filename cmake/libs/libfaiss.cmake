include(CheckCXXCompilerFlag)

include_directories(thirdparty/faiss)

knowhere_file_glob(
  GLOB FAISS_SRCS thirdparty/faiss/faiss/*.cpp
  thirdparty/faiss/faiss/impl/*.cpp thirdparty/faiss/faiss/invlists/*.cpp
  thirdparty/faiss/faiss/utils/*.cpp
  thirdparty/faiss/faiss/cppcontrib/knowhere/*.cpp)

knowhere_file_glob(GLOB FAISS_AVX512_SRCS
                   thirdparty/faiss/faiss/impl/*avx512.cpp)

knowhere_file_glob(
  GLOB
  FAISS_AVX2_SRCS
  thirdparty/faiss/faiss/impl/*avx.cpp
  thirdparty/faiss/faiss/impl/pq4_fast_scan_search_1.cpp
  thirdparty/faiss/faiss/impl/pq4_fast_scan_search_qbs.cpp
  thirdparty/faiss/faiss/utils/partitioning_avx2.cpp
  thirdparty/faiss/faiss/IndexPQFastScan.cpp
  thirdparty/faiss/faiss/IndexIVFFastScan.cpp
  thirdparty/faiss/faiss/IndexIVFPQFastScan.cpp)

list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX512_SRCS})

# disable RHNSW
knowhere_file_glob(GLOB FAISS_RHNSW_SRCS thirdparty/faiss/faiss/impl/RHNSW.cpp)
list(REMOVE_ITEM FAISS_SRCS ${FAISS_RHNSW_SRCS})

if(__X86_64)
  set(UTILS_SRC src/simd/distances_ref.cc src/simd/hook.cc)
  set(UTILS_SSE_SRC src/simd/distances_sse.cc)
  set(UTILS_AVX_SRC src/simd/distances_avx.cc)
  set(UTILS_AVX512_SRC src/simd/distances_avx512.cc)
  set(UTILS_AVX512ICX_SRC src/simd/distances_avx512icx.cc)
  set(SPARSE_SIMD_AVX512_SRC src/simd/sparse_simd_avx512.cc)
  set(SPARSE_SEEK_AVX512_SRC src/simd/sparse_seek_avx512.cc)

  add_library(utils_sse OBJECT ${UTILS_SSE_SRC})
  add_library(utils_avx OBJECT ${UTILS_AVX_SRC})
  add_library(utils_avx512 OBJECT ${UTILS_AVX512_SRC})
  add_library(utils_avx512icx OBJECT ${UTILS_AVX512ICX_SRC})
  add_library(sparse_simd_avx512 OBJECT ${SPARSE_SIMD_AVX512_SRC})
  add_library(sparse_seek_avx512 OBJECT ${SPARSE_SEEK_AVX512_SRC})

  target_compile_options(utils_sse PRIVATE -msse4.2 -mpopcnt)
  target_compile_options(utils_avx PRIVATE -mfma -mf16c -mavx2 -mpopcnt)
  target_compile_options(utils_avx512 PRIVATE -mfma -mf16c -mavx512f -mavx512dq
                                              -mavx512bw -mpopcnt -mavx512vl)
  target_compile_options(utils_avx512icx PRIVATE -mfma -mf16c -mavx512f -mavx512dq
                                              -mavx512bw -mpopcnt -mavx512vl -mavx512vpopcntdq)
  target_compile_options(sparse_simd_avx512 PRIVATE -mavx512f -mavx512dq)
  target_compile_options(sparse_seek_avx512 PRIVATE -mavx512f -mavx512dq)
  target_include_directories(sparse_simd_avx512 PRIVATE ${Boost_INCLUDE_DIRS})

  add_library(
    knowhere_utils STATIC
    ${UTILS_SRC} $<TARGET_OBJECTS:utils_sse> $<TARGET_OBJECTS:utils_avx>
    $<TARGET_OBJECTS:utils_avx512> $<TARGET_OBJECTS:utils_avx512icx>
    $<TARGET_OBJECTS:sparse_simd_avx512> $<TARGET_OBJECTS:sparse_seek_avx512>)
  target_link_libraries(knowhere_utils PUBLIC glog::glog)
  target_link_libraries(knowhere_utils PUBLIC xxHash::xxhash)
endif()

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

if(__X86_64)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX2_SRCS})

  knowhere_file_glob(GLOB FAISS_NEON_SRCS thirdparty/faiss/faiss/impl/*neon.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_NEON_SRCS})

  knowhere_file_glob(GLOB FAISS_RVV_SRCS thirdparty/faiss/faiss/impl/*rvv.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_RVV_SRCS})

  add_library(faiss_avx2 OBJECT ${FAISS_AVX2_SRCS})
  target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -msse4.2
                                            -mavx2 -mfma -mf16c -mpopcnt>)
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
  target_include_directories(faiss_avx512 PRIVATE ${Boost_INCLUDE_DIRS})

  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})

  add_dependencies(faiss faiss_avx2 faiss_avx512 knowhere_utils)
  target_compile_options(
    faiss
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -msse4.2
            -mpopcnt
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)
  target_link_libraries(
    faiss PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                 faiss_avx2 faiss_avx512 knowhere_utils)
  target_compile_definitions(faiss PRIVATE FINTEGER=int)
endif()

if(__AARCH64)
  knowhere_file_glob(GLOB FAISS_AVX_SRCS thirdparty/faiss/faiss/impl/*avx.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX_SRCS})

  knowhere_file_glob(GLOB FAISS_RVV_SRCS thirdparty/faiss/faiss/impl/*rvv.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_RVV_SRCS})

  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})

  target_compile_options(
    faiss
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
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

if(__RISCV64)
  knowhere_file_glob(GLOB FAISS_AVX_SRCS thirdparty/faiss/faiss/impl/*avx.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX_SRCS})

  knowhere_file_glob(GLOB FAISS_NEON_SRCS thirdparty/faiss/faiss/impl/*neon.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_NEON_SRCS})

  add_library(faiss STATIC ${FAISS_SRCS})
  target_include_directories(faiss PRIVATE ${Boost_INCLUDE_DIRS})

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

if(__PPC64)
  knowhere_file_glob(GLOB FAISS_AVX_SRCS thirdparty/faiss/faiss/impl/*avx.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX_SRCS})

  knowhere_file_glob(GLOB FAISS_NEON_SRCS thirdparty/faiss/faiss/impl/*neon.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_NEON_SRCS})

  knowhere_file_glob(GLOB FAISS_RVV_SRCS thirdparty/faiss/faiss/impl/*rvv.cpp)
  list(REMOVE_ITEM FAISS_SRCS ${FAISS_RVV_SRCS})

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
