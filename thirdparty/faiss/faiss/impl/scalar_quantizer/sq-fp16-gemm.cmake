# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)
include(CheckCXXCompilerFlag)

# Shared by both build entry points. Paths are absolute; source properties are
# set in the calling directory.
# Dispatch and scalar code are built without ISA flags; only the kernel files
# use them.
function(faiss_configure_fp16_gemm out_sources faiss_source_dir)
  get_filename_component(fp16_source_dir "${faiss_source_dir}" ABSOLUTE)
  set(fp16_kernel_dir "${fp16_source_dir}/impl/scalar_quantizer")
  set(fp16_kernel_sources "")
  set(fp16_fma_compiled 0)
  set(fp16_amx_compiled 0)

  if(NOT WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    check_cxx_compiler_flag("-mavx512f -mavx512bw -mf16c -mfma"
      FAISS_COMPILER_SUPPORTS_FP16_GEMM_FMA)
    if(FAISS_COMPILER_SUPPORTS_FP16_GEMM_FMA)
      list(APPEND fp16_kernel_sources "${fp16_kernel_dir}/sq-fp16-gemm-fma.cpp")
      set_property(SOURCE "${fp16_kernel_dir}/sq-fp16-gemm-fma.cpp" APPEND
        PROPERTY COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mf16c;-mfma")
      set(fp16_fma_compiled 1)
    else()
      message(STATUS "FP16 GEMM: compiler lacks the flags for the FMA kernel; scalar backend only.")
    endif()

    check_cxx_compiler_flag("-mamx-tile -mamx-fp16"
      FAISS_COMPILER_SUPPORTS_FP16_GEMM_AMX)
    if(FAISS_COMPILER_SUPPORTS_FP16_GEMM_AMX)
      list(APPEND fp16_kernel_sources "${fp16_kernel_dir}/sq-fp16-gemm-amx.cpp")
      set_property(SOURCE "${fp16_kernel_dir}/sq-fp16-gemm-amx.cpp" APPEND
        PROPERTY COMPILE_OPTIONS "-mamx-tile;-mamx-fp16")
      set(fp16_amx_compiled 1)
    else()
      message(STATUS "FP16 GEMM: compiler lacks the flags for the AMX-FP16 kernel.")
    endif()
  else()
    message(STATUS "FP16 GEMM: only the scalar backend is available on this platform.")
  endif()

  # Define macros only for kernels actually built, so non-x86 or older compilers
  # never reference missing symbols.
  set_property(SOURCE "${fp16_kernel_dir}/sq-fp16-gemm-common.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS
      "FAISS_FP16_GEMM_FMA_COMPILED=${fp16_fma_compiled}"
      "FAISS_AMX_FP16_COMPILED=${fp16_amx_compiled}")
  set(${out_sources} "${fp16_kernel_sources}" PARENT_SCOPE)
endfunction()
