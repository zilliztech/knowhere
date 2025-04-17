include(CheckSymbolExists)

macro(detect_target_arch)
  check_symbol_exists(__aarch64__ "" __AARCH64)
  check_symbol_exists(__x86_64__ "" __X86_64)
  check_symbol_exists(__powerpc64__ "" __PPC64)
  check_symbol_exists(__riscv "" __RISCV64)

  if(NOT __AARCH64
     AND NOT __X86_64
     AND NOT __PPC64
     AND NOT __RISCV64)
    message(FATAL "knowhere only support amd64, ppc64, riscv64 and arm64 architecture.")
  endif()
endmacro()


if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(__AARCH64 1)
else()
    detect_target_arch()
endif()
