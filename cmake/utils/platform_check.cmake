include(CheckSymbolExists)

macro(detect_target_arch)
  check_symbol_exists(__aarch64__ "" __AARCH64)
  check_symbol_exists(__x86_64__ "" __X86_64)
  check_symbol_exists(__powerpc64__ "" __PPC64)
  check_symbol_exists(__riscv "" __RISCV64)
  check_symbol_exists(__loongarch64 "" __LOONGARCH64)

  if(NOT __AARCH64
     AND NOT __X86_64
     AND NOT __PPC64
     AND NOT __RISCV64
     AND NOT __LOONGARCH64)
    message(FATAL "knowhere only supports amd64, arm64, loongarch64, ppc64 and riscv64 architectures.")
  endif()
endmacro()


if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(__AARCH64 1)
else()
    detect_target_arch()
endif()
