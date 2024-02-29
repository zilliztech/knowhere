include(CheckSymbolExists)

macro(detect_target_arch)
  check_symbol_exists(__aarch64__ "" __AARCH64)
  check_symbol_exists(__x86_64__ "" __X86_64)
  check_symbol_exists(__powerpc64__ "" __PPC64)

  if(NOT __AARCH64 AND NOT __X86_64 AND NOT __PPC64)
    message(FATAL "knowhere only support amd64, ppc64 and arm64 architecture.")
  endif()
endmacro()

detect_target_arch()
