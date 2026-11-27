set( MILVUS-COMMON-VERSION 60f998e4494ad1630ef5ce58a06cc26565ed5bc1 )
set( GIT_REPOSITORY "https://github.com/zilliztech/milvus-common.git" )

message(STATUS "milvus-common repo: ${GIT_REPOSITORY}")
message(STATUS "milvus-common version: ${MILVUS-COMMON-VERSION}")
message(STATUS "Building milvus-common-${MILVUS-COMMON-VERSION} from source")

include( FetchContent )

FetchContent_Declare(
        milvus-common
        GIT_REPOSITORY  ${GIT_REPOSITORY}
        GIT_TAG         ${MILVUS-COMMON-VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-build )

FetchContent_GetProperties( milvus-common )
FetchContent_MakeAvailable( milvus-common )
# Place the shared library next to libknowhere.so in the top-level build
# directory; FetchContent targets would otherwise land inside their own
# subdirectory, breaking the wheel's -L/-rpath (python/setup.py).
set_target_properties(milvus-common PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )
# libatomic is a separate library only on Linux (needed for 64-bit atomics on
# some architectures with GCC). On macOS, atomic operations are part of the
# standard library.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(milvus-common PUBLIC atomic)
endif()

set( MILVUS_COMMON_INCLUDE_DIR ${milvus-common_SOURCE_DIR}/include CACHE INTERNAL "Path to milvus-common include directory" )

# Compatibility shim: vendored cardinal adapters (e.g. cardinal v3.0.12's
# know/libcardinal.cmake under WITH_CARDINAL=ON) still call
# find_package(milvus-common) and link milvus-common::milvus-common, which the
# Conan CMakeDeps config used to provide. Route those lookups to the
# source-built target instead.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/milvus-common-compat")
configure_file("${CMAKE_CURRENT_LIST_DIR}/milvus-commonConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/milvus-common-compat/milvus-commonConfig.cmake" @ONLY)
set(milvus-common_DIR "${CMAKE_BINARY_DIR}/milvus-common-compat" CACHE PATH
    "Compat config for the source-built milvus-common" FORCE)

include_directories(SYSTEM ${MILVUS_COMMON_INCLUDE_DIR})
