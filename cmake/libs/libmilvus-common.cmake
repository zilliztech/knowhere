
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")
set( MILVUS-COMMON-VERSION submit_v2 )
set( GIT_REPOSITORY  "https://github.com/foxspy/milvus-common.git")

message(STATUS "milvus-common repo: ${GIT_REPOSITORY}")
message(STATUS "milvus-common version: ${MILVUS-COMMON-VERSION}")
message(STATUS "Building milvus-common-${MILVUS-COMMON-VERSION} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

set( CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT} )

include( FetchContent )

FetchContent_Declare(
        milvus-common
        GIT_REPOSITORY  ${GIT_REPOSITORY}
        GIT_TAG         ${MILVUS-COMMON-VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-build
        SOURCE_SUBDIR   cpp
        DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH} )

FetchContent_GetProperties( milvus-common )
if ( NOT milvus-common_POPULATED )
    FetchContent_Populate( milvus-common )
    # Adding the following target:
    # milvus-common
    add_subdirectory( ${milvus-common_SOURCE_DIR}
                      ${milvus-common_BINARY_DIR} )
endif()

set( MILVUS_COMMON_INCLUDE_DIR ${milvus-common_SOURCE_DIR}/include CACHE INTERNAL "Path to milvus-common include directory" )

include_directories(${MILVUS_COMMON_INCLUDE_DIR})
