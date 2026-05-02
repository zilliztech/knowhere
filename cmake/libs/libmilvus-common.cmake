
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")
set( MILVUS-COMMON-VERSION 28a2b48f5ab5e14280cfd5d6bf9b34a9cb9a6fd7 )
set( GIT_REPOSITORY  "https://github.com/zilliztech/milvus-common.git" )

message(STATUS "milvus-common repo: ${GIT_REPOSITORY}")
message(STATUS "milvus-common version: ${MILVUS-COMMON-VERSION}")
message(STATUS "Building milvus-common-${MILVUS-COMMON-VERSION} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

list(APPEND CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT} )

include( FetchContent )

find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(folly REQUIRED)
if(APPLE)
    set(BLA_VENDOR Apple)
    find_package(LAPACK REQUIRED)
    find_package(BLAS REQUIRED)
    set(MILVUS_COMMON_BLAS_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    find_package(OpenBLAS CONFIG REQUIRED)
    set(MILVUS_COMMON_BLAS_LIBS OpenBLAS::OpenBLAS)
endif()
find_package(prometheus-cpp REQUIRED)

FetchContent_Declare(
        milvus-common
        GIT_REPOSITORY  ${GIT_REPOSITORY}
        GIT_TAG         ${MILVUS-COMMON-VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-build
        DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH} )

FetchContent_GetProperties( milvus-common )
if ( NOT milvus-common_POPULATED )
    FetchContent_Populate( milvus-common )
    add_subdirectory( ${milvus-common_SOURCE_DIR}
                      ${milvus-common_BINARY_DIR} )

    # milvus-common's CMakeLists hardcodes CMAKE_CXX_STANDARD=17, but folly v2026
    # headers require C++20 (consteval/constinit). Override on the target.
    set_target_properties(milvus-common PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)

    if(NOT APPLE)
        target_link_libraries(milvus-common PUBLIC atomic)
    endif()
endif()

target_link_libraries(milvus-common PUBLIC ${MILVUS_COMMON_BLAS_LIBS})

set( MILVUS_COMMON_INCLUDE_DIR ${milvus-common_SOURCE_DIR}/include
     CACHE INTERNAL "Path to milvus-common include directory" )

include_directories(${MILVUS_COMMON_INCLUDE_DIR})
