
# When MILVUS_COMMON_DIR is set, use a pre-built local milvus-common.
# Otherwise fall back to FetchContent from GitHub (upstream default).

if(MILVUS_COMMON_DIR)
    message(STATUS "Using pre-built milvus-common from: ${MILVUS_COMMON_DIR}")

    find_library(MILVUS_COMMON_LIBRARY
        NAMES milvus-common
        PATHS "${MILVUS_COMMON_DIR}/build"
        NO_DEFAULT_PATH
    )

    if(NOT MILVUS_COMMON_LIBRARY)
        message(FATAL_ERROR
            "milvus-common library not found in ${MILVUS_COMMON_DIR}/build. "
            "Please build milvus-common first: cd ${MILVUS_COMMON_DIR} && make"
        )
    endif()

    message(STATUS "Found milvus-common library: ${MILVUS_COMMON_LIBRARY}")

    add_library(milvus-common SHARED IMPORTED GLOBAL)
    set_target_properties(milvus-common PROPERTIES
        IMPORTED_LOCATION "${MILVUS_COMMON_LIBRARY}"
    )

    set(MILVUS_COMMON_INCLUDE_DIR "${MILVUS_COMMON_DIR}/include"
        CACHE INTERNAL "Path to milvus-common include directory")

    if(NOT EXISTS "${MILVUS_COMMON_INCLUDE_DIR}")
        message(FATAL_ERROR "milvus-common include directory not found: ${MILVUS_COMMON_INCLUDE_DIR}")
    endif()

    set_target_properties(milvus-common PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MILVUS_COMMON_INCLUDE_DIR}"
    )

else()
    # Upstream default: fetch from git and build as subdirectory
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")
    set( MILVUS-COMMON-VERSION dba70b6 )
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

        if(NOT APPLE)
            target_link_libraries(milvus-common PUBLIC atomic)
        endif()
    endif()

    set( MILVUS_COMMON_INCLUDE_DIR ${milvus-common_SOURCE_DIR}/include
         CACHE INTERNAL "Path to milvus-common include directory" )

endif()

include_directories(${MILVUS_COMMON_INCLUDE_DIR})
