cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(knowhere CXX C)

# Use short SHA1 as version
# currenly checkout a commit for cardinal v2. TODO: need a tag here
set(CARDINAL_VERSION ce2908ae8375f537f95c88d04dad7936c7123d8d)
set(CARDINAL_REPO_URL "https://github.com/zilliztech/cardinal.git")

set(CARDINAL_ROOT  "${KNOWHERE_THRID_ROOT}/cardinalv2")
message(STATUS "Build Cardinal-${CARDINAL_VERSION}")

# Clone and checkout cardinal with given repo url and version
if (NOT EXISTS "${CARDINAL_ROOT}/.git")
    execute_process(COMMAND git clone ${CARDINAL_REPO_URL} ${CARDINAL_ROOT}
            RESULT_VARIABLE CARDINAL_CLONE_RESULT
            OUTPUT_VARIABLE CARDINAL_CLONE_OUTPUT
            ERROR_VARIABLE CARDINAL_CLONE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
    if (NOT CARDINAL_CLONE_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to clone cardinal: ${CARDINAL_CLONE_ERROR}")
    else()
      message(STATUS "Successfully Clone Cardinal Repo")
      execute_process(COMMAND git -C ${CARDINAL_ROOT} checkout ${CARDINAL_VERSION}
            RESULT_VARIABLE CARDINAL_CHECKOUT_RESULT
            OUTPUT_VARIABLE CARDINAL_CHECKOUT_OUTPUT
            ERROR_VARIABLE CARDINAL_CHECKOUT_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
        if (NOT CARDINAL_CHECKOUT_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to checkout cardinal: ${CARDINAL_CHECKOUT_ERROR}")
        else()
            message(STATUS "Successfully checkout Cardinal Version : ${CARDINAL_VERSION}")
        endif()
    endif()
else()
    execute_process(
          COMMAND git -C ${CARDINAL_ROOT} rev-parse HEAD
          OUTPUT_VARIABLE GIT_COMMIT_HASH
          OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Cardinal repo already exist! git commit :  ${GIT_COMMIT_HASH}")
endif()

# Force checkout the version specified as `CARDINAL_VERSION` if `CARDINAL_VERSION_FORCE_CHECKOUT` is set
# Default do not checkout for better development convenience
if(CARDINAL_VERSION_FORCE_CHECKOUT)
    message(STATUS "Checking out cardinal version ${CARDINAL_VERSION}")

    execute_process(
            COMMAND git -C ${CARDINAL_ROOT} fetch
            RESULT_VARIABLE CARDINAL_FETCH_RESULT
            OUTPUT_VARIABLE CARDINAL_FETCH_OUTPUT
            ERROR_VARIABLE CARDINAL_FETCH_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

    if(NOT CARDINAL_FETCH_RESULT EQUAL "0")
        message(
                FATAL_ERROR "Failed to fetch cardinal: ${CARDINAL_FETCH_ERROR}")
    endif()

    message(STATUS "Fetched cardinal ${CARDINAL_FETCH_OUTPUT}")

    execute_process(
            COMMAND git -C ${CARDINAL_ROOT} checkout ${CARDINAL_VERSION}
            RESULT_VARIABLE CARDINAL_CHECKOUT_RESULT
            OUTPUT_VARIABLE CARDINAL_CHECKOUT_OUTPUT
            ERROR_VARIABLE CARDINAL_CHECKOUT_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

    if(NOT CARDINAL_CHECKOUT_RESULT EQUAL "0")
        message(
                FATAL_ERROR "Failed to checkout cardinal: ${CARDINAL_CHECKOUT_ERROR}")
    endif()
endif()

# Compile Cardinal sparse integration alongside ordinary sparse, sharing Knowhere's
# codecs and searchers. Older Cardinal revisions may have no sparse files.
knowhere_file_glob(GLOB_RECURSE CARDINAL_SPARSE_SRCS ${CARDINAL_ROOT}/know/sparse/src/*.cc)
list(APPEND KNOWHERE_SRCS ${CARDINAL_SPARSE_SRCS})
set(KNOWHERE_SRCS ${KNOWHERE_SRCS} PARENT_SCOPE)
if(WITH_UT)
    knowhere_file_glob(GLOB_RECURSE CARDINAL_SPARSE_TESTS ${CARDINAL_ROOT}/know/sparse/tests/*.cc)
    list(APPEND KNOWHERE_UT_SRCS ${CARDINAL_SPARSE_TESTS})
    set(KNOWHERE_UT_SRCS ${KNOWHERE_UT_SRCS} PARENT_SCOPE)
endif()

include(${CARDINAL_ROOT}/know/libcardinal.cmake)
