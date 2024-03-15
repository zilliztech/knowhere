set(CARDINAL_VERSION master)
set(CARDINAL_REPO_URL "https://github.com/zilliztech/cardinal.git")

set(CARDINAL_REPO_DIR "${CMAKE_SOURCE_DIR}/thirdparty/cardinal")

message(STATUS "Build Cardinal-${CARDINAL_VERSION}")

if (NOT EXISTS "${CARDINAL_REPO_DIR}/.git")
    execute_process(COMMAND git clone ${CARDINAL_REPO_URL} ${CARDINAL_REPO_DIR}
            RESULT_VARIABLE CARDINAL_CLONE_RESULT
            OUTPUT_VARIABLE CARDINAL_CLONE_OUTPUT
            ERROR_VARIABLE CARDINAL_CLONE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
    if (NOT CARDINAL_CLONE_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to clone cardinal: ${CARDINAL_CLONE_ERROR}")
    else()
        execute_process(COMMAND git -C ${CARDINAL_REPO_DIR} checkout ${CARDINAL_VERSION}
            RESULT_VARIABLE CARDINAL_CHECKOUT_RESULT
            OUTPUT_VARIABLE CARDINAL_CHECKOUT_OUTPUT
            ERROR_VARIABLE CARDINAL_CHECKOUT_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
        if (NOT CARDINAL_CHECKOUT_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to checkout cardinal: ${CARDINAL_CHECKOUT_ERROR}")
        endif()
    endif()
endif()

include(${CARDINAL_REPO_DIR}/know/libcardinal.cmake)
