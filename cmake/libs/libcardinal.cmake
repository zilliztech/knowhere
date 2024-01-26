# Update KNOWHERE_VERSION for the first occurrence
set( CARDINAL_VERSION master )
set( CARDINAL_GIT_REPOSITORY  "https://github.com/zilliztech/cardinal.git")

message(STATUS "Cardinal repo: ${CARDINAL_VERSION}")
message(STATUS "Cardinal version: ${CARDINAL_GIT_REPOSITORY}")

message(STATUS "Building cardinal-${KNOWHERE_SOURCE_VER} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

set( CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/generators )

set( BUILD_CARDINAL_AS_PLUGIN ON CACHE BOOL "" FORCE )

message(STATUS "Build Cardinal with plugin option : ${BUILD_CARDINAL_AS_PLUGIN}")

include( FetchContent )
FetchContent_Declare(
        cardinal
        GIT_REPOSITORY  ${CARDINAL_GIT_REPOSITORY}
        GIT_TAG         ${CARDINAL_VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/cardinal-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/cardinal-build
        DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH} )

FetchContent_MakeAvailable(
        cardinal
)
