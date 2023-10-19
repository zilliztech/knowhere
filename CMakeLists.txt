# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License

cmake_minimum_required(VERSION 3.23.0 FATAL_ERROR)
project(knowhere CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
include(GNUInstallDirs)
include(ExternalProject)
include(cmake/utils/utils.cmake)

knowhere_option(WITH_UT "Build with UT test" OFF)
knowhere_option(WITH_ASAN "Build with ASAN" OFF)
knowhere_option(WITH_DISKANN "Build with diskann index" OFF)
knowhere_option(WITH_RAFT "Build with RAFT indexes" OFF)
knowhere_option(WITH_BENCHMARK "Build with benchmark" OFF)
knowhere_option(WITH_COVERAGE "Build with coverage" OFF)
knowhere_option(WITH_CCACHE "Build with ccache" ON)
knowhere_option(WITH_PROFILER "Build with profiler" OFF)

if(KNOWHERE_VERSION)
  message(STATUS "Building KNOWHERE version: ${KNOWHERE_VERSION}")
  add_definitions(-DKNOWHERE_VERSION=${KNOWHERE_VERSION})
endif()

if(WITH_CCACHE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    message(STATUS "Using ccache: ${CCACHE_FOUND}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_FOUND})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_FOUND})
    # let ccache preserve C++ comments, because some of them may be meaningful
    # to the compiler
    set(ENV{CCACHE_COMMENTS} "1")
  endif()
endif()

list( APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/)

if(WITH_RAFT)
  if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(${CUDAToolkit_VERSION_MAJOR} GREATER 10)
    # cuda11 support --threads for compile some large .cu more efficient
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--threads=4>)
  endif()
endif()

add_definitions(-DNOT_COMPILE_FOR_SWIG)

include(cmake/utils/compile_flags.cmake)
include(cmake/utils/platform_check.cmake)
include(cmake/libs/libfaiss.cmake)
include(cmake/libs/libhnsw.cmake)

include_directories(thirdparty/faiss)

find_package(OpenMP REQUIRED)

find_package(folly REQUIRED)
set(FOLLY_LIBRARIES Folly::folly)
include_directories(${folly_INCLUDE_DIRS})

if(WITH_RAFT)
  include(cmake/libs/libraft.cmake)
endif()

find_package(nlohmann_json REQUIRED)
find_package(glog REQUIRED)
find_package(prometheus-cpp REQUIRED)
find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIR})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version" FORCE)

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(WITH_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

knowhere_file_glob(GLOB_RECURSE KNOWHERE_SRCS src/common/*.cc src/index/*.cc
                   src/io/*.cc src/index/*.cu src/common/raft/*.cu
                   src/common/raft/*.cc)

set(KNOWHERE_LINKER_LIBS "")

if(WITH_DISKANN)
  add_definitions(-DKNOWHERE_WITH_DISKANN)
  include(cmake/libs/libdiskann.cmake)
else()
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_DISKANN_SRCS src/index/diskann/*.cc)
  list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_DISKANN_SRCS})
endif()

knowhere_file_glob(GLOB_RECURSE KNOWHERE_GPU_SRCS src/index/gpu/flat_gpu/*.cc
                     src/index/gpu/ivf_gpu/*.cc)
list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_GPU_SRCS})

if(NOT WITH_RAFT)
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_RAFT_SRCS src/index/ivf_raft/*.cc
                     src/index/ivf_raft/*.cu src/index/cagra/*.cu
                     src/common/raft/*.cu src/common/raft/*.cc)
  list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_RAFT_SRCS})
endif()

include_directories(src)
include_directories(include)

list(APPEND KNOWHERE_LINKER_LIBS faiss)
list(APPEND KNOWHERE_LINKER_LIBS glog::glog)
list(APPEND KNOWHERE_LINKER_LIBS nlohmann_json::nlohmann_json)
list(APPEND KNOWHERE_LINKER_LIBS prometheus-cpp::core prometheus-cpp::push)
list(APPEND KNOWHERE_LINKER_LIBS ${FOLLY_LIBRARIES})

add_library(knowhere SHARED ${KNOWHERE_SRCS})
add_dependencies(knowhere ${KNOWHERE_LINKER_LIBS})
if(WITH_RAFT)
    list(APPEND KNOWHERE_LINKER_LIBS raft::raft CUDA::cublas CUDA::cusolver)
endif()
target_link_libraries(knowhere PUBLIC ${KNOWHERE_LINKER_LIBS})
target_include_directories(
  knowhere PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(WITH_UT)
  add_subdirectory(tests/ut)
endif()

if(WITH_BENCHMARK)
  add_subdirectory(benchmark)
endif()

install(TARGETS knowhere
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/knowhere"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
