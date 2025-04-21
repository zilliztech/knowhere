# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License

if(NOT DEFINED KNOWHERE_UT_SRCS)
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_UT_SRCS *.cc)
else()
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_ORIGINAL_UT_SRCS *.cc)
  list(APPEND KNOWHERE_UT_SRCS ${KNOWHERE_ORIGINAL_UT_SRCS})
endif()

if(NOT WITH_DISKANN)
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_DISKANN_TESTS test_diskann.cc)
  list(REMOVE_ITEM KNOWHERE_UT_SRCS ${KNOWHERE_DISKANN_TESTS})
endif()

if (WITH_CARDINAL)
  knowhere_file_glob(GLOB_RECURSE CARDINAL_UNSUPPORTED_TESTS test_feder.cc)
  list(REMOVE_ITEM KNOWHERE_UT_SRCS ${CARDINAL_UNSUPPORTED_TESTS})
else()
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_CLUSTER_TESTS test_cluster.cc)
  list(REMOVE_ITEM KNOWHERE_UT_SRCS ${KNOWHERE_CLUSTER_TESTS})
endif()

add_executable(knowhere_tests ${KNOWHERE_UT_SRCS})

find_package(Catch2)
if (NOT Catch2_FOUND)
  message("Catch2 not found, fetching Catch2 from github")
  Include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.1.1 # or a later release
  )
  FetchContent_MakeAvailable(Catch2)
endif()

set_target_properties(knowhere_tests PROPERTIES
  CXX_STANDARD 17
)

target_link_libraries(knowhere_tests PRIVATE
        Catch2::Catch2WithMain
        atomic
        stdc++fs
        knowhere)
