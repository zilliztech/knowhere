# Copyright (C) 2019-2024 Zilliz. All rights reserved.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(customIdx CXX C)

find_package(nlohmann_json REQUIRED)
find_package(glog REQUIRED)

list(APPEND CUSTOMIDX_LINKER_LIBS glog::glog)
list(APPEND CUSTOMIDX_LINKER_LIBS nlohmann_json::nlohmann_json)

add_library(customIdx SHARED customIdx/customIdx.cc)
target_link_libraries(customIdx PUBLIC ${CUSTOMIDX_LINKER_LIBS} "-Wl,--disable-new-dtags")
