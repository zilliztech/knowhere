# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_guard(GLOBAL)

function(knowhere_prepare_cuda)
  # This also runs when Knowhere is included with add_subdirectory, bypassing Make.
  execute_process(
    COMMAND bash "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../scripts/install_cuda.sh"
    RESULT_VARIABLE install_result)
  if(NOT install_result STREQUAL "0")
    message(FATAL_ERROR "Failed to prepare CUDA 13.2 (exit: ${install_result}).")
  endif()

  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.2/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.2/bin/nvcc" PARENT_SCOPE)
  set(CUDAToolkit_ROOT "/usr/local/cuda-13.2" CACHE PATH "CUDA Toolkit root" FORCE)
  set(CUDAToolkit_ROOT "/usr/local/cuda-13.2" PARENT_SCOPE)
endfunction()
