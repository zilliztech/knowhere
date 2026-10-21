# Copyright (C) 2026 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

# Source libraries are linked statically or included as headers, so their
# documents must accompany Knowhere independently of the ELF dependency graph.
# Conan generate() writes all host dependency documents into licenses/conan.
function(knowhere_collect_binding_licenses source_root build_root with_diskann)
  set(license_root "${build_root}/licenses")
  set(source_licenses "${license_root}/source")
  file(REMOVE_RECURSE "${source_licenses}")
  set(documents LICENSE thirdparty/faiss/LICENSE thirdparty/faiss/THIRD_PARTY_NOTICES
      thirdparty/hnswlib/LICENSE)
  if(with_diskann)
    list(APPEND documents thirdparty/DiskANN/LICENSE thirdparty/DiskANN/NOTICE.txt)
  endif()
  foreach(document IN LISTS documents)
    if(NOT EXISTS "${source_root}/${document}")
      message(FATAL_ERROR "Required binding license document is missing: ${source_root}/${document}")
    endif()
    configure_file("${source_root}/${document}" "${source_licenses}/${document}" COPYONLY)
  endforeach()
  if(NOT EXISTS "${license_root}/conan/dependencies.json")
    message(WARNING "Conan host license inventory is missing; rerun conan install with C or JNI bindings enabled")
    file(WRITE "${source_licenses}/missing-licenses.txt"
      "Conan host dependency license inventory is missing; rerun conan install with C or JNI bindings enabled.\n")
  endif()
  # Both build-tree and installed libraries can discover an ancestor licenses
  # directory through the existing native bundler.
  install(DIRECTORY "${license_root}/" DESTINATION licenses)
endfunction()
