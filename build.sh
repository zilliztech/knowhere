#!/usr/bin/env bash
set -euo pipefail

mkdir -p build && cd build
#add conan remote
# conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
#DEBUG CPU
# conan install .. --build=missing -o with_ut=False WITH_BENCHMARK=TRUE  -s compiler.libcxx=libstdc++11 -s build_type=Release
#RELEASE CPU
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s build_type=Release
#DEBUG GPU
# conan install .. --build=missing -o with_ut=False WITH_BENCHMARK=TRUE -o with_cuvs=True -s compiler.libcxx=libstdc++11 -s build_type=Release
#RELEASE GPU
# conan install .. --build=missing -o with_ut=False WITH_BENCHMARK=TRUE -o with_cuvs=True -s compiler.libcxx=libstdc++11 -s build_type=Release
#DISKANN SUPPORT
# conan install .. --build=missing -o with_ut=False WITH_BENCHMARK=TRUE -o with_diskann=True -s compiler.libcxx=libstdc++11 -s build_type=Release
#build with conan
conan build ..
#verbose
export VERBOSE=1
