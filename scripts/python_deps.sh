#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit immediately for non zero status
set -e

UNAME="$(uname -s)"

case "${UNAME}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${UNAME}"
esac


if [[ "${MACHINE}" == "Linux" ]]; then
    if [[ -x "$(command -v apt)" ]]; then
        apt install -y sudo
        # for Ubuntu 18.04
        release_num=$(lsb_release -r --short)
        sudo apt install -y libcurl4-openssl-dev libaio libaio-devel libopenblas-openmp-dev
        pip3 install conan==1.61.0
    elif [[ -x "$(command -v yum)" ]]; then
        yum install -y sudo
        case $(uname -m) in
            i386) sudo yum install -y blas-devel libaio libaio-devel ;;
            i686) sudo yum install -y blas-devel libaio libaio-devel ;;
            x86_64) sudo yum install -y openblas-devel libaio libaio-devel ;;
            arm) ;;
        esac
        pip3 install conan==1.61.0
    elif [[ -x "$(command -v apk)" ]]; then
        apk add openblas-dev libaio libaio-dev
        pip3 install conan==1.61.0
    fi
fi

if [[ "${MACHINE}" == "Mac"  ]]; then
    brew install llvm@16
    brew install libomp openblas

    echo "Setting CC and CXX to use llvm-clang"
    export PATH="/usr/local/opt/llvm/bin:$PATH"
    export LDFLAGS="-L/usr/local/opt/llvm/lib -L/usr/local/opt/llvm/lib/c++ -Wl,-rpath,/usr/local/opt/llvm/lib/c++ $LDFLAGS"
    export CPPFLAGS="-I/usr/local/opt/llvm/include $CPPFLAGS"
    export CMAKE_PREFIX_PATH="/usr/local;${CMAKE_PREFIX_PATH}"
    export CC="$(brew --prefix llvm)/bin/clang"
    export CXX="$(brew --prefix llvm)/bin/clang++"
    pip3 install conan==1.61.0
    conan profile new default --detect --force
    conan profile update settings.compiler=clang default
    conan profile update settings.compiler.version=16 default
    conan profile update settings.compiler.cppstd=17 default
    conan profile update settings.compiler.libcxx=libc++ default
fi
