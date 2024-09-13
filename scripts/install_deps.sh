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
        # for Ubuntu 18.04
        release_num=$(lsb_release -r --short)
        ubuntu_alias="bionic"
        sudo apt install -y g++ gcc make ccache python3 python3-pip gfortran libcurl4-openssl-dev libaio-dev \
             libdouble-conversion-dev libevent-dev libgflags-dev lcov
        if [ "$release_num" == "20.04" ];then
            sudo apt install -y python3-setuptools swig
            ubuntu_alias="focal"
        fi
        #DiskANN dependencies
        sudo apt-get install -y libboost-program-options-dev
        sudo apt-get install -y libaio-dev
        pip3 install conan==1.61.0
        conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
        # Pre-installation of openblas can save about 15 minutes of openblas building time.
        # But the apt-installed openblas version is 0.2.20, while the latest openblas version is 0.3.19.
        # So we only pre-install openblas in Unittest, and compile openblas-0.3.19 when release.
        if [[ "${INSTALL_OPENBLAS}" == "true" ]]
          then
            sudo apt install -y libopenblas-openmp-dev
          else
            wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee \
            /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
            && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ '$ubuntu_alias' main' | sudo tee \
            /etc/apt/sources.list.d/kitware.list >/dev/null \
            && sudo apt update && sudo apt install -y cmake \
            && wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.10.1.tar.gz \
            && tar zxvf v3.10.1.tar.gz && cd lapack-3.10.1/  \
            && cmake -B build -S . -DCMAKE_SKIP_RPATH=ON -DBUILD_SHARED_LIBS=ON \
            -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DCMAKE_Fortran_COMPILER=gfortran -DLAPACKE_WITH_TMG=ON -DCBLAS=OFF \
            -DBUILD_DEPRECATED=OFF \
            && cmake --build build \
            && sudo cmake --install build

            wget https://github.com/xianyi/OpenBLAS/archive/refs/tags/v0.3.21.tar.gz && \
            tar zxvf v0.3.21.tar.gz && cd OpenBLAS-0.3.21 && \
            make NO_STATIC=1 NO_LAPACK=1 NO_LAPACKE=1 NO_AFFINITY=1 USE_OPENMP=1 \
            TARGET=HASWELL DYNAMIC_ARCH=1 \
            NUM_THREADS=64 MAJOR_VERSION=3 libs shared && \
            sudo make PREFIX=/usr/local NUM_THREADS=64 MAJOR_VERSION=3 install && \
            sudo ln -s /usr/local/lib/libopenblasp-r0.3.21.so /usr/local/lib/libblas.so && \
            sudo ln -s /usr/local/lib/libopenblasp-r0.3.21.so /usr/local/lib/libblas.so.3 && \
            sudo ln -s /usr/local/lib/pkgconfig/openblas.pc /usr/local/lib/pkgconfig/blas.pc
        fi
    elif [[ -x "$(command -v yum)" ]]; then
        # for CentOS 7
        sudo yum install -y epel-release centos-release-scl-rh wget && \
        sudo yum install -y git make automake ccache python3-devel \
            devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-gcc-gfortran \
            llvm-toolset-7.0-clang llvm-toolset-7.0-clang-tools-extra

        echo "source scl_source enable devtoolset-7" | sudo tee -a /etc/profile.d/devtoolset-7.sh
        echo "source scl_source enable llvm-toolset-7.0" | sudo tee -a /etc/profile.d/llvm-toolset-7.sh
        echo "export CLANG_TOOLS_PATH=/opt/rh/llvm-toolset-7.0/root/usr/bin" | sudo tee -a /etc/profile.d/llvm-toolset-7.sh
        source "/etc/profile.d/llvm-toolset-7.sh"
        #DiskANN dependencies
        sudo yum -y install boost-program-options
        sudo yum -y install boost libaio
        #CMake 3.18 or higher is required
        wget -c https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-x86_64.tar.gz && \
        tar -zxvf cmake-3.22.2-linux-x86_64.tar.gz && \
        sudo ln -sf $(pwd)/cmake-3.22.2-linux-x86_64/bin/cmake /usr/bin/cmake
        pip3 install conan==1.61.0
        conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
    fi
fi

if [[ "${MACHINE}" == "Mac"  ]]; then
    brew install libomp llvm ninja openblas
    pip3 install conan==1.61.0
    conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
fi

if [[ "${MACHINE}" == "MinGw"  ]]; then
    pacman -Sy --noconfirm --needed \
    git make tar dos2unix zip unzip patch \
    mingw-w64-x86_64-toolchain \
    mingw-w64-x86_64-make \
    mingw-w64-x86_64-cmake \
    mingw-w64-x86_64-openblas
fi
