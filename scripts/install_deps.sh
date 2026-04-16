#!/usr/bin/env bash
#
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
#
# Dependency installer for Knowhere (local dev and CI).
#
# Supported platforms:
#   - Linux x86_64 / aarch64 (apt-based or yum-based)
#   - macOS x86_64 / arm64 (Homebrew-based)
#
# Usage:
#   bash scripts/install_deps.sh

set -euo pipefail

CONAN_VERSION="2.25.1"
CONAN_REMOTE_URL="https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2"

UNAME="$(uname -s)"
case "${UNAME}" in
    Linux*)     OS=Linux;;
    Darwin*)    OS=Mac;;
    *)          OS="UNKNOWN:${UNAME}";;
esac

echo "[install_deps] Installing dependencies..."

if [[ "${OS}" == "Linux" ]]; then
    # Use sudo for package manager commands when not running as root.
    if [ "$(id -u)" -ne 0 ]; then
        SUDO="sudo"
    else
        SUDO=""
    fi

    if command -v apt-get >/dev/null 2>&1; then
        ${SUDO} apt-get update || true

        packages=(
            g++ gcc make cmake ccache
            libaio-dev                 # DiskANN async I/O
            libcurl4-openssl-dev       # folly dependency
            libdouble-conversion-dev   # folly dependency
            libevent-dev               # gRPC dependency
            libgflags-dev              # folly / glog dependency
            libopenblas-openmp-dev     # faiss BLAS operations
            libomp-dev                 # OpenMP parallelization
            python3 python3-pip
            clang-tidy-14              # CI: static analysis (analyzer CI)
            lcov                       # CI: code coverage reports
            binutils                   # wheel: readelf wheel RPATH inspection
            patchelf                   # wheel: RPATH rewriting
            python3-dev                # wheel: Python C extension headers
            swig                       # wheel: C++ → Python binding generator
            unzip                      # wheel: readelf wheel RPATH inspection
        )
        ${SUDO} apt-get install -y "${packages[@]}"

    elif command -v yum >/dev/null 2>&1; then
        ${SUDO} yum makecache || true

        packages=(
            gcc-c++ gcc make cmake ccache
            libaio-devel             # DiskANN async I/O
            libcurl-devel            # folly dependency
            double-conversion-devel  # folly dependency
            libevent-devel           # gRPC dependency
            gflags-devel             # folly / glog dependency
            openblas-devel           # faiss BLAS operations
            libomp-devel             # OpenMP parallelization
            python3 python3-pip
            clang-tools-extra        # CI: static analysis (provides clang-tidy)
            lcov                     # CI: code coverage reports
            binutils                 # wheel: readelf wheel RPATH inspection
            patchelf                 # wheel: RPATH rewriting
            python3-devel            # wheel: Python C extension headers
            swig                     # wheel: C++ → Python binding generator
            unzip                    # wheel: readelf wheel RPATH inspection
        )
        ${SUDO} yum install -y "${packages[@]}"

    else
        echo "[install_deps] Unsupported Linux package manager. Expected apt-get or yum."
        exit 1
    fi

elif [[ "${OS}" == "Mac" ]]; then
    brew install libomp llvm ninja openblas ccache cmake
    # libomp: OpenMP parallelization
    # llvm: clang toolchain
    # ninja: fast build system
    # openblas: faiss BLAS operations

else
    echo "[install_deps] Unsupported OS: ${OS}"
    exit 1
fi

pip3 install conan==${CONAN_VERSION}  # C/C++ package manager

pip3 install -U setuptools
# wheel must be installed before bfloat16: bfloat16 has no pre-built wheel
# for Python 3.8, so pip builds from source. Without the wheel package, pip
# uses PEP 517 build isolation which can't see the installed numpy.
pip3 install wheel 'numpy<2'
pip3 install bfloat16   # wheel: bfloat16 dtype support for PyKnowhere
pip3 install auditwheel  # wheel: manylinux wheel repair

echo "[install_deps] Configuring conan profile and remote..."
conan profile detect --force || true
conan remote add default-conan-local2 ${CONAN_REMOTE_URL} || true

echo "[install_deps] Done."
