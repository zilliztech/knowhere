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
#   - Linux x86_64 / aarch64 (Ubuntu 22.04, apt-based)
#   - macOS x86_64 / arm64 (Homebrew-based)
#
# Usage:
#   bash scripts/install_deps.sh

set -euo pipefail

CONAN_VERSION="1.65.0"
CONAN_REMOTE_URL="https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local"

UNAME="$(uname -s)"
case "${UNAME}" in
    Linux*)     OS=Linux;;
    Darwin*)    OS=Mac;;
    *)          OS="UNKNOWN:${UNAME}";;
esac

echo "[install_deps] Installing dependencies..."

if [[ "${OS}" == "Linux" ]]; then
    # Use sudo for apt-get when not running as root (e.g. GHA runners).
    if [ "$(id -u)" -ne 0 ]; then
        SUDO="sudo"
    else
        SUDO=""
    fi

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
        unzip                      # wheel: readelf wheel RPATH inspection
    )
    ${SUDO} apt-get install -y "${packages[@]}"

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

echo "[install_deps] Configuring conan remote..."
conan remote add default-conan-local ${CONAN_REMOTE_URL} || true

echo "[install_deps] Done."
