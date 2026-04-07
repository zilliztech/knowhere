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
# bfloat16 dtype support for PyKnowhere.
# On x86_64 a pre-built wheel exists; on aarch64 only the 1.1 sdist is
# available, which (a) imports numpy in setup.py and (b) uses the removed
# Py_TYPE-as-lvalue pattern incompatible with Python 3.11+.
pip3 install wheel 'numpy<2'
ARCH="$(uname -m)"
if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    # bfloat16 1.1 (only sdist for ARM) uses the Py_TYPE-as-lvalue pattern
    # removed in Python 3.11+. Download, patch, and install from source.
    _bf16_dir=$(mktemp -d)
    curl -sL https://files.pythonhosted.org/packages/source/b/bfloat16/bfloat16-1.1.tar.gz \
        -o "${_bf16_dir}/bfloat16-1.1.tar.gz"
    tar xzf "${_bf16_dir}/bfloat16-1.1.tar.gz" -C "${_bf16_dir}"
    sed -i 's/Py_TYPE(&NPyBfloat16_Descr) = &PyArrayDescr_Type;/Py_SET_TYPE(\&NPyBfloat16_Descr, \&PyArrayDescr_Type);/' \
        "${_bf16_dir}/bfloat16-1.1/bfloat16.cc"
    pip3 install --no-build-isolation "${_bf16_dir}/bfloat16-1.1/"
    rm -rf "${_bf16_dir}"
else
    pip3 install bfloat16
fi
pip3 install auditwheel  # wheel: manylinux wheel repair

echo "[install_deps] Configuring conan remote..."
conan remote add default-conan-local ${CONAN_REMOTE_URL} || true

echo "[install_deps] Done."
