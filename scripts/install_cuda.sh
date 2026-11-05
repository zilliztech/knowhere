#!/usr/bin/env bash
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

# Prepare CUDA on CPU builders without installing a GPU driver.
set -euo pipefail

CUDA_VERSION=13.2.0-1
CUDA_ROOT=/usr/local/cuda-13.2

cuda_installed() {
    [[ -x "${CUDA_ROOT}/bin/nvcc" && -f "${CUDA_ROOT}/include/cuda_runtime.h" ]] || return 1
    "${CUDA_ROOT}/bin/nvcc" --version | grep -q 'release 13\.2,' || return 1
    local library
    for library in cudart cublas cublasLt cusparse cusolver; do
        [[ -e "${CUDA_ROOT}/lib64/lib${library}.so" ]] || return 1
    done
}

if cuda_installed; then
    echo "CUDA Toolkit 13.2 is already installed."
    exit 0
fi

source /etc/os-release
case "${ID}:${VERSION_ID}" in
    ubuntu:22.04) distribution=ubuntu2204 ;;
    ubuntu:24.04) distribution=ubuntu2404 ;;
    *) echo "Install CUDA Toolkit 13.2 at ${CUDA_ROOT} on ${ID} ${VERSION_ID}." >&2; exit 1 ;;
esac
case "$(uname -m)" in
    x86_64) architecture=x86_64 ;;
    aarch64) architecture=sbsa ;;
    *) echo "Unsupported CUDA build architecture: $(uname -m)" >&2; exit 1 ;;
esac

sudo_cmd=()
if [[ $EUID != 0 ]]; then
    sudo_cmd=(sudo -n)
fi
work_dir=$(mktemp -d)
trap 'rm -rf "${work_dir}"' EXIT
"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y ca-certificates curl
curl --fail --location --retry 3 \
    "https://developer.download.nvidia.com/compute/cuda/repos/${distribution}/${architecture}/cuda-keyring_1.1-1_all.deb" \
    -o "${work_dir}/cuda-keyring.deb"
"${sudo_cmd[@]}" dpkg -i "${work_dir}/cuda-keyring.deb"
"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y "cuda-toolkit-13-2=${CUDA_VERSION}"
if ! cuda_installed; then
    echo "CUDA Toolkit 13.2 installation is incomplete at ${CUDA_ROOT}." >&2
    exit 1
fi
