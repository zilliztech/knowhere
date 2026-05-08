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

CONAN_VERSION="2.28.1"
CONAN_REMOTE_URL="https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2"
CMAKE_MIN_VERSION="${CMAKE_MIN_VERSION:-3.28.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KNOWHERE_PYTHON_VENV="${KNOWHERE_PYTHON_VENV:-${REPO_ROOT}/python/.venv}"

UNAME="$(uname -s)"
case "${UNAME}" in
    Linux*)     OS=Linux;;
    Darwin*)    OS=Mac;;
    *)          OS="UNKNOWN:${UNAME}";;
esac

echo "[install_deps] Installing dependencies..."

version_ge() {
    local actual="$1"
    local required="$2"
    [[ "$(printf '%s\n%s\n' "${required}" "${actual}" | sort -V | head -n1)" == "${required}" ]]
}

cmake_version() {
    command -v cmake >/dev/null 2>&1 || return 1
    cmake --version | awk 'NR == 1 { print $3 }'
}

cmake_satisfies_minimum() {
    local current
    current="$(cmake_version 2>/dev/null || true)"
    [[ -n "${current}" ]] && version_ge "${current}" "${CMAKE_MIN_VERSION}"
}

pip3_install() {
    if pip3 install "$@"; then
        return
    fi

    echo "[install_deps] pip3 install failed; retrying with --break-system-packages for PEP 668 environments."
    pip3 install --break-system-packages "$@"
}

python_user_bin() {
    local user_base
    user_base="$(python3 -m site --user-base 2>/dev/null || true)"
    if [[ -n "${user_base}" ]]; then
        printf '%s/bin\n' "${user_base}"
    fi
}

install_uv() {
    if ! command -v pip3 >/dev/null 2>&1; then
        echo "[install_deps] pip3 is required to bootstrap uv."
        exit 1
    fi

    local user_bin
    user_bin="$(python_user_bin)"
    export PATH="${user_bin:+${user_bin}:}$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "[install_deps] Installing uv with pip3..."
        pip3_install --user uv
    fi

    if ! command -v uv >/dev/null 2>&1; then
        echo "[install_deps] uv was installed, but the uv executable is not on PATH."
        echo "[install_deps] Checked Python user bin: ${user_bin:-unavailable}; fallback bin: $HOME/.local/bin."
        exit 1
    fi
}

configure_uv_tool_path() {
    if [[ -z "${UV_TOOL_BIN_DIR:-}" ]]; then
        if [[ -w /usr/local/bin ]]; then
            export UV_TOOL_BIN_DIR="/usr/local/bin"
        else
            export UV_TOOL_BIN_DIR="$HOME/.local/bin"
        fi
    fi
    mkdir -p "${UV_TOOL_BIN_DIR}"
    export PATH="${UV_TOOL_BIN_DIR}:$HOME/.local/bin:$PATH"
}

persist_path_for_ci() {
    [[ -n "${GITHUB_PATH:-}" ]] || return 0

    local user_bin
    user_bin="$(python_user_bin)"

    printf '%s\n' "${UV_TOOL_BIN_DIR}" >> "${GITHUB_PATH}"
    if [[ -n "${user_bin}" ]]; then
        printf '%s\n' "${user_bin}" >> "${GITHUB_PATH}"
    fi
    printf '%s\n' "$HOME/.local/bin" >> "${GITHUB_PATH}"
}

install_python_tools() {
    install_uv
    configure_uv_tool_path

    echo "[install_deps] Installing Conan ${CONAN_VERSION} with uv..."
    uv tool install --force "conan==${CONAN_VERSION}"

    if [[ "${OS}" == "Linux" ]]; then
        echo "[install_deps] Preparing Python wheel environment with uv at ${KNOWHERE_PYTHON_VENV}..."
        if [[ ! -x "${KNOWHERE_PYTHON_VENV}/bin/python" ]]; then
            uv venv --python python3 "${KNOWHERE_PYTHON_VENV}"
        fi
        uv pip install --python "${KNOWHERE_PYTHON_VENV}/bin/python" -U setuptools wheel 'numpy>=2,<3' ml-dtypes auditwheel
    fi

    if [[ -x "${KNOWHERE_PYTHON_VENV}/bin/python" ]]; then
        export PATH="${KNOWHERE_PYTHON_VENV}/bin:${PATH}"
        export PYTHON="${KNOWHERE_PYTHON_VENV}/bin/python"
    fi

    persist_path_for_ci
}

if [[ "${OS}" == "Linux" ]]; then
    # Use sudo for package manager commands when not running as root.
    if [ "$(id -u)" -ne 0 ]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "[install_deps] Need root privileges or sudo to install packages."
            exit 1
        fi
        SUDO=(sudo)
    else
        SUDO=()
    fi

    run_privileged() {
        "${SUDO[@]}" "$@"
    }

    apt_get() {
        run_privileged env DEBIAN_FRONTEND=noninteractive apt-get "$@"
    }

    yum_cmd() {
        run_privileged yum "$@"
    }

    ubuntu_codename() {
        local codename=""
        if [[ -r /etc/os-release ]]; then
            # shellcheck disable=SC1091
            . /etc/os-release
            codename="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
        fi
        printf '%s\n' "${codename}"
    }

    install_llvm22_apt_repo() {
        local codename
        codename="$(ubuntu_codename)"

        case "${codename}" in
            jammy|noble)
                if apt-cache policy clang-tidy-22 2>/dev/null | awk '/Candidate:/ { found=1; ok=($2 != "(none)") } END { exit !(found && ok) }'; then
                    return
                fi

                # Equivalent to the repository setup in https://apt.llvm.org/llvm.sh,
                # but without invoking add-apt-repository or installing the full clang
                # toolchain. CI only needs the packages listed below.
                echo "[install_deps] Installing LLVM 22 apt repository for ${codename}..."
                apt_get install -y wget ca-certificates gnupg

                local key_tmp
                key_tmp="$(mktemp)"
                wget -O "${key_tmp}.asc" https://apt.llvm.org/llvm-snapshot.gpg.key
                gpg --dearmor -o "${key_tmp}.gpg" "${key_tmp}.asc"
                run_privileged install -D -m 0644 "${key_tmp}.gpg" /usr/share/keyrings/apt.llvm.org.gpg
                rm -f "${key_tmp}" "${key_tmp}.asc" "${key_tmp}.gpg"

                printf 'deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/%s/ llvm-toolchain-%s-22 main\n' "${codename}" "${codename}" \
                    | run_privileged tee /etc/apt/sources.list.d/llvm-toolchain-"${codename}"-22.list >/dev/null
                apt_get update
                ;;
            *)
                echo "[install_deps] Using distribution LLVM 22 packages for ${codename:-unknown Ubuntu release}."
                ;;
        esac
    }

    install_recent_cmake_apt() {
        if cmake_satisfies_minimum; then
            echo "[install_deps] CMake $(cmake_version) satisfies >= ${CMAKE_MIN_VERSION}."
            return
        fi

        local ubuntu_codename
        ubuntu_codename="$(ubuntu_codename)"

        case "${ubuntu_codename}" in
            jammy|noble)
                echo "[install_deps] Installing CMake >= ${CMAKE_MIN_VERSION} from Kitware for ${ubuntu_codename}..."
                apt_get install -y ca-certificates gnupg wget

                local key_tmp
                key_tmp="$(mktemp)"
                wget -O "${key_tmp}.asc" https://apt.kitware.com/keys/kitware-archive-latest.asc
                gpg --dearmor -o "${key_tmp}.gpg" "${key_tmp}.asc"
                run_privileged install -D -m 0644 "${key_tmp}.gpg" /usr/share/keyrings/kitware-archive-keyring.gpg
                rm -f "${key_tmp}" "${key_tmp}.asc" "${key_tmp}.gpg"

                printf 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ %s main\n' "${ubuntu_codename}" \
                    | run_privileged tee /etc/apt/sources.list.d/kitware.list >/dev/null
                apt_get update
                ;;
            *)
                echo "[install_deps] Using distribution CMake package for ${ubuntu_codename:-unknown Ubuntu release}."
                ;;
        esac

        apt_get install -y cmake
        if ! cmake_satisfies_minimum; then
            echo "[install_deps] CMake >= ${CMAKE_MIN_VERSION} is required, but installed $(cmake_version 2>/dev/null || echo none)."
            exit 1
        fi
        echo "[install_deps] Installed CMake $(cmake_version)."
    }

    if command -v apt-get >/dev/null 2>&1; then
        apt_get update || true
        install_llvm22_apt_repo

        packages=(
            g++ gcc make ccache
            libaio-dev                 # DiskANN async I/O
            libcurl4-openssl-dev       # folly dependency
            libdouble-conversion-dev   # folly dependency
            libevent-dev               # gRPC dependency
            libgflags-dev              # folly / glog dependency
            python3 python3-pip
            python3-venv                # uv-managed wheel environment
            clang-tidy-22              # CI: static analysis (analyzer CI)
            libomp-22-dev              # CI analysis/builds: version-matched OpenMP headers (omp.h)
            lcov                       # CI: code coverage reports
            binutils                   # wheel: readelf wheel RPATH inspection
            patchelf                   # wheel: RPATH rewriting
            python3-dev                # wheel: Python C extension headers
            swig                       # wheel: C++ → Python binding generator
            unzip                      # wheel: readelf wheel RPATH inspection
        )
        apt_get install -y "${packages[@]}"
        install_recent_cmake_apt

    elif command -v yum >/dev/null 2>&1; then
        yum_cmd makecache || true

        packages=(
            gcc-c++ gcc make cmake ccache
            libaio-devel             # DiskANN async I/O
            libcurl-devel            # folly dependency
            double-conversion-devel  # folly dependency
            libevent-devel           # gRPC dependency
            libomp-devel             # OpenMP parallelism
            gflags-devel             # folly / glog dependency
            python3 python3-pip
            clang-tools-extra        # CI: static analysis (provides clang-tidy)
            lcov                     # CI: code coverage reports
            binutils                 # wheel: readelf wheel RPATH inspection
            patchelf                 # wheel: RPATH rewriting
            python3-devel            # wheel: Python C extension headers
            swig                     # wheel: C++ → Python binding generator
            unzip                    # wheel: readelf wheel RPATH inspection
        )
        yum_cmd install -y "${packages[@]}"

    else
        echo "[install_deps] Unsupported Linux package manager. Expected apt-get or yum."
        exit 1
    fi

elif [[ "${OS}" == "Mac" ]]; then
    brew install libomp llvm ninja ccache cmake
    # libomp: OpenMP parallelization
    # llvm: clang toolchain
    # ninja: fast build system

else
    echo "[install_deps] Unsupported OS: ${OS}"
    exit 1
fi

install_python_tools

echo "[install_deps] Configuring conan profile and remote..."
conan profile detect --force || true
conan remote add default-conan-local2 ${CONAN_REMOTE_URL} || true

# Remove stale libelf cache from previous runs (uploaded with --only-recipe,
# missing exports_sources content). Let it be fetched from conancenter instead.
conan remove libelf/0.8.13 -c 2>/dev/null || true

echo "[install_deps] Done."
