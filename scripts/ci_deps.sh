#!/usr/bin/env bash
# Composable CI dependency installer for Knowhere pipelines.
# Source this script and call the functions you need:
#
#   source scripts/ci_deps.sh
#   install_base_deps
#   install_build_deps
#   install_wheel_deps
#
# Functions can be composed in any order. Each is idempotent.

set -eo pipefail

# Use sudo for apt-get when not running as root (e.g. GHA runners).
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

CONAN_VERSION="1.65.0"
CONAN_REMOTE_URL="https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local"

# Core libraries needed by every knowhere build.
install_base_deps() {
    echo "[ci_deps] Installing base dependencies..."
    ${SUDO} apt-get update || true
    ${SUDO} apt-get install -y \
        libaio-dev \
        libcurl4-openssl-dev \
        libdouble-conversion-dev \
        libevent-dev \
        libgflags-dev \
        python3 \
        python3-pip
    pip3 install conan==${CONAN_VERSION}
}

# C++ compiler toolchain (for GHA runners; Docker images already have these).
install_build_deps() {
    echo "[ci_deps] Installing build dependencies..."
    ${SUDO} apt-get install -y \
        g++ \
        gcc \
        ccache \
        libopenblas-openmp-dev
}

# Additional packages for building and packaging python wheels.
install_wheel_deps() {
    echo "[ci_deps] Installing wheel/packaging dependencies..."
    ${SUDO} apt-get install -y \
        unzip \
        binutils \
        patchelf
    pip3 install -U setuptools
    pip3 install 'numpy<2'
    pip3 install --no-build-isolation bfloat16
    pip3 install auditwheel
}

# Dependencies for E2E test stages (running an installed wheel).
install_test_runner_deps() {
    echo "[ci_deps] Installing test runner dependencies..."
    ${SUDO} apt-get update || true
    ${SUDO} apt-get install -y \
        libopenblas-openmp-dev \
        libaio-dev \
        libdouble-conversion-dev \
        libevent-dev
}

# clang-tidy for static analysis.
install_analyzer_deps() {
    echo "[ci_deps] Installing analyzer dependencies..."
    ${SUDO} apt-get install -y \
        cmake \
        clang-tidy-14 \
        libomp-dev
}

# lcov for code coverage reports.
install_coverage_deps() {
    echo "[ci_deps] Installing coverage dependencies..."
    ${SUDO} apt-get install -y lcov
}

# Configure the Milvus conan remote.
setup_conan_remote() {
    echo "[ci_deps] Configuring conan remote..."
    conan remote add default-conan-local ${CONAN_REMOTE_URL} || true
}
