#!/usr/bin/env bash
# CI dependency installer for Knowhere pipelines.
#
# Installs all dependencies needed across CI pipelines:
#   bash scripts/ci_deps.sh
#
# Pipeline-specific extras (e.g. libopenblas-dev, build-essential, git)
# should be installed inline in the pipeline after calling this script.

set -euo pipefail

# Use sudo for apt-get when not running as root (e.g. GHA runners).
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

CONAN_VERSION="1.65.0"
CONAN_REMOTE_URL="https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local"

echo "[ci_deps] Installing dependencies..."

${SUDO} apt-get update || true
${SUDO} apt-get install -y \
    libaio-dev \
    libcurl4-openssl-dev \
    libdouble-conversion-dev \
    libevent-dev \
    libgflags-dev \
    python3 \
    python3-pip \
    g++ \
    gcc \
    ccache \
    libopenblas-openmp-dev \
    unzip \
    binutils \
    patchelf \
    cmake \
    clang-tidy-14 \
    libomp-dev \
    lcov

pip3 install conan==${CONAN_VERSION}
pip3 install -U setuptools
# wheel must be installed before bfloat16: bfloat16 has no pre-built wheel
# for Python 3.8, so pip builds from source. Without the wheel package, pip
# uses PEP 517 build isolation which can't see the installed numpy.
pip3 install wheel 'numpy<2'
pip3 install bfloat16 auditwheel

echo "[ci_deps] Configuring conan remote..."
conan remote add default-conan-local ${CONAN_REMOTE_URL} || true

echo "[ci_deps] Done."
