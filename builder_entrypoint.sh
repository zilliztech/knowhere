#!/usr/bin/env bash

BUILD_TYPE=$1

# Trap to ensure we always drop into shell, even on error
trap 'echo "==> Error occurred! Entering interactive shell for debugging..."; exec /bin/bash' ERR

set -e

echo "==> Starting full build with mode: $BUILD_TYPE"

echo "==> Step 1: Cleaning build directory"
rm -rf build

echo "==> Step 2: Creating build directory"
mkdir build && cd build

echo "==> Step 3: Running conan install"
conan install .. --build=missing -o with_ut=True -o with_diskann=True -s compiler.libcxx=libstdc++11 -s build_type=$BUILD_TYPE

echo "==> Step 4: Running conan build"
conan build ..
echo "==> Build complete!"

echo "==> Step 5: Starting Redis server for NCS tests"
redis-server --daemonize yes

echo "Entering interactive shell..."
exec /bin/bash
