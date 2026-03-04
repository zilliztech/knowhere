#!/usr/bin/env bash
# Centralized conan build configuration for all CI pipelines.
# Single source of truth for conan install flags — update HERE, not in each pipeline file.
#
# Usage: ci/conan_build.sh [options]
#   --gpu        Enable cuVS GPU support (-o with_cuvs=True)
#   --cardinal   Enable Cardinal support (-o with_cardinal=True)
#   --ut         Enable unit tests (-o with_ut=True)
#   --asan       Enable address sanitizer (-o with_asan=True)
#   --cpu        Add CPU-specific flags (--build=liburing)
#   --release    Set build type to Release (-s build_type=Release)
#   --no-cppstd  Omit -s compiler.cppstd=17 (for legacy release pipelines that
#                historically built without it — preserve existing behavior)
#   --no-build   Only run conan install, skip conan build
set -eo pipefail

FLAGS="--update --build=missing -s compiler.libcxx=libstdc++11 -o with_diskann=True"
NO_BUILD=0
ADD_CPPSTD=1

for arg in "$@"; do
    case "$arg" in
        --gpu)       FLAGS="$FLAGS -o with_cuvs=True" ;;
        --cardinal)  FLAGS="$FLAGS -o with_cardinal=True" ;;
        --ut)        FLAGS="$FLAGS -o with_ut=True" ;;
        --asan)      FLAGS="$FLAGS -o with_asan=True" ;;
        --cpu)       FLAGS="$FLAGS --build=liburing" ;;
        --release)   FLAGS="$FLAGS -s build_type=Release" ;;
        --no-cppstd) ADD_CPPSTD=0 ;;
        --no-build)  NO_BUILD=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

if [ "$ADD_CPPSTD" -eq 1 ]; then
    FLAGS="$FLAGS -s compiler.cppstd=17"
fi

echo "[conan_build.sh] conan install .. $FLAGS"
conan install .. $FLAGS

if [ "$NO_BUILD" -eq 0 ]; then
    echo "[conan_build.sh] conan build .."
    conan build ..
fi
