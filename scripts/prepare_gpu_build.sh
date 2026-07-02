#!/usr/bin/env bash

# This script sets CMAKE_CUDA_ARCHITECTURES to RAPIDS, which compiles for all
# supported GPU architectures (Turing through Blackwell) at build time.
sed 's/set(CMAKE_CUDA_ARCHITECTURES .*$/set(CMAKE_CUDA_ARCHITECTURES RAPIDS)/' CMakeLists.txt > tmp
mv tmp CMakeLists.txt
