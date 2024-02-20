#!/usr/bin/env bash

# This script is to modify CMakeLists.txt to build knowhere for certain GPU serials
#   GTX 1060 - sm_61
#   GTX 1660 - sm_75
sed 's/set(CMAKE_CUDA_ARCHITECTURES \${supported_archs})/set(CMAKE_CUDA_ARCHITECTURES \"61-real;75-real\")/' CMakeLists.txt > tmp
mv tmp CMakeLists.txt
