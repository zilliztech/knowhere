#!/usr/bin/env bash

# This script is to modify CMakeLists.txt to build knowhere for certain GPU serials
#   GTX 1060 - sm_61
#   GTX 1660 - sm_75
#   GTX 2080 SUPER - sm_75
sed 's/set(CMAKE_CUDA_ARCHITECTURES .*$/set(CMAKE_CUDA_ARCHITECTURES \"75-real\")/' CMakeLists.txt > tmp
mv tmp CMakeLists.txt
