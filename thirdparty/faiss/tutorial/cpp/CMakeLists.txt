# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(1-Flat EXCLUDE_FROM_ALL 1-Flat.cpp)
target_link_libraries(1-Flat PRIVATE faiss)

add_executable(2-IVFFlat EXCLUDE_FROM_ALL 2-IVFFlat.cpp)
target_link_libraries(2-IVFFlat PRIVATE faiss)

add_executable(3-IVFPQ EXCLUDE_FROM_ALL 3-IVFPQ.cpp)
target_link_libraries(3-IVFPQ PRIVATE faiss)

add_executable(4-GPU EXCLUDE_FROM_ALL 4-GPU.cpp)
target_link_libraries(4-GPU PRIVATE faiss)

add_executable(5-Multiple-GPUs EXCLUDE_FROM_ALL 5-Multiple-GPUs.cpp)
target_link_libraries(5-Multiple-GPUs PRIVATE faiss)

add_executable(5-GPU EXCLUDE_FROM_ALL 5-GPU.cpp)
target_link_libraries(5-GPU PRIVATE faiss)

add_executable(6-GPU EXCLUDE_FROM_ALL 6-GPU.cpp)
target_link_libraries(6-GPU PRIVATE faiss)

add_executable(6-RUN EXCLUDE_FROM_ALL 6-RUN.cpp)
target_link_libraries(6-RUN PRIVATE faiss)

add_executable(7-GPU EXCLUDE_FROM_ALL 7-GPU.cpp)
target_link_libraries(7-GPU PRIVATE faiss)

add_executable(8-GPU EXCLUDE_FROM_ALL 8-GPU.cpp)
target_link_libraries(8-GPU PRIVATE faiss)

add_executable(9-BinaryFlat EXCLUDE_FROM_ALL 9-BinaryFlat.cpp)
target_link_libraries(9-BinaryFlat PRIVATE faiss)

add_executable(tutorial_faiss_test EXCLUDE_FROM_ALL tutorial_faiss_test.cpp)
target_link_libraries(tutorial_faiss_test PRIVATE faiss)