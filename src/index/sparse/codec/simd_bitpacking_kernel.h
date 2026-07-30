// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
// on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
// for the specific language governing permissions and limitations under the License.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void
knowhere_simd_pack_128_blocks(const uint32_t* in, uint8_t* out, size_t block_count, uint32_t bits);

void
knowhere_simd_unpack_128_blocks(const uint8_t* in, uint32_t* out, size_t block_count, uint32_t bits);

void
knowhere_simd_unpack_d1_128_blocks(const uint8_t* in, uint32_t* out, size_t block_count, uint32_t bits,
                                   uint32_t previous_value);

void
knowhere_simd_integrate_doc_id_gaps(uint32_t* values, size_t count, uint32_t previous_value);

#ifdef __cplusplus
}
#endif
