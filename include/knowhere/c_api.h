/* Copyright (C) 2026 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#ifndef KNOWHERE_C_API_H
#define KNOWHERE_C_API_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(KNOWHERE_C_API_BUILD)
#define KNOWHERE_C_API __declspec(dllexport)
#else
#define KNOWHERE_C_API __declspec(dllimport)
#endif
#else
#define KNOWHERE_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ABI 1: fixed-width, native-endian values. Handles are opaque, typed resource
 * identifiers, never addresses, and never reused during the library lifetime.
 * Do not unload the library while handles or calls remain active.
 *
 * All status-returning functions catch C++ exceptions. On failure, read
 * knowhere_last_error() immediately on the same thread. Calls on one index or
 * BinarySet are serialized. Destroy prevents new operations and is idempotent;
 * already-started calls retain ownership until they finish. Other operations on
 * a destroyed handle return KNOWHERE_CLOSED. Mixing resource types is invalid.
 *
 * Every pointer must designate an accessible region of the stated capacity.
 * Buffers are borrowed only for the duration of a synchronous call, except
 * Build/Deserialize, which make retained native copies. The caller must prevent
 * concurrent modification of borrowed input and output buffers.
 */
typedef uint64_t knowhere_index_handle;
typedef uint64_t knowhere_binary_set_handle;

enum knowhere_dtype { KNOWHERE_FP32 = 1, KNOWHERE_BIN1 = 2, KNOWHERE_FP16 = 3, KNOWHERE_BF16 = 4, KNOWHERE_INT8 = 5 };

enum knowhere_error_code {
    KNOWHERE_SUCCESS = 0,
    KNOWHERE_INVALID_ARGUMENT = 1,
    KNOWHERE_CLOSED = 2,
    KNOWHERE_UNSUPPORTED = 3,
    KNOWHERE_ERROR = 4,
    KNOWHERE_OUT_OF_MEMORY = 5
};

/* Contiguous row-major vectors with no stride. rows >= 0, dimensions > 0.
 * dimensions counts bits for BIN1 and elements otherwise; BIN1 dimensions must
 * be a multiple of 8. FP16/BF16 contain IEEE binary16/bfloat16 bit patterns.
 * FP32 buffers require 4-byte alignment, FP16/BF16 2-byte alignment. bytes may
 * exceed the required size. Zero-row data may be null. Sizes must fit size_t
 * and INT64_MAX; all dimension/size multiplication is checked for overflow.
 */
typedef struct knowhere_vectors {
    const void* data;
    uint64_t bytes;
    int64_t rows;
    int64_t dimensions;
    int32_t dtype;
} knowhere_vectors;

/* Bit i (least-significant bit first) excludes row i when set. A null struct
 * or bits == 0 means no filter, and data may then be null. Otherwise bits must
 * cover all base/index rows and bytes must cover ceil(bits/8); trailing bits
 * outside the base/index are ignored. Negative bits are invalid.
 */
typedef struct knowhere_bitset {
    const uint8_t* data;
    uint64_t bytes;
    int64_t bits;
} knowhere_bitset;

/* Caller-owned nq * top_k output slots in row-major order. ids requires
 * 8-byte alignment and distances 4-byte alignment. top_k must be positive and
 * is authoritative: a conflicting JSON "k" is invalid. Zero queries validate
 * shape/filter/parameters but leave output unchanged. On error, output buffer
 * contents are unspecified. Distance semantics follow Knowhere's metric_type.
 */
typedef struct knowhere_search_result {
    int64_t* ids;
    uint64_t ids_bytes;
    float* distances;
    uint64_t distances_bytes;
    int64_t top_k;
} knowhere_search_result;

/* Thread-local diagnostic, valid until the next status-returning API call on
 * this thread. Empty after success; messages can be truncated at 4095 bytes.
 * This accessor and the version accessors do not clear the diagnostic.
 */
KNOWHERE_C_API const char*
knowhere_last_error(void);
KNOWHERE_C_API int32_t
knowhere_c_abi_version(void);
KNOWHERE_C_API int32_t
knowhere_index_version_minimum(void);
KNOWHERE_C_API int32_t
knowhere_index_version_current(void);
KNOWHERE_C_API int32_t
knowhere_index_version_maximum(void);

/* Process-wide thread pools. Index search and brute force schedule one task per
 * query row on the search pool and run each task single-threaded, so a call
 * with one query occupies one pool thread; Build runs on the build pool. A pool
 * that was never sized is created on first use with the hardware thread count.
 * Resize creates the pool at the given size or resizes the existing pool; it is
 * safe at any time, but only a call before the first search or build replaces
 * the default. threads must be in [1, INT32_MAX]. The size accessors report 0
 * while the pool does not exist yet and require a non-null output.
 */
KNOWHERE_C_API int32_t
knowhere_search_thread_pool_resize(int64_t threads);
KNOWHERE_C_API int32_t
knowhere_search_thread_pool_size(int64_t* output);
KNOWHERE_C_API int32_t
knowhere_build_thread_pool_resize(int64_t threads);
KNOWHERE_C_API int32_t
knowhere_build_thread_pool_size(int64_t* output);

/* Create any index/dtype combination registered in this Knowhere build. Index
 * versions outside the advertised inclusive range are unsupported. DISKANN and
 * INDEX_CARDINAL_TIERED receive a LocalFileManager. All handle outputs are zero
 * on failure. parameters throughout this API is a JSON object, or null for {}.
 */
KNOWHERE_C_API int32_t
knowhere_index_create(const char* type, int32_t dtype, int32_t version, knowhere_index_handle* output);
KNOWHERE_C_API int32_t
knowhere_index_destroy(knowhere_index_handle handle);

/* Initialize exactly once, by either Build or Deserialize. After an engine
 * initialization failure, destroy and recreate the index; partially initialized
 * state cannot be reused. Parameter validation failures before the engine call
 * leave the index uninitialized. Build owns a copy of all vector bytes.
 *
 * DISKANN is a file-build index: vectors MUST be null, and JSON data_path names
 * a local file containing uint32 rows, uint32 dimensions, then vector data in
 * the creation dtype. Build writes files under JSON index_prefix. Other index
 * types require non-null vectors; injecting LocalFileManager into a factory
 * does not by itself change that index's Build input contract.
 */
KNOWHERE_C_API int32_t
knowhere_index_build(knowhere_index_handle handle, const knowhere_vectors* vectors, const char* parameters);
/* Query dtype must match creation dtype and dimensions must match the index. */
KNOWHERE_C_API int32_t
knowhere_index_search(knowhere_index_handle handle, const knowhere_vectors* queries, const knowhere_bitset* excluded,
                      knowhere_search_result* result, const char* parameters);
/* Requires an initialized index; both output pointers are required. */
KNOWHERE_C_API int32_t
knowhere_index_info(knowhere_index_handle handle, int64_t* rows, int64_t* dimensions);
/* Requires an initialized index; the returned BinarySet is caller-owned.
 * DISKANN's BinarySet does not contain its disk index files. After a file build,
 * serialize and close the builder, then create a new index and deserialize
 * with the same JSON index_prefix before searching. Keep those index files
 * available for the lifetime of every index loaded from that prefix.
 */
KNOWHERE_C_API int32_t
knowhere_index_serialize(knowhere_index_handle handle, knowhere_binary_set_handle* output);
/* Takes an independent snapshot of the BinarySet. It may be changed or closed
 * after this call without invalidating the index. The caller must select the
 * same index type, dtype and compatible version as the original index.
 */
KNOWHERE_C_API int32_t
knowhere_index_deserialize(knowhere_index_handle handle, knowhere_binary_set_handle data, const char* parameters);
/* Exact search through Knowhere's BruteForce::SearchWithBuf. Base and queries
 * must have the same dtype and dimensions. Both vector buffers are borrowed.
 */
KNOWHERE_C_API int32_t
knowhere_bruteforce(const knowhere_vectors* base, const knowhere_vectors* queries, const knowhere_bitset* excluded,
                    knowhere_search_result* result, const char* parameters);

KNOWHERE_C_API int32_t
knowhere_binary_set_create(knowhere_binary_set_handle* output);
KNOWHERE_C_API int32_t
knowhere_binary_set_destroy(knowhere_binary_set_handle handle);
KNOWHERE_C_API int32_t
knowhere_binary_set_count(knowhere_binary_set_handle handle, uint64_t* output);
/* Names are enumerated in lexicographic order. required includes the NUL byte.
 * output == null and capacity == 0 queries size. A short buffer is invalid and
 * does not receive partial output, but required is still set. index is zero-based.
 */
KNOWHERE_C_API int32_t
knowhere_binary_set_name(knowhere_binary_set_handle handle, uint64_t index, char* output, uint64_t capacity,
                         uint64_t* required);
KNOWHERE_C_API int32_t
knowhere_binary_set_length(knowhere_binary_set_handle handle, const char* name, uint64_t* output);
/* Allocate or replace an entry with zero-filled storage. Names are nonempty,
 * NUL-terminated strings. Length is uint64_t for large blobs, but must fit
 * size_t and INT64_MAX because Knowhere uses signed 64-bit Binary lengths.
 */
KNOWHERE_C_API int32_t
knowhere_binary_set_allocate(knowhere_binary_set_handle handle, const char* name, uint64_t length);
/* Chunk ranges must be completely within the entry: no partial read/write.
 * A zero-length chunk at the end is valid and its data pointer may be null.
 */
KNOWHERE_C_API int32_t
knowhere_binary_set_write(knowhere_binary_set_handle handle, const char* name, uint64_t offset, const void* data,
                          uint64_t length);
KNOWHERE_C_API int32_t
knowhere_binary_set_read(knowhere_binary_set_handle handle, const char* name, uint64_t offset, void* data,
                         uint64_t length);

#ifdef __cplusplus
}
#endif
#endif /* KNOWHERE_C_API_H */
