/* Copyright (C) 2026 Zilliz. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "knowhere/c_api.h"

#define CHECK(x)                                                                               \
    do {                                                                                       \
        if (!(x)) {                                                                            \
            fprintf(stderr, "%s:%d: %s; %s\n", __FILE__, __LINE__, #x, knowhere_last_error()); \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)
#define OK(x) CHECK((x) == KNOWHERE_SUCCESS)

static void
check_hits(knowhere_search_result* result, int filtered) {
    CHECK(result->ids[0] == (filtered ? 1 : 0));
    CHECK(result->ids[1] == (filtered ? 2 : 1));
    CHECK(result->distances[0] == (filtered ? 1.0f : 0.0f));
    CHECK(result->distances[1] == (filtered ? 4.0f : 1.0f));
}

static void
test_binary_set(void) {
    knowhere_binary_set_handle set = 0;
    uint64_t n = 0;
    unsigned char bytes[6] = {0};
    char name[16];
    OK(knowhere_binary_set_create(&set));
    OK(knowhere_binary_set_allocate(set, "one", 6));
    OK(knowhere_binary_set_allocate(set, "two", 2));
    OK(knowhere_binary_set_count(set, &n));
    CHECK(n == 2);
    OK(knowhere_binary_set_name(set, 0, NULL, 0, &n));
    CHECK(n == 4);
    CHECK(knowhere_binary_set_name(set, 0, name, 2, &n) == KNOWHERE_INVALID_ARGUMENT);
    OK(knowhere_binary_set_name(set, 0, name, sizeof(name), &n));
    CHECK(strcmp(name, "one") == 0);
    OK(knowhere_binary_set_read(set, "one", 0, bytes, sizeof(bytes)));
    CHECK(memcmp(bytes, "\0\0\0\0\0\0", 6) == 0);
    OK(knowhere_binary_set_write(set, "one", 0, "abc", 3));
    OK(knowhere_binary_set_write(set, "one", 3, "def", 3));
    OK(knowhere_binary_set_read(set, "one", 1, bytes, 4));
    CHECK(memcmp(bytes, "bcde", 4) == 0);
    CHECK(knowhere_binary_set_read(set, "one", 4, bytes, 3) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_binary_set_write(set, "one", UINT64_MAX, bytes, 1) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_binary_set_length(set, "missing", &n) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_binary_set_allocate(set, "huge", UINT64_MAX) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_binary_set_allocate(set, "", 1) == KNOWHERE_INVALID_ARGUMENT);
    OK(knowhere_binary_set_write(set, "one", 6, NULL, 0));
    OK(knowhere_binary_set_read(set, "one", 6, NULL, 0));
    CHECK(knowhere_binary_set_write(set, "one", 7, NULL, 0) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_index_destroy(set) == KNOWHERE_INVALID_ARGUMENT);
    OK(knowhere_binary_set_destroy(set));
    OK(knowhere_binary_set_destroy(set));
    CHECK(knowhere_binary_set_count(set, &n) == KNOWHERE_CLOSED);
}

static void
test_typed(int dtype, const void* base_data, uint64_t bytes, const void* query_data, uint64_t query_bytes, int64_t dim,
           const char* type, const char* metric, int binary) {
    knowhere_vectors base = {base_data, bytes, 3, dim, dtype};
    knowhere_vectors query = {query_data, query_bytes, 1, dim, dtype};
    int64_t ids[2] = {-1, -1};
    float distances[2] = {-1, -1};
    knowhere_search_result result = {ids, sizeof(ids), distances, sizeof(distances), 2};
    unsigned char excluded_data = 1;
    knowhere_bitset excluded = {&excluded_data, 1, 3};
    knowhere_index_handle index = 0, loaded = 0;
    knowhere_binary_set_handle set = 0;
    char parameters[128];
    int64_t rows, dimensions;
    if (strcmp(type, "HNSW") == 0) {
        snprintf(parameters, sizeof(parameters), "{\"metric_type\":\"%s\",\"M\":4,\"efConstruction\":16,\"ef\":16}",
                 metric);
    } else if (strcmp(type, "IVF_FLAT") == 0) {
        snprintf(parameters, sizeof(parameters), "{\"metric_type\":\"%s\",\"nlist\":1,\"nprobe\":1}", metric);
    } else {
        snprintf(parameters, sizeof(parameters), "{\"metric_type\":\"%s\"}", metric);
    }
    OK(knowhere_bruteforce(&base, &query, NULL, &result, parameters));
    if (binary) {
        CHECK(ids[0] == 0 && ids[1] == 1 && distances[0] == 0 && distances[1] == 1);
    } else {
        check_hits(&result, 0);
    }
    OK(knowhere_bruteforce(&base, &query, &excluded, &result, parameters));
    if (binary) {
        CHECK(ids[0] == 1 && ids[1] == 2 && distances[0] == 1 && distances[1] == 2);
    } else {
        check_hits(&result, 1);
    }
    OK(knowhere_index_create(type, dtype, knowhere_index_version_current(), &index));
    CHECK(knowhere_index_search(index, &query, NULL, &result, parameters) != KNOWHERE_SUCCESS);
    OK(knowhere_index_build(index, &base, parameters));
    CHECK(knowhere_index_build(index, &base, parameters) != KNOWHERE_SUCCESS);
    OK(knowhere_index_info(index, &rows, &dimensions));
    CHECK(rows == 3 && dimensions == dim);
    OK(knowhere_index_search(index, &query, NULL, &result, parameters));
    CHECK(ids[0] == 0 && ids[1] == 1 && distances[0] == 0 && distances[1] == 1);
    OK(knowhere_index_serialize(index, &set));
    OK(knowhere_index_destroy(index));
    OK(knowhere_index_destroy(index));
    CHECK(knowhere_index_info(index, &rows, &dimensions) == KNOWHERE_CLOSED);
    OK(knowhere_index_create(type, dtype, knowhere_index_version_current(), &loaded));
    OK(knowhere_index_deserialize(loaded, set, parameters));
    CHECK(knowhere_index_deserialize(loaded, set, parameters) != KNOWHERE_SUCCESS);
    /* Mutating the source BinarySet after Deserialize must not change the index. */
    {
        uint64_t count, i, required, length;
        OK(knowhere_binary_set_count(set, &count));
        for (i = 0; i < count; ++i) {
            char* name;
            OK(knowhere_binary_set_name(set, i, NULL, 0, &required));
            name = (char*)malloc((size_t)required);
            CHECK(name != NULL);
            OK(knowhere_binary_set_name(set, i, name, required, &required));
            OK(knowhere_binary_set_length(set, name, &length));
            OK(knowhere_binary_set_allocate(set, name, length));
            free(name);
        }
    }
    OK(knowhere_binary_set_destroy(set));
    OK(knowhere_index_search(loaded, &query, &excluded, &result, parameters));
    CHECK(ids[0] == 1 && ids[1] == 2 && distances[0] == 1 && distances[1] == (binary ? 2 : 4));
    OK(knowhere_index_destroy(loaded));
}

static void
test_invalid(void) {
    float data[6] = {0, 0, 1, 0, 0, 2};
    knowhere_vectors base = {data, sizeof(data), 3, 2, KNOWHERE_FP32};
    knowhere_vectors query = {data, 8, 1, 2, KNOWHERE_FP32};
    int64_t ids[2];
    float dis[2];
    knowhere_search_result result = {ids, sizeof(ids), dis, sizeof(dis), 2};
    knowhere_index_handle index = 99;
    knowhere_bitset excluded = {(const uint8_t*)data, 0, 3};
    CHECK(knowhere_index_create("FLAT", 99, 0, &index) == KNOWHERE_UNSUPPORTED && index == 0);
    CHECK(knowhere_index_create("FLAT", KNOWHERE_FP32, -1, &index) == KNOWHERE_UNSUPPORTED && index == 0);
    CHECK(knowhere_index_create("unknown", KNOWHERE_FP32, 0, &index) != KNOWHERE_SUCCESS && index == 0);
    CHECK(knowhere_bruteforce(&base, &query, NULL, &result, "{") == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_bruteforce(&base, &query, NULL, &result, "[]") == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_bruteforce(&base, &query, NULL, &result, "{\"k\":1}") == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_bruteforce(&base, &query, &excluded, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
    query.bytes = 1;
    CHECK(knowhere_bruteforce(&base, &query, NULL, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
    query.bytes = 8;
    query.dimensions = 3;
    CHECK(knowhere_bruteforce(&base, &query, NULL, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
    query.dimensions = 2;
    query.rows = 0;
    query.bytes = 0;
    query.data = NULL;
    ids[0] = 123;
    OK(knowhere_bruteforce(&base, &query, NULL, &result, "{\"metric_type\":\"L2\"}"));
    CHECK(ids[0] == 123);
    {
        knowhere_index_handle empty = 0;
        knowhere_binary_set_handle serialized = 77;
        knowhere_vectors mismatch = {data, 4, 1, 2, KNOWHERE_FP16};
        knowhere_vectors unaligned = {(const unsigned char*)data + 1, 8, 1, 2, KNOWHERE_FP32};
        OK(knowhere_index_create("FLAT", KNOWHERE_FP32, knowhere_index_version_current(), &empty));
        CHECK(knowhere_index_build(empty, NULL, "{}") == KNOWHERE_INVALID_ARGUMENT);
        CHECK(knowhere_index_serialize(empty, &serialized) == KNOWHERE_INVALID_ARGUMENT && serialized == 0);
        CHECK(knowhere_index_build(empty, &mismatch, "{}") == KNOWHERE_INVALID_ARGUMENT);
        OK(knowhere_index_build(empty, &base, "{\"metric_type\":\"L2\"}"));
        CHECK(knowhere_index_search(empty, &mismatch, NULL, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
        CHECK(knowhere_index_search(empty, &unaligned, NULL, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
        OK(knowhere_index_search(empty, &query, NULL, &result, "{\"metric_type\":\"L2\"}"));
        CHECK(ids[0] == 123);
        OK(knowhere_index_destroy(empty));
        CHECK(knowhere_index_search(empty, &query, NULL, &result, "{}") == KNOWHERE_CLOSED);
        CHECK(strlen(knowhere_last_error()) > 0);
    }
    query.dimensions = 0;
    CHECK(knowhere_bruteforce(&base, &query, NULL, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
    /* A wrapped rows * row_bytes or count * sizeof product must fail instead of passing the capacity check. */
    query.dimensions = 2;
    query.rows = INT64_MAX;
    query.bytes = UINT64_MAX;
    query.data = data;
    CHECK(knowhere_bruteforce(&base, &query, NULL, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
    CHECK(strstr(knowhere_last_error(), "overflow") != NULL);
    query.rows = 1;
    query.bytes = 8;
    result.top_k = INT64_MAX;
    CHECK(knowhere_bruteforce(&base, &query, NULL, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
    CHECK(strstr(knowhere_last_error(), "overflow") != NULL);
    result.top_k = 2;
}

static void
test_failed_initialization(void) {
    float data[6] = {0, 0, 1, 0, 0, 2};
    knowhere_vectors base = {data, sizeof(data), 3, 2, KNOWHERE_FP32};
    knowhere_vectors query = {data, 8, 1, 2, KNOWHERE_FP32};
    int64_t ids[2];
    float distances[2];
    knowhere_search_result result = {ids, sizeof(ids), distances, sizeof(distances), 2};
    knowhere_index_handle index = 0;
    knowhere_binary_set_handle empty = 0, serialized = 55;
    const char* valid = "{\"metric_type\":\"L2\"}";
    OK(knowhere_index_create("FLAT", KNOWHERE_FP32, knowhere_index_version_current(), &index));
    /* The engine receives this invalid metric and fails initialization. */
    CHECK(knowhere_index_build(index, &base, "{\"metric_type\":\"invalid\"}") != KNOWHERE_SUCCESS);
    CHECK(knowhere_index_build(index, &base, valid) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_index_search(index, &query, NULL, &result, valid) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_index_serialize(index, &serialized) == KNOWHERE_INVALID_ARGUMENT && serialized == 0);
    OK(knowhere_index_destroy(index));
    OK(knowhere_index_destroy(index));

    OK(knowhere_binary_set_create(&empty));
    OK(knowhere_index_create("FLAT", KNOWHERE_FP32, knowhere_index_version_current(), &index));
    /* An empty BinarySet is rejected by the real Flat deserializer. */
    CHECK(knowhere_index_deserialize(index, empty, valid) != KNOWHERE_SUCCESS);
    OK(knowhere_binary_set_destroy(empty));
    CHECK(knowhere_index_build(index, &base, valid) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_index_search(index, &query, NULL, &result, valid) == KNOWHERE_INVALID_ARGUMENT);
    OK(knowhere_index_destroy(index));
    OK(knowhere_index_destroy(index));

    /* A fresh resource still builds after either failure and retains copied input. */
    OK(knowhere_index_create("FLAT", KNOWHERE_FP32, knowhere_index_version_current(), &index));
    OK(knowhere_index_build(index, &base, valid));
    data[2] = 20;
    data[5] = 20;
    OK(knowhere_index_search(index, &query, NULL, &result, valid));
    check_hits(&result, 0);
    OK(knowhere_index_destroy(index));
}

static void
test_thread_pools(void) {
    float data[6] = {0, 0, 1, 0, 0, 2};
    float queries[4] = {0, 0, 0, 2};
    knowhere_vectors base = {data, sizeof(data), 3, 2, KNOWHERE_FP32};
    knowhere_vectors query = {queries, sizeof(queries), 2, 2, KNOWHERE_FP32};
    int64_t ids[4] = {-1, -1, -1, -1};
    float distances[4] = {-1, -1, -1, -1};
    knowhere_search_result result = {ids, sizeof(ids), distances, sizeof(distances), 2};
    int64_t size = -1;
    /* Nothing has searched or built yet, so the lazily created pools do not exist. */
    OK(knowhere_search_thread_pool_size(&size));
    CHECK(size == 0);
    OK(knowhere_build_thread_pool_size(&size));
    CHECK(size == 0);
    CHECK(knowhere_search_thread_pool_size(NULL) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_search_thread_pool_resize(0) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_search_thread_pool_resize(-1) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_search_thread_pool_resize((int64_t)INT32_MAX + 1) == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_build_thread_pool_resize(0) == KNOWHERE_INVALID_ARGUMENT);
    OK(knowhere_search_thread_pool_resize(3));
    OK(knowhere_search_thread_pool_size(&size));
    CHECK(size == 3);
    /* Two queries become two tasks on the sized pool and still return exact results. */
    OK(knowhere_bruteforce(&base, &query, NULL, &result, "{\"metric_type\":\"L2\"}"));
    CHECK(ids[0] == 0 && ids[1] == 1 && ids[2] == 2 && ids[3] == 0);
    CHECK(distances[0] == 0 && distances[1] == 1 && distances[2] == 0 && distances[3] == 4);
    /* Resizing a live pool takes effect immediately. */
    OK(knowhere_search_thread_pool_resize(5));
    OK(knowhere_search_thread_pool_size(&size));
    CHECK(size == 5);
    OK(knowhere_build_thread_pool_resize(2));
    OK(knowhere_build_thread_pool_size(&size));
    CHECK(size == 2);
}

/* Deterministic values in [-1, 1): a linear congruential sequence keeps the
 * test free of rand() state while making ties improbable.
 */
static void
fill(float* values, int64_t count, uint32_t seed) {
    uint32_t state = seed;
    for (int64_t i = 0; i < count; i++) {
        state = state * 1664525u + 1013904223u;
        values[i] = (float)(state >> 8) / (float)(1u << 23) - 1.0f;
    }
}

/* knowhere_bruteforce_batched returns, for every query, the same ids as
 * knowhere_bruteforce with the same distances; the order of equal distances
 * may differ, so ids are compared as sets.
 */
static void
check_batched(const char* metric, int64_t nq, int64_t nb, int64_t dim, int64_t k) {
    float* base_data = malloc((size_t)(nb * dim) * sizeof(float));
    float* query_data = malloc((size_t)(nq * dim) * sizeof(float));
    int64_t* ids = malloc((size_t)(nq * k) * sizeof(int64_t));
    int64_t* batched_ids = malloc((size_t)(nq * k) * sizeof(int64_t));
    float* distances = malloc((size_t)(nq * k) * sizeof(float));
    float* batched_distances = malloc((size_t)(nq * k) * sizeof(float));
    CHECK(base_data && query_data && ids && batched_ids && distances && batched_distances);
    fill(base_data, nb * dim, 7u);
    fill(query_data, nq * dim, 11u);
    knowhere_vectors base = {base_data, (uint64_t)(nb * dim) * sizeof(float), nb, dim, KNOWHERE_FP32};
    knowhere_vectors query = {query_data, (uint64_t)(nq * dim) * sizeof(float), nq, dim, KNOWHERE_FP32};
    knowhere_search_result result = {ids, (uint64_t)(nq * k) * sizeof(int64_t), distances,
                                     (uint64_t)(nq * k) * sizeof(float), k};
    knowhere_search_result batched = {batched_ids, (uint64_t)(nq * k) * sizeof(int64_t), batched_distances,
                                      (uint64_t)(nq * k) * sizeof(float), k};
    char parameters[64];
    snprintf(parameters, sizeof(parameters), "{\"metric_type\":\"%s\"}", metric);
    OK(knowhere_bruteforce(&base, &query, NULL, &result, parameters));
    OK(knowhere_bruteforce_batched(&base, &query, &batched, parameters));
    for (int64_t q = 0; q < nq; q++) {
        for (int64_t i = 0; i < k; i++) {
            int found = 0;
            for (int64_t j = 0; j < k; j++) {
                if (batched_ids[q * k + j] == ids[q * k + i]) {
                    float expected = distances[q * k + i];
                    float actual = batched_distances[q * k + j];
                    float tolerance = 1e-3f * (expected < 0 ? -expected : expected) + 1e-3f;
                    CHECK(actual - expected <= tolerance && expected - actual <= tolerance);
                    found = 1;
                }
            }
            CHECK(found);
        }
    }
    free(batched_distances);
    free(distances);
    free(batched_ids);
    free(ids);
    free(query_data);
    free(base_data);
}

static void
test_bruteforce_batched(void) {
    /* Above and below faiss's BLAS threshold, and the other two metrics; the
     * queries are not normalized, which COSINE must do itself.
     */
    check_batched("L2", 64, 300, 24, 5);
    check_batched("L2", 3, 300, 24, 5);
    check_batched("IP", 40, 200, 24, 4);
    check_batched("COSINE", 40, 200, 24, 4);

    float data[4] = {0, 0, 3, 4};
    float origin[2] = {0, 0};
    knowhere_vectors base = {data, sizeof(data), 2, 2, KNOWHERE_FP32};
    knowhere_vectors query = {origin, sizeof(origin), 1, 2, KNOWHERE_FP32};
    int64_t ids[4] = {-9, -9, -9, -9};
    float distances[4] = {-9, -9, -9, -9};
    knowhere_search_result result = {ids, sizeof(ids), distances, sizeof(distances), 4};
    /* Fewer base rows than top_k leave the tail at -1. */
    OK(knowhere_bruteforce_batched(&base, &query, &result, "{\"metric_type\":\"L2\"}"));
    CHECK(ids[0] == 0 && ids[1] == 1 && ids[2] == -1 && ids[3] == -1);
    CHECK(distances[0] == 0.0f && distances[1] == 25.0f);
    /* Zero queries validate and leave the output as it was. */
    query.rows = 0;
    ids[0] = -9;
    OK(knowhere_bruteforce_batched(&base, &query, &result, "{\"metric_type\":\"L2\"}"));
    CHECK(ids[0] == -9);
    query.rows = 1;
    CHECK(knowhere_bruteforce_batched(&base, &query, &result, "{\"metric_type\":\"HAMMING\"}") ==
          KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_bruteforce_batched(&base, &query, &result, "{}") == KNOWHERE_INVALID_ARGUMENT);
    CHECK(knowhere_bruteforce_batched(&base, &query, &result, "{\"metric_type\":\"L2\",\"k\":1}") ==
          KNOWHERE_INVALID_ARGUMENT);
    uint16_t half[4] = {0, 0, 0x4200, 0x4400};
    knowhere_vectors half_base = {half, sizeof(half), 2, 2, KNOWHERE_FP16};
    knowhere_vectors half_query = {half, 4, 1, 2, KNOWHERE_FP16};
    CHECK(knowhere_bruteforce_batched(&half_base, &half_query, &result, "{\"metric_type\":\"L2\"}") ==
          KNOWHERE_INVALID_ARGUMENT);
}

int
main(void) {
    float fp32[] = {0, 0, 1, 0, 0, 2};
    uint16_t fp16[] = {0, 0, 0x3c00, 0, 0, 0x4000};
    uint16_t bf16[] = {0, 0, 0x3f80, 0, 0, 0x4000};
    int8_t int8[] = {0, 0, 1, 0, 0, 2};
    uint8_t binary[] = {0, 1, 3};
    CHECK(knowhere_c_abi_version() == 1);
    CHECK(knowhere_index_version_minimum() <= knowhere_index_version_current());
    CHECK(knowhere_index_version_current() <= knowhere_index_version_maximum());
    test_thread_pools();
    test_binary_set();
    test_invalid();
    test_failed_initialization();
    test_bruteforce_batched();
    test_typed(KNOWHERE_FP32, fp32, sizeof(fp32), fp32, 8, 2, "FLAT", "L2", 0);
    test_typed(KNOWHERE_FP16, fp16, sizeof(fp16), fp16, 4, 2, "FLAT", "L2", 0);
    test_typed(KNOWHERE_BF16, bf16, sizeof(bf16), bf16, 4, 2, "FLAT", "L2", 0);
    test_typed(KNOWHERE_INT8, int8, sizeof(int8), int8, 2, 2, "FLAT", "L2", 0);
    test_typed(KNOWHERE_BIN1, binary, sizeof(binary), binary, 1, 8, "BIN_FLAT", "HAMMING", 1);
    test_typed(KNOWHERE_FP32, fp32, sizeof(fp32), fp32, 8, 2, "HNSW", "L2", 0);
    test_typed(KNOWHERE_FP32, fp32, sizeof(fp32), fp32, 8, 2, "IVF_FLAT", "L2", 0);
    puts("Knowhere C API tests passed");
    return 0;
}
