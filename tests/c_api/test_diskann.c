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
#define _XOPEN_SOURCE 700

#include <ftw.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "knowhere/c_api.h"

/* This test requires a real Knowhere build with DISKANN enabled. Missing
 * support is a failure, not a skipped or substituted algorithm test.
 * Parameters and the raw file format follow tests/ut/test_diskann.cc.
 */
#define ROWS 1000
#define DIMENSIONS 32
#define QUERIES 20
#define TOP_K 5
#define EXCLUDED_ROWS 100
#define CHECK(expression)                                                                               \
    do {                                                                                                \
        if (!(expression)) {                                                                            \
            fprintf(stderr, "%s:%d: %s; %s\n", __FILE__, __LINE__, #expression, knowhere_last_error()); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)
#define OK(expression) CHECK((expression) == KNOWHERE_SUCCESS)

static void
check_search_results(const float* data, const float* queries, const knowhere_search_result* result,
                     const knowhere_search_result* truth, const knowhere_bitset* excluded) {
    size_t query, rank, other, column;
    size_t hits = 0;
    for (query = 0; query < QUERIES; ++query) {
        for (rank = 0; rank < TOP_K; ++rank) {
            const size_t offset = query * TOP_K + rank;
            const int64_t id = result->ids[offset];
            double expected_distance = 0.0;
            CHECK(id >= 0 && id < ROWS);
            CHECK(excluded == NULL || (excluded->data[id / 8] & (1u << (id % 8))) == 0);
            for (other = 0; other < rank; ++other) {
                CHECK(result->ids[query * TOP_K + other] != id);
            }
            for (column = 0; column < DIMENSIONS; ++column) {
                const double delta = (double)queries[query * DIMENSIONS + column] - data[id * DIMENSIONS + column];
                expected_distance += delta * delta;
            }
            CHECK(result->distances[offset] >= expected_distance - 1e-4);
            CHECK(result->distances[offset] <= expected_distance + 1e-4);
            if (rank > 0) {
                CHECK(result->distances[offset - 1] <= result->distances[offset]);
            }
            for (other = 0; other < TOP_K; ++other) {
                if (truth->ids[query * TOP_K + other] == id) {
                    ++hits;
                    break;
                }
            }
        }
    }
    fprintf(stderr, "DiskANN %s recall: %zu/%d\n", excluded == NULL ? "unfiltered" : "filtered", hits, QUERIES * TOP_K);
    CHECK(hits * 100 >= 95 * QUERIES * TOP_K);
}

static void
write_raw_data(const char* path, const float* data) {
    const uint32_t rows = ROWS;
    const uint32_t dimensions = DIMENSIONS;
    FILE* file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(&rows, sizeof(rows), 1, file) == 1);
    CHECK(fwrite(&dimensions, sizeof(dimensions), 1, file) == 1);
    CHECK(fwrite(data, sizeof(float), (size_t)ROWS * DIMENSIONS, file) == (size_t)ROWS * DIMENSIONS);
    CHECK(fclose(file) == 0);
}

static int
remove_owned_path(const char* path, const struct stat* status, int type, struct FTW* traversal) {
    (void)status;
    (void)type;
    (void)traversal;
    return remove(path);
}

int
main(void) {
    char directory[] = "/tmp/knowhere-c-diskann-XXXXXX";
    char raw_path[256];
    char index_prefix[256];
    char build_parameters[1536];
    char load_parameters[768];
    char missing_parameters[768];
    char search_parameters[768];
    float* data = (float*)calloc((size_t)ROWS * DIMENSIONS, sizeof(float));
    float query_data[QUERIES * DIMENSIONS] = {0};
    uint8_t excluded_data[(ROWS + 7) / 8] = {0};
    int64_t ids[QUERIES * TOP_K] = {0};
    float distances[QUERIES * TOP_K] = {0};
    int64_t truth_ids[QUERIES * TOP_K] = {0};
    float truth_distances[QUERIES * TOP_K] = {0};
    knowhere_search_result result = {ids, sizeof(ids), distances, sizeof(distances), TOP_K};
    knowhere_search_result truth = {truth_ids, sizeof(truth_ids), truth_distances, sizeof(truth_distances), TOP_K};
    knowhere_vectors base = {data, sizeof(float) * (uint64_t)ROWS * DIMENSIONS, ROWS, DIMENSIONS, KNOWHERE_FP32};
    knowhere_vectors query = {query_data, sizeof(query_data), QUERIES, DIMENSIONS, KNOWHERE_FP32};
    knowhere_bitset excluded = {excluded_data, sizeof(excluded_data), ROWS};
    knowhere_index_handle builder = 0, loaded = 0, missing = 0;
    knowhere_binary_set_handle serialized = 0;
    int64_t actual_rows = 0, actual_dimensions = 0;
    uint32_t random_state = 42;
    size_t row, column;
    int written;
    const double pq_budget = (double)sizeof(float) * DIMENSIONS * ROWS * 0.125 / (1024.0 * 1024.0 * 1024.0);

    CHECK(data != NULL);
    CHECK(mkdtemp(directory) != NULL);
    fprintf(stderr, "DiskANN C test files: %s\n", directory);
    CHECK(snprintf(raw_path, sizeof(raw_path), "%s/raw.bin", directory) < (int)sizeof(raw_path));
    CHECK(snprintf(index_prefix, sizeof(index_prefix), "%s/index", directory) < (int)sizeof(index_prefix));

    /* Use a uniform dataset and verify ANN recall against exact search, as
     * the upstream DiskANN tests do. Isolated near-zero points in a distant
     * cluster can be unreachable in the approximate graph. A fixed input
     * seed does not fix DiskANN's randomized graph construction.
     */
    for (row = 0; row < ROWS; ++row) {
        for (column = 0; column < DIMENSIONS; ++column) {
            random_state = random_state * UINT32_C(1664525) + UINT32_C(1013904223);
            data[row * DIMENSIONS + column] = (float)(random_state >> 8) / 16777216.0f;
        }
    }
    for (row = 0; row < QUERIES; ++row) {
        for (column = 0; column < DIMENSIONS; ++column) {
            query_data[row * DIMENSIONS + column] = data[row * 17 * DIMENSIONS + column] + 0.001f;
        }
    }
    for (row = 0; row < EXCLUDED_ROWS; ++row) {
        excluded_data[row / 8] |= (uint8_t)(1u << (row % 8));
    }
    write_raw_data(raw_path, data);
    written = snprintf(build_parameters, sizeof(build_parameters),
                       "{\"dim\":%d,\"metric_type\":\"L2\",\"index_prefix\":\"%s\",\"data_path\":\"%s\","
                       "\"max_degree\":32,\"search_list_size\":128,\"pq_code_budget_gb\":%.17g,"
                       "\"build_dram_budget_gb\":0.25,\"search_cache_budget_gb\":0,\"disk_pq_dims\":0}",
                       DIMENSIONS, index_prefix, raw_path, pq_budget);
    CHECK(written >= 0 && written < (int)sizeof(build_parameters));
    written = snprintf(load_parameters, sizeof(load_parameters),
                       "{\"dim\":%d,\"metric_type\":\"L2\",\"index_prefix\":\"%s\","
                       "\"search_cache_budget_gb\":0,\"warm_up\":false}",
                       DIMENSIONS, index_prefix);
    CHECK(written >= 0 && written < (int)sizeof(load_parameters));
    written = snprintf(search_parameters, sizeof(search_parameters),
                       "{\"dim\":%d,\"metric_type\":\"L2\",\"index_prefix\":\"%s\","
                       "\"search_list_size\":%d,\"beamwidth\":4}",
                       DIMENSIONS, index_prefix, ROWS);
    CHECK(written >= 0 && written < (int)sizeof(search_parameters));
    written = snprintf(missing_parameters, sizeof(missing_parameters),
                       "{\"dim\":%d,\"metric_type\":\"L2\",\"index_prefix\":\"%s/missing\","
                       "\"search_cache_budget_gb\":0}",
                       DIMENSIONS, directory);
    CHECK(written >= 0 && written < (int)sizeof(missing_parameters));

    OK(knowhere_index_create("DISKANN", KNOWHERE_FP32, knowhere_index_version_current(), &builder));
    CHECK(knowhere_index_build(builder, &base, build_parameters) == KNOWHERE_INVALID_ARGUMENT);
    OK(knowhere_index_build(builder, NULL, build_parameters));
    OK(knowhere_index_info(builder, &actual_rows, &actual_dimensions));
    CHECK(actual_rows == ROWS && actual_dimensions == DIMENSIONS);
    OK(knowhere_index_serialize(builder, &serialized));
    OK(knowhere_index_destroy(builder));
    OK(knowhere_index_destroy(builder));
    CHECK(unlink(raw_path) == 0);

    /* DiskANN's BinarySet does not contain its disk index files. Keep the
     * files at index_prefix and load them through the injected LocalFileManager.
     * A missing index_prefix must fail rather than yielding an empty index.
     */
    OK(knowhere_index_create("DISKANN", KNOWHERE_FP32, knowhere_index_version_current(), &missing));
    CHECK(knowhere_index_deserialize(missing, serialized, missing_parameters) != KNOWHERE_SUCCESS);
    OK(knowhere_index_destroy(missing));
    OK(knowhere_index_create("DISKANN", KNOWHERE_FP32, knowhere_index_version_current(), &loaded));
    OK(knowhere_index_deserialize(loaded, serialized, load_parameters));
    OK(knowhere_binary_set_destroy(serialized));
    OK(knowhere_index_info(loaded, &actual_rows, &actual_dimensions));
    CHECK(actual_rows == ROWS && actual_dimensions == DIMENSIONS);

    OK(knowhere_bruteforce(&base, &query, NULL, &truth, "{\"metric_type\":\"L2\"}"));
    OK(knowhere_index_search(loaded, &query, NULL, &result, search_parameters));
    check_search_results(data, query_data, &result, &truth, NULL);
    OK(knowhere_bruteforce(&base, &query, &excluded, &truth, "{\"metric_type\":\"L2\"}"));
    OK(knowhere_index_search(loaded, &query, &excluded, &result, search_parameters));
    check_search_results(data, query_data, &result, &truth, &excluded);

    OK(knowhere_index_destroy(loaded));
    CHECK(knowhere_index_search(loaded, &query, NULL, &result, search_parameters) == KNOWHERE_CLOSED);
    free(data);
    CHECK(nftw(directory, remove_owned_path, 16, FTW_DEPTH | FTW_PHYS) == 0);
    puts("Knowhere DiskANN C API local-file round trip passed");
    return 0;
}
