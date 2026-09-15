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
#define CHECK(expression)                                                                               \
    do {                                                                                                \
        if (!(expression)) {                                                                            \
            fprintf(stderr, "%s:%d: %s; %s\n", __FILE__, __LINE__, #expression, knowhere_last_error()); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)
#define OK(expression) CHECK((expression) == KNOWHERE_SUCCESS)

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
    float query_data[DIMENSIONS] = {0};
    uint8_t excluded_data[(ROWS + 7) / 8] = {0};
    int64_t ids[2] = {-1, -1};
    float distances[2] = {-1, -1};
    knowhere_search_result result = {ids, sizeof(ids), distances, sizeof(distances), 2};
    knowhere_vectors query = {query_data, sizeof(query_data), 1, DIMENSIONS, KNOWHERE_FP32};
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

    /* The first three rows are (0,0), (1,0), (0,2), padded with zeroes.
     * All other rows are far from the query and provide enough distinct
     * training samples for DiskANN's 256-centroid product quantizer.
     */
    data[DIMENSIONS] = 1.0f;
    data[2 * DIMENSIONS + 1] = 2.0f;
    for (row = 3; row < ROWS; ++row) {
        for (column = 0; column < DIMENSIONS; ++column) {
            random_state = random_state * UINT32_C(1664525) + UINT32_C(1013904223);
            data[row * DIMENSIONS + column] = 10.0f + (float)(random_state >> 8) / 16777216.0f;
        }
    }
    write_raw_data(raw_path, data);
    written = snprintf(build_parameters, sizeof(build_parameters),
                       "{\"dim\":%d,\"metric_type\":\"L2\",\"index_prefix\":\"%s\",\"data_path\":\"%s\","
                       "\"max_degree\":32,\"search_list_size\":128,\"pq_code_budget_gb\":%.17g,"
                       "\"build_dram_budget_gb\":0.25,\"search_cache_budget_gb\":0,\"disk_pq_dims\":0,"
                       "\"shuffle_build\":false}",
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
    {
        knowhere_vectors base = {data, sizeof(float) * (uint64_t)ROWS * DIMENSIONS, ROWS, DIMENSIONS, KNOWHERE_FP32};
        CHECK(knowhere_index_build(builder, &base, build_parameters) == KNOWHERE_INVALID_ARGUMENT);
    }
    free(data);
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

    OK(knowhere_index_search(loaded, &query, NULL, &result, search_parameters));
    CHECK(ids[0] == 0 && ids[1] == 1);
    CHECK(distances[0] == 0.0f && distances[1] == 1.0f);
    excluded_data[0] = 1;
    OK(knowhere_index_search(loaded, &query, &excluded, &result, search_parameters));
    CHECK(ids[0] == 1 && ids[1] == 2);
    CHECK(distances[0] == 1.0f && distances[1] == 4.0f);

    OK(knowhere_index_destroy(loaded));
    CHECK(knowhere_index_search(loaded, &query, NULL, &result, search_parameters) == KNOWHERE_CLOSED);
    CHECK(nftw(directory, remove_owned_path, 16, FTW_DEPTH | FTW_PHYS) == 0);
    puts("Knowhere DiskANN C API local-file round trip passed");
    return 0;
}
