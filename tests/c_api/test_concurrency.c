/* Copyright (C) 2026 Zilliz. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "knowhere/c_api.h"

#define CHECK(x)                                 \
    do {                                         \
        if (!(x)) {                              \
            fprintf(stderr, "Failed: %s\n", #x); \
            abort();                             \
        }                                        \
    } while (0)

static void*
read_while_closing(void* argument) {
    knowhere_binary_set_handle handle = *(knowhere_binary_set_handle*)argument;
    for (int i = 0; i < 200; i++) {
        unsigned char data[16];
        int32_t code = knowhere_binary_set_read(handle, "data", 0, data, sizeof(data));
        CHECK(code == KNOWHERE_SUCCESS || code == KNOWHERE_CLOSED);
        if (code == KNOWHERE_SUCCESS) {
            CHECK(memcmp(data, "0123456789abcdef", sizeof(data)) == 0);
        } else {
            CHECK(strstr(knowhere_last_error(), "closed") != NULL);
        }
    }
    return NULL;
}

static void*
independent_errors(void* unused) {
    (void)unused;
    for (int i = 0; i < 1000; i++) {
        knowhere_binary_set_handle handle = 0;
        CHECK(knowhere_binary_set_create(&handle) == KNOWHERE_SUCCESS);
        CHECK(knowhere_binary_set_allocate(handle, "", 1) == KNOWHERE_INVALID_ARGUMENT);
        CHECK(strstr(knowhere_last_error(), "name") != NULL);
        CHECK(knowhere_binary_set_destroy(handle) == KNOWHERE_SUCCESS);
        CHECK(knowhere_last_error()[0] == '\0');
    }
    return NULL;
}

int
main(void) {
    pthread_t errors;
    CHECK(pthread_create(&errors, NULL, independent_errors, NULL) == 0);
    knowhere_binary_set_handle previous = 0;
    for (int iteration = 0; iteration < 100; iteration++) {
        knowhere_binary_set_handle handle = 0;
        pthread_t readers[4];
        CHECK(knowhere_binary_set_create(&handle) == KNOWHERE_SUCCESS);
        CHECK(handle != previous);
        CHECK(knowhere_binary_set_allocate(handle, "data", 16) == KNOWHERE_SUCCESS);
        CHECK(knowhere_binary_set_write(handle, "data", 0, "0123456789abcdef", 16) == KNOWHERE_SUCCESS);
        for (int i = 0; i < 4; i++) {
            CHECK(pthread_create(&readers[i], NULL, read_while_closing, &handle) == 0);
        }
        CHECK(knowhere_binary_set_destroy(handle) == KNOWHERE_SUCCESS);
        for (int i = 0; i < 4; i++) {
            CHECK(pthread_join(readers[i], NULL) == 0);
        }
        uint64_t count;
        CHECK(knowhere_binary_set_count(handle, &count) == KNOWHERE_CLOSED);
        CHECK(knowhere_binary_set_destroy(handle) == KNOWHERE_SUCCESS);
        previous = handle;
    }
    CHECK(pthread_join(errors, NULL) == 0);
    puts("Knowhere concurrent C API tests passed");
    return 0;
}
