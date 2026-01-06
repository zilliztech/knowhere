/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/utils/utils.h>

#include <cassert>
#include <cstdio>
#include <cstring>

#include <sys/types.h>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <sys/time.h>
#include <unistd.h>
#endif // !_MSC_VER

#include <cinttypes>

/**************************************************
 * Get some stats about the system
 **************************************************/



namespace faiss::cppcontrib::knowhere {

int64_t get_l3_size() {
    static int64_t l3_size = -1;
    constexpr int64_t KB = 1024;
    if (l3_size == -1) {
        FILE* file =
                fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
        int64_t result = 0;
        constexpr int64_t line_length = 128;
        char line[line_length];
        if (file) {
            char* ret = fgets(line, sizeof(line) - 1, file);

            sscanf(line, "%" SCNd64 "K", &result);
            l3_size = result * KB;

            fclose(file);
        } else {
            l3_size = 12 * KB * KB; // 12M
        }
    }
    return l3_size;
}

}


