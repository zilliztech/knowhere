// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>

#include "knowhere/expected.h"
#include "knowhere/log.h"

namespace knowhere {

// Validate the raw artifact before using its shape to resolve parameter budgets.
// The backend's get_bin_metadata does not check stream failures.
inline Status
ReadDiskANNRawMetadata(const std::string& path, size_t element_size, size_t& count, size_t& dim) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto file_size = input.tellg();
    int32_t shape[2] = {0, 0};
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(shape), sizeof(shape)) || shape[0] <= 0 || shape[1] <= 0) {
        LOG_KNOWHERE_ERROR_ << "Cannot read valid DiskANN raw vector metadata from " << path;
        return Status::disk_file_error;
    }
    const uint64_t rows = shape[0];
    const uint64_t dimension = shape[1];
    if (element_size == 0 || rows > (std::numeric_limits<uint64_t>::max() - sizeof(shape)) / element_size / dimension ||
        file_size < 0 || static_cast<uint64_t>(file_size) != sizeof(shape) + rows * dimension * element_size) {
        LOG_KNOWHERE_ERROR_ << "DiskANN raw vector file size disagrees with its metadata: " << path;
        return Status::disk_file_error;
    }
    count = rows;
    dim = dimension;
    return Status::success;
}

}  // namespace knowhere
