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
#include <memory>
#include <string>

#include "knowhere/binaryset.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/comp/task.h"
#else
class ThreadPool;
#endif

namespace milvus {
struct OpContext;
}  // namespace milvus

namespace knowhere {

class EmbListRawStorage {
 public:
    virtual ~EmbListRawStorage() = default;

    virtual int64_t
    Dim() const = 0;

    virtual int64_t
    Count() const = 0;

    virtual size_t
    CodeSize() const = 0;

    virtual Status
    Add(const DataSetPtr dataset) = 0;

    virtual Status
    ReconstructN(int64_t start, int64_t n, uint8_t* out) const = 0;

    virtual Status
    Serialize(BinarySet& binset) const = 0;

    virtual expected<DataSetPtr>
    CalcDistance(const DataSetPtr dataset, const int64_t* labels, size_t labels_len, const std::string& metric_type,
                 bool is_cosine, std::shared_ptr<ThreadPool> pool, milvus::OpContext* op_context) const = 0;
};

expected<std::shared_ptr<EmbListRawStorage>>
CreateEmbListRawStorageForBuild(const DataSetPtr dataset, const std::string& metric_type);

expected<std::shared_ptr<EmbListRawStorage>>
ReadEmbListRawStorageFromBinary(const BinaryPtr& raw_index_bin, const std::string& metric_type);

expected<std::shared_ptr<EmbListRawStorage>>
ReadEmbListRawStorageFromFile(const std::string& filename, int io_flags, const std::string& metric_type);

}  // namespace knowhere
