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

#ifndef INDEX_NODE_H
#define INDEX_NODE_H

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/object.h"

namespace knowhere {

class IndexNode : public Object {
 public:
    virtual Status
    Build(const DataSet& dataset, const Config& cfg) {
        RETURN_IF_ERROR(Train(dataset, cfg));
        return Add(dataset, cfg);
    }

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) = 0;

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) = 0;

    virtual expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;

    // not thread safe.
    class iterator {
     public:
        virtual std::pair<int64_t, float>
        Next() = 0;
        [[nodiscard]] virtual bool
        HasNext() const = 0;
        virtual ~iterator() {
        }
    };

    virtual expected<std::vector<std::shared_ptr<iterator>>>
    AnnIterator(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
        throw std::runtime_error("annIterator not supported for current index type");
    }

    virtual expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;

    virtual expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const = 0;

    virtual bool
    HasRawData(const std::string& metric_type) const = 0;

    virtual expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const = 0;

    virtual Status
    Serialize(BinarySet& binset) const = 0;

    virtual Status
    Deserialize(const BinarySet& binset, const Config& config) = 0;

    virtual Status
    DeserializeFromFile(const std::string& filename, const Config& config) = 0;

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const = 0;

    virtual int64_t
    Dim() const = 0;

    virtual int64_t
    Size() const = 0;

    virtual int64_t
    Count() const = 0;

    virtual std::string
    Type() const = 0;

    virtual ~IndexNode() {
    }
};

}  // namespace knowhere

#endif /* INDEX_NODE_H */
