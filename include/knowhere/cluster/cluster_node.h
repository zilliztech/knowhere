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

#ifndef CLUSTER_NODE_H
#define CLUSTER_NODE_H

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/object.h"
#include "knowhere/operands.h"

namespace knowhere {
class ClusterNode : public Object {
 public:
    // kmeans train, return id_mapping
    // (rows, uint32_t* id_mapping)
    virtual expected<DataSetPtr>
    Train(const DataSet& dataset, const Config& cfg) = 0;

    // cluster assign, return id_mapping
    // (rows, uint32_t* id_mapping)
    virtual expected<DataSetPtr>
    Assign(const DataSet& dataset) = 0;

    // return centroids, must be called after trained
    // (rows, dim, centroid_vector_list)
    virtual expected<DataSetPtr>
    GetCentroids() const = 0;

    virtual std::unique_ptr<Config>
    CreateConfig() const = 0;

    virtual std::string
    Type() const = 0;

    virtual ~ClusterNode() {
    }
};
}  // namespace knowhere

#endif /* CLUSTER_NODE_H */
