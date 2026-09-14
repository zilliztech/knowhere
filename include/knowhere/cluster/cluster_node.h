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

    // cluster assign, return id_mapping and optional distance.
    // Legacy callers may read id_mapping from tensor; callers that need
    // distance should read ids and distance.
    virtual expected<DataSetPtr>
    Assign(const DataSet& dataset) = 0;

    virtual expected<DataSetPtr>
    Assign(const DataSet& dataset, const Config& cfg) {
        (void)cfg;
        return Assign(dataset);
    }

    // build a compaction plan from assignment result.
    virtual expected<DataSetPtr>
    BuildCompactionPlan(const DataSet& assignment, const Config& cfg) {
        (void)assignment;
        (void)cfg;
        return expected<DataSetPtr>::Err(Status::not_implemented, "BuildCompactionPlan is not implemented");
    }

    // return centroids, must be called after trained
    // (rows, dim, centroid_vector_list)
    virtual expected<DataSetPtr>
    GetCentroids() const = 0;

    // Inject externally-computed centroids, bypassing Train. After success,
    // Assign and GetCentroids behave as they do after Train.
    virtual Status
    SetCentroids(const DataSet& centroids) {
        (void)centroids;
        return Status::not_implemented;
    }

    virtual std::unique_ptr<Config>
    CreateConfig() const = 0;

    virtual std::string
    Type() const = 0;

    virtual ~ClusterNode() {
    }
};
}  // namespace knowhere

#endif /* CLUSTER_NODE_H */
