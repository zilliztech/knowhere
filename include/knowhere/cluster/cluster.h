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

#ifndef CLUSTER_H
#define CLUSTER_H

#include "knowhere/binaryset.h"
#include "knowhere/cluster/cluster_node.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"

namespace knowhere {
template <typename T1>
class Cluster {
 public:
    template <typename T2>
    friend class Cluster;

    Cluster() : node(nullptr) {
    }

    template <typename... Args>
    static Cluster<T1>
    Create(Args&&... args) {
        return Cluster(new (std::nothrow) T1(std::forward<Args>(args)...));
    }

    Cluster(const Cluster<T1>& cluster) {
        if (cluster.node == nullptr) {
            node = nullptr;
            return;
        }
        cluster.node->IncRef();
        node = cluster.node;
    }

    Cluster(Cluster<T1>&& cluster) {
        if (cluster.node == nullptr) {
            node = nullptr;
            return;
        }
        node = cluster.node;
        cluster.node = nullptr;
    }

    template <typename T2>
    Cluster(const Cluster<T2>& cluster) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (cluster.node == nullptr) {
            node = nullptr;
            return;
        }
        cluster.node->IncRef();
        node = cluster.node;
    }

    template <typename T2>
    Cluster(Cluster<T2>&& cluster) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (cluster.node == nullptr) {
            node = nullptr;
            return;
        }
        node = cluster.node;
        cluster.node = nullptr;
    }

    template <typename T2>
    Cluster<T1>&
    operator=(const Cluster<T2>& cluster) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        if (cluster.node == nullptr) {
            node = nullptr;
            return *this;
        }
        node = cluster.node;
        node->IncRef();
        return *this;
    }

    template <typename T2>
    Cluster<T1>&
    operator=(Cluster<T2>&& cluster) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        node = cluster.node;
        cluster.node = nullptr;
        return *this;
    }

    T1*
    Node() {
        return node;
    }

    const T1*
    Node() const {
        return node;
    }

    expected<DataSetPtr>
    Train(const DataSet& dataset, const Json& json);

    expected<DataSetPtr>
    Assign(const DataSet& dataset);

    expected<DataSetPtr>
    GetCentroids() const;

    std::string
    Type() const;

    ~Cluster() {
        if (node == nullptr)
            return;
        node->DecRef();
        if (!node->Ref())
            delete node;
    }

 private:
    Cluster(T1* node) : node(node) {
        static_assert(std::is_base_of<ClusterNode, T1>::value);
    }

    T1* node;
};

}  // namespace knowhere

#endif /* CLUSTER_H */
