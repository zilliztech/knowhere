// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

// Note: utils.h must be included BEFORE this header in test files
// This avoids ODR violations since utils.h lacks include guards
// Forward declare the function we need from utils.h
float
GetKNNRecall(const knowhere::DataSet& ground_truth, const knowhere::DataSet& result);

namespace knowhere::test {

// Default thresholds for recall validation
constexpr float kDefaultKnnRecallThreshold = 0.6f;
constexpr float kHighRecallThreshold = 0.8f;
constexpr float kBruteForceRecallThreshold = 0.95f;

/**
 * @brief Helper class for index testing that reduces boilerplate code
 *
 * Usage:
 *   IndexTestHelper<fp32> helper(index_name, version);
 *   helper.SetDataset(train_ds, query_ds);
 *   helper.Build(config);
 *   auto recall = helper.SearchAndGetRecall(config);
 *   REQUIRE(recall > kDefaultKnnRecallThreshold);
 */
template <typename DataType>
class IndexTestHelper {
 public:
    IndexTestHelper(const std::string& index_name, int32_t version)
        : index_name_(index_name), version_(version), index_(std::nullopt) {
    }

    IndexTestHelper(const std::string& index_name)
        : IndexTestHelper(index_name, Version::GetCurrentVersion().VersionNumber()) {
    }

    // Create index instance
    Status
    Create() {
        auto idx_result = IndexFactory::Instance().Create<DataType>(index_name_, version_);
        if (!idx_result.has_value()) {
            return idx_result.error();
        }
        index_ = std::move(idx_result.value());
        return Status::success;
    }

    // Set training and query datasets
    void
    SetDataset(const DataSetPtr& train_ds, const DataSetPtr& query_ds) {
        train_ds_ = train_ds;
        query_ds_ = query_ds;
    }

    // Build index with given config
    Status
    Build(const Json& config) {
        if (!index_.has_value()) {
            auto status = Create();
            if (status != Status::success) {
                return status;
            }
        }
        return index_->Build(train_ds_, config);
    }

    // Build and verify basic properties
    bool
    BuildAndVerify(const Json& config, int64_t expected_count) {
        auto status = Build(config);
        if (status != Status::success) {
            return false;
        }
        return index_->Size() > 0 && index_->Count() == expected_count;
    }

    // Search and return results
    expected<DataSetPtr>
    Search(const Json& config, const BitsetView& bitset = nullptr) {
        return index_->Search(query_ds_, config, bitset);
    }

    // Range search
    expected<DataSetPtr>
    RangeSearch(const Json& config, const BitsetView& bitset = nullptr) {
        return index_->RangeSearch(query_ds_, config, bitset);
    }

    // Get ground truth using brute force search
    expected<DataSetPtr>
    GetGroundTruth(const Json& config, const BitsetView& bitset = nullptr) {
        return BruteForce::Search<DataType>(train_ds_, query_ds_, config, bitset);
    }

    // Search and compute recall against ground truth
    float
    SearchAndGetRecall(const Json& config, const BitsetView& bitset = nullptr) {
        auto gt = GetGroundTruth(config, bitset);
        REQUIRE(gt.has_value());

        auto results = Search(config, bitset);
        REQUIRE(results.has_value());

        return GetKNNRecall(*gt.value(), *results.value());
    }

    // Serialize index to BinarySet
    Status
    Serialize(BinarySet& bs) {
        return index_->Serialize(bs);
    }

    // Deserialize index from BinarySet
    Status
    Deserialize(const BinarySet& bs, const Json& config = {}) {
        return index_->Deserialize(bs, config);
    }

    // Serialize and deserialize (round-trip test)
    bool
    SerializeDeserializeRoundTrip(const Json& config = {}) {
        BinarySet bs;
        auto status = Serialize(bs);
        if (status != Status::success) {
            return false;
        }

        // Create new index for deserialization
        auto new_idx = IndexFactory::Instance().Create<DataType>(index_name_, version_);
        if (!new_idx.has_value()) {
            return false;
        }

        status = new_idx.value().Deserialize(bs, config);
        if (status != Status::success) {
            return false;
        }

        index_ = std::move(new_idx.value());
        return true;
    }

    // Get index type
    std::string
    Type() const {
        return index_.has_value() ? index_->Type() : "";
    }

    // Get index size
    int64_t
    Size() const {
        return index_.has_value() ? index_->Size() : 0;
    }

    // Get index count
    int64_t
    Count() const {
        return index_.has_value() ? index_->Count() : 0;
    }

    // Check if index has raw data
    bool
    HasRawData(const std::string& metric_type) const {
        return index_.has_value() ? index_->HasRawData(metric_type) : false;
    }

    // Get underlying index reference
    Index<IndexNode>&
    GetIndex() {
        return index_.value();
    }

    const Index<IndexNode>&
    GetIndex() const {
        return index_.value();
    }

 private:
    std::string index_name_;
    int32_t version_;
    std::optional<Index<IndexNode>> index_;
    DataSetPtr train_ds_;
    DataSetPtr query_ds_;
};

/**
 * @brief Run a complete index test: create -> build -> serialize -> deserialize -> search -> verify recall
 *
 * @return true if all steps succeed and recall >= threshold
 */
template <typename DataType>
bool
RunCompleteIndexTest(const std::string& index_name, const DataSetPtr& train_ds, const DataSetPtr& query_ds,
                     const Json& config, float recall_threshold = kDefaultKnnRecallThreshold,
                     int32_t version = Version::GetCurrentVersion().VersionNumber()) {
    IndexTestHelper<DataType> helper(index_name, version);
    helper.SetDataset(train_ds, query_ds);

    // Build
    if (!helper.BuildAndVerify(config, train_ds->GetRows())) {
        return false;
    }

    // Verify type
    if (helper.Type() != index_name) {
        return false;
    }

    // Serialize/Deserialize round-trip
    if (!helper.SerializeDeserializeRoundTrip(config)) {
        return false;
    }

    // Search and verify recall
    float recall = helper.SearchAndGetRecall(config);
    return recall >= recall_threshold;
}

/**
 * @brief Verify search results are correctly filtered by bitset
 */
inline bool
VerifyBitsetFiltering(const DataSet& results, const BitsetView& bitset, int64_t topk) {
    auto ids = results.GetIds();
    auto nq = results.GetRows();

    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < topk; ++j) {
            auto id = ids[i * topk + j];
            if (id >= 0 && bitset.test(id)) {
                return false;  // Found a filtered-out ID in results
            }
        }
    }
    return true;
}

/**
 * @brief Verify distances are in expected order (ascending for L2, descending for IP/COSINE)
 */
inline bool
VerifyDistanceOrder(const DataSet& results, bool ascending = true) {
    auto nq = results.GetRows();
    auto k = results.GetDim();
    auto distances = results.GetDistance();
    auto ids = results.GetIds();

    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < k - 1; ++j) {
            if (ids[i * k + j] == -1 || ids[i * k + j + 1] == -1) {
                break;
            }
            float d1 = distances[i * k + j];
            float d2 = distances[i * k + j + 1];
            if (ascending && d1 > d2) {
                return false;
            }
            if (!ascending && d1 < d2) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Verify range search results are within specified bounds
 */
inline bool
VerifyRangeSearchBounds(const DataSet& results, float radius, float range_filter, bool is_ip_like = false) {
    auto lims = results.GetLims();
    auto distances = results.GetDistance();
    auto nq = results.GetRows();

    for (size_t i = 0; i < lims[nq]; ++i) {
        float d = distances[i];
        if (is_ip_like) {
            // For IP-like metrics: radius <= d <= range_filter
            if (d < radius || d > range_filter) {
                return false;
            }
        } else {
            // For L2-like metrics: range_filter <= d <= radius
            if (d < range_filter || d > radius) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace knowhere::test
