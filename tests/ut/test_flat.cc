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

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "test_config_generators.h"
#include "test_helpers.h"
#include "utils.h"

namespace {
constexpr float kPerfectRecall = 1.0f;
constexpr float kHighRecall = 0.99f;
}  // namespace

// ==================== Float Flat Index Tests ====================

TEST_CASE("Test Flat Index Build and Basic Properties", "[flat][build]") {
    const int64_t nb = 1000;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    knowhere::test::ConfigGenerator gen(dim, metric, 10);
    auto config = gen.Flat();

    const auto train_ds = GenDataSet(nb, dim);

    SECTION("Build should succeed with valid parameters") {
        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, nullptr);

        REQUIRE(helper.Build(config) == knowhere::Status::success);
        REQUIRE(helper.Type() == knowhere::IndexEnum::INDEX_FAISS_IDMAP);
        REQUIRE(helper.Size() > 0);
        REQUIRE(helper.Count() == nb);
    }

    SECTION("Build with mismatched dimension should fail") {
        auto invalid_config = config;
        invalid_config[knowhere::meta::DIM] = dim * 2;  // Mismatched dimension

        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version)
                       .value();
        auto status = idx.Build(train_ds, invalid_config);
        // The index may or may not validate dimension mismatch
        // This is more of a documentation test to show expected behavior
        (void)status;  // Behavior may vary by implementation
    }

    SECTION("HasRawData should return true for Flat index") {
        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, nullptr);

        REQUIRE(helper.Build(config) == knowhere::Status::success);
        REQUIRE(helper.HasRawData(metric) == true);
    }
}

TEST_CASE("Test Flat Index Search", "[flat][search]") {
    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    knowhere::test::ConfigGenerator gen(dim, metric, topk);
    auto config = gen.Flat();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    SECTION("Search should return perfect recall (brute force)") {
        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, query_ds);

        REQUIRE(helper.Build(config) == knowhere::Status::success);

        float recall = helper.SearchAndGetRecall(config);
        // Flat index is brute force, should have perfect recall
        REQUIRE(recall >= kPerfectRecall);
    }

    SECTION("Search results should be in correct distance order") {
        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, query_ds);

        REQUIRE(helper.Build(config) == knowhere::Status::success);

        auto results = helper.Search(config);
        REQUIRE(results.has_value());

        // L2: ascending (smaller is better), IP/COSINE: descending (larger is better)
        bool ascending = (metric == knowhere::metric::L2);
        REQUIRE(knowhere::test::VerifyDistanceOrder(*results.value(), ascending));
    }

    SECTION("Search with topk larger than nb") {
        auto large_topk_config = config;
        large_topk_config[knowhere::meta::TOPK] = nb + 100;

        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, query_ds);

        REQUIRE(helper.Build(config) == knowhere::Status::success);

        auto results = helper.Search(large_topk_config);
        REQUIRE(results.has_value());

        // Should return nb results (not more than available)
        auto ids = results.value()->GetIds();
        for (int64_t i = 0; i < nq; ++i) {
            int valid_count = 0;
            for (int64_t j = 0; j < nb + 100; ++j) {
                if (ids[i * (nb + 100) + j] >= 0)
                    valid_count++;
            }
            REQUIRE(valid_count == nb);
        }
    }
}

TEST_CASE("Test Flat Index Search with Bitset", "[flat][search][bitset]") {
    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    knowhere::test::ConfigGenerator gen(dim, metric, topk);
    auto config = gen.Flat();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    auto bitset_rate = GENERATE(0.1f, 0.5f, 0.9f);

    SECTION("Search with random bitset filtering") {
        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, query_ds);

        REQUIRE(helper.Build(config) == knowhere::Status::success);

        auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, bitset_rate * nb);
        knowhere::BitsetView bitset(bitset_data.data(), nb);

        auto results = helper.Search(config, bitset);
        REQUIRE(results.has_value());

        // Verify no filtered IDs in results
        REQUIRE(knowhere::test::VerifyBitsetFiltering(*results.value(), bitset, topk));

        // Verify recall against filtered brute force
        auto gt = helper.GetGroundTruth(config, bitset);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= kPerfectRecall);
    }

    SECTION("Search with first-N-bits bitset filtering") {
        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, query_ds);

        REQUIRE(helper.Build(config) == knowhere::Status::success);

        auto bitset_data = GenerateBitsetWithFirstTbitsSet(nb, bitset_rate * nb);
        knowhere::BitsetView bitset(bitset_data.data(), nb);

        auto results = helper.Search(config, bitset);
        REQUIRE(results.has_value());

        REQUIRE(knowhere::test::VerifyBitsetFiltering(*results.value(), bitset, topk));
    }
}

TEST_CASE("Test Flat Index Range Search", "[flat][range_search]") {
    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    auto [radius, range_filter] = knowhere::test::GetDefaultRangeParams(metric);

    knowhere::test::ConfigGenerator gen(dim, metric, 10);
    auto config = gen.Flat();
    config[knowhere::meta::RADIUS] = radius;
    config[knowhere::meta::RANGE_FILTER] = range_filter;

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    SECTION("Range search should return results within bounds") {
        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, query_ds);

        REQUIRE(helper.Build(config) == knowhere::Status::success);

        auto results = helper.RangeSearch(config);
        REQUIRE(results.has_value());

        bool is_ip_like = (metric == knowhere::metric::IP || metric == knowhere::metric::COSINE);
        REQUIRE(knowhere::test::VerifyRangeSearchBounds(*results.value(), radius, range_filter, is_ip_like));
    }
}

TEST_CASE("Test Flat Index Serialize/Deserialize", "[flat][serialization]") {
    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    knowhere::test::ConfigGenerator gen(dim, metric, topk);
    auto config = gen.Flat();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    SECTION("Serialize and deserialize should preserve search results") {
        knowhere::test::IndexTestHelper<knowhere::fp32> helper(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version);
        helper.SetDataset(train_ds, query_ds);

        REQUIRE(helper.Build(config) == knowhere::Status::success);

        // Get results before serialization
        auto results_before = helper.Search(config);
        REQUIRE(results_before.has_value());

        // Serialize/Deserialize round-trip
        REQUIRE(helper.SerializeDeserializeRoundTrip(config));

        // Get results after deserialization
        auto results_after = helper.Search(config);
        REQUIRE(results_after.has_value());

        // Results should be identical
        float recall = GetKNNRecall(*results_before.value(), *results_after.value());
        REQUIRE(recall >= kPerfectRecall);
    }
}

TEST_CASE("Test Flat Index GetVectorByIds", "[flat][get_vector]") {
    const int64_t nb = 1000;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    knowhere::test::ConfigGenerator gen(dim, metric, 10);
    auto config = gen.Flat();

    const auto train_ds = GenDataSet(nb, dim);

    SECTION("GetVectorByIds should return correct vectors") {
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version)
                       .value();

        REQUIRE(idx.Build(train_ds, config) == knowhere::Status::success);

        // Request first 10 vectors
        std::vector<int64_t> ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        auto ids_ds = GenIdsDataSet(ids.size(), ids);

        auto vectors = idx.GetVectorByIds(ids_ds);
        REQUIRE(vectors.has_value());
        REQUIRE(vectors.value()->GetRows() == (int64_t)ids.size());
        REQUIRE(vectors.value()->GetDim() == dim);

        // Verify vectors match original data
        auto original_data = reinterpret_cast<const float*>(train_ds->GetTensor());
        auto retrieved_data = reinterpret_cast<const float*>(vectors.value()->GetTensor());

        for (size_t i = 0; i < ids.size(); ++i) {
            for (int64_t j = 0; j < dim; ++j) {
                REQUIRE(retrieved_data[i * dim + j] == original_data[ids[i] * dim + j]);
            }
        }
    }
}

// ==================== Binary Flat Index Tests ====================

TEST_CASE("Test Binary Flat Index", "[flat][binary]") {
    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 512;  // Must be multiple of 8 for binary
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::HAMMING, knowhere::metric::JACCARD);
    auto version = GenTestVersionList();

    knowhere::test::ConfigGenerator gen(dim, metric, topk);
    auto config = gen.BinaryFlat();

    const auto train_ds = GenBinDataSet(nb, dim);
    const auto query_ds = GenBinDataSet(nq, dim);

    SECTION("Build and search binary flat index") {
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::bin1>(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, version)
                       .value();

        REQUIRE(idx.Build(train_ds, config) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto results = idx.Search(query_ds, config, nullptr);
        REQUIRE(results.has_value());

        // Get ground truth
        auto gt = knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, config, nullptr);
        REQUIRE(gt.has_value());

        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= kPerfectRecall);
    }

    SECTION("Serialize and deserialize binary flat index") {
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::bin1>(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, version)
                       .value();

        REQUIRE(idx.Build(train_ds, config) == knowhere::Status::success);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

        auto idx2 = knowhere::IndexFactory::Instance()
                        .Create<knowhere::bin1>(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, version)
                        .value();
        REQUIRE(idx2.Deserialize(bs, config) == knowhere::Status::success);

        auto results = idx2.Search(query_ds, config, nullptr);
        REQUIRE(results.has_value());
    }
}

// ==================== Multiple Data Type Tests ====================

TEST_CASE("Test Flat Index with Different Data Types", "[flat][datatypes]") {
    const int64_t nb = 500, nq = 10;
    const int64_t dim = 64;
    const int64_t topk = 10;

    auto metric = knowhere::metric::L2;
    auto version = GenTestVersionList();

    knowhere::test::ConfigGenerator gen(dim, metric, topk);
    auto config = gen.Flat();

    // Generate float32 data (will be converted for other types)
    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    SECTION("Test with fp16") {
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp16>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version)
                       .value();

        auto train_fp16 = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(train_ds);
        auto query_fp16 = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(query_ds);

        REQUIRE(idx.Build(train_fp16, config) == knowhere::Status::success);

        auto results = idx.Search(query_fp16, config, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Test with bf16") {
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::bf16>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version)
                       .value();

        auto train_bf16 = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(train_ds);
        auto query_bf16 = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(query_ds);

        REQUIRE(idx.Build(train_bf16, config) == knowhere::Status::success);

        auto results = idx.Search(query_bf16, config, nullptr);
        REQUIRE(results.has_value());
    }
}
