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

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/utils/Heap.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/utils.h"
#include "simd/hook.h"
#include "utils.h"

namespace {

std::vector<uint8_t>
MakeBitset(size_t num_bits, std::initializer_list<size_t> filtered_ids) {
    std::vector<uint8_t> bits((num_bits + 7) / 8, 0);
    for (auto id : filtered_ids) {
        bits[id >> 3] |= static_cast<uint8_t>(1U << (id & 7));
    }
    return bits;
}

knowhere::IdArray
MakeIdArray(const std::vector<int32_t>& ids, knowhere::IdArray::Type type = knowhere::IdArray::Type::ARRAY) {
    knowhere::IdArray array;
    array.SetType(type);
    if (!ids.empty()) {
        if (type == knowhere::IdArray::Type::ARRAY) {
            array.Set(ids.data(), ids.size());
        } else {
            array.Append(ids.data(), ids.size());
        }
    }
    return array;
}

}  // namespace

template <typename T>
void
check_search(const knowhere::DataSetPtr train_ds, const knowhere::DataSetPtr query_ds, const int64_t k,
             const knowhere::MetricType metric, const knowhere::Json& conf) {
    auto base = knowhere::ConvertToDataTypeIfNeeded<T>(train_ds);
    auto query = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds);

    auto res = knowhere::BruteForce::Search<T>(base, query, conf, nullptr);
    auto nq = query_ds->GetRows();
    REQUIRE(res.has_value());
    auto ids = res.value()->GetIds();
    auto dist = res.value()->GetDistance();
    for (int64_t i = 0; i < nq; i++) {
        REQUIRE(ids[i * k] == i);
        if (metric == knowhere::metric::L2) {
            REQUIRE(dist[i * k] == 0);
        } else {
            REQUIRE(std::abs(dist[i * k] - 1.0) < 0.00001);
        }
    }
}

template <typename T>
void
check_search_with_buf(const knowhere::DataSetPtr train_ds, const knowhere::DataSetPtr query_ds, const int64_t k,
                      const knowhere::MetricType metric, const knowhere::Json& conf) {
    auto nq = query_ds->GetRows();
    auto ids = new int64_t[nq * k];
    auto dist = new float[nq * k];

    auto base = knowhere::ConvertToDataTypeIfNeeded<T>(train_ds);
    auto query = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds);

    auto res = knowhere::BruteForce::SearchWithBuf<T>(base, query, ids, dist, conf, nullptr);
    REQUIRE(res == knowhere::Status::success);
    for (int64_t i = 0; i < nq; i++) {
        REQUIRE(ids[i * k] == i);
        if (metric == knowhere::metric::L2) {
            REQUIRE(dist[i * k] == 0);
        } else {
            REQUIRE(std::abs(dist[i * k] - 1.0) < 0.00001);
        }
    }
    delete[] ids;
    delete[] dist;
}

template <typename T>
void
check_range_search(const knowhere::DataSetPtr train_ds, const knowhere::DataSetPtr query_ds, const int64_t k,
                   const knowhere::MetricType metric, const knowhere::Json& conf) {
    auto base = knowhere::ConvertToDataTypeIfNeeded<T>(train_ds);
    auto query = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds);

    auto res = knowhere::BruteForce::RangeSearch<T>(base, query, conf, nullptr);
    REQUIRE(res.has_value());
    auto ids = res.value()->GetIds();
    auto dist = res.value()->GetDistance();
    auto lims = res.value()->GetLims();
    auto nq = query_ds->GetRows();
    for (int64_t i = 0; i < nq; i++) {
        REQUIRE(lims[i] == (size_t)i);
        REQUIRE(ids[i] == i);
        if (metric == knowhere::metric::L2) {
            REQUIRE(dist[i] == 0);
        } else {
            REQUIRE(std::abs(dist[i] - 1.0) < 0.00001);
        }
    }
}

template <typename T>
void
check_search_with_out_ids(const uint64_t nb, const uint64_t nq, const uint64_t dim, const int64_t k,
                          const knowhere::MetricType metric, const knowhere::Json& conf) {
    auto total_train_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(nb, dim));
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(nq, dim));
    std::vector<int64_t> block_prefix = {0, 111, 333, 500, 555, 666, 888, 1000};

    // generate filter id and data
    auto filter_bits = GenerateBitsetWithRandomTbitsSet(nb, 100);
    knowhere::BitsetView bitset(filter_bits.data(), nb);

    std::vector<float> dis(nq * k, std::numeric_limits<float>::quiet_NaN());
    std::vector<int64_t> ids(nq * k, -1);
    if (metric == knowhere::metric::L2) {
        faiss::float_maxheap_array_t heaps{nq, (size_t)k, ids.data(), dis.data()};
        heaps.heapify();
        for (size_t i = 0; i < block_prefix.size() - 1; i++) {
            auto begin_id = block_prefix[i];
            auto end_id = block_prefix[i + 1];
            auto blk_rows = end_id - begin_id;
            auto tensor = (const T*)total_train_ds->GetTensor() + dim * begin_id;
            auto blk_train_ds = knowhere::GenDataSet(blk_rows, dim, tensor, begin_id);
            auto partial_v = knowhere::BruteForce::Search<T>(blk_train_ds, query_ds, conf, bitset);
            REQUIRE(partial_v.has_value());
            auto partial_res = partial_v.value();
            heaps.addn_with_ids(k, partial_res->GetDistance(), partial_res->GetIds(), k, 0, nq);
        }
        heaps.reorder();
    } else {
        faiss::float_minheap_array_t heaps{nq, (size_t)k, ids.data(), dis.data()};
        heaps.heapify();
        for (size_t i = 0; i < block_prefix.size() - 1; i++) {
            auto begin_id = block_prefix[i];
            auto end_id = block_prefix[i + 1];
            auto blk_rows = end_id - begin_id;
            auto tensor = (const T*)total_train_ds->GetTensor() + dim * begin_id;
            auto blk_train_ds = knowhere::GenDataSet(blk_rows, dim, tensor, begin_id);
            auto partial_v = knowhere::BruteForce::Search<T>(blk_train_ds, query_ds, conf, bitset);
            REQUIRE(partial_v.has_value());
            auto partial_res = partial_v.value();
            heaps.addn_with_ids(k, partial_res->GetDistance(), partial_res->GetIds(), k, 0, nq);
        }
        heaps.reorder();
    }

    auto gt = knowhere::BruteForce::Search<T>(total_train_ds, query_ds, conf, bitset);
    auto gt_ids = gt.value()->GetIds();
    const float* gt_dis = gt.value()->GetDistance();
    for (size_t i = 0; i < nq * k; i++) {
        REQUIRE(gt_ids[i] == ids[i]);
        REQUIRE(GetRelativeLoss(gt_dis[i], dis[i]) < 0.00001);
    }
}

TEST_CASE("Test Brute Force", "[float vector]") {
    using Catch::Approx;

    const int64_t nb = 1000;
    const int64_t nq = 10;
    const int64_t dim = 128;
    const int64_t k = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = CopyDataSet(train_ds, nq);

    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
        {knowhere::meta::RADIUS, knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99},
    };

    SECTION("Test Search") {
        check_search<knowhere::fp32>(train_ds, query_ds, k, metric, conf);
        check_search<knowhere::fp16>(train_ds, query_ds, k, metric, conf);
        check_search<knowhere::bf16>(train_ds, query_ds, k, metric, conf);
        check_search<knowhere::int8>(train_ds, query_ds, k, metric, conf);
    }

    SECTION("Test Search With Buf") {
        check_search_with_buf<knowhere::fp32>(train_ds, query_ds, k, metric, conf);
        check_search_with_buf<knowhere::fp16>(train_ds, query_ds, k, metric, conf);
        check_search_with_buf<knowhere::bf16>(train_ds, query_ds, k, metric, conf);
        check_search_with_buf<knowhere::int8>(train_ds, query_ds, k, metric, conf);
    }

    SECTION("Test Range Search") {
        check_range_search<knowhere::fp32>(train_ds, query_ds, k, metric, conf);
        check_range_search<knowhere::fp16>(train_ds, query_ds, k, metric, conf);
        check_range_search<knowhere::bf16>(train_ds, query_ds, k, metric, conf);
        check_range_search<knowhere::int8>(train_ds, query_ds, k, metric, conf);
    }
}

TEST_CASE("Test Brute Force", "[binary vector]") {
    using Catch::Approx;

    const int64_t nb = 1000;
    const int64_t nq = 10;
    const int64_t dim = 1024;
    const int64_t k = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::HAMMING, knowhere::metric::JACCARD,
                           knowhere::metric::SUPERSTRUCTURE, knowhere::metric::SUBSTRUCTURE);

    const auto train_ds = GenBinDataSet(nb, dim);
    const auto query_ds = CopyBinDataSet(train_ds, nq);

    std::unordered_map<std::string, float> radius_map = {
        {knowhere::metric::HAMMING, 1.0},
        {knowhere::metric::JACCARD, 0.1},
    };
    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
    };

    SECTION("Test Search") {
        auto res = knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, conf, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(ids[i * k] == i);
            REQUIRE(dist[i * k] == 0);
        }
    }

    SECTION("Test Search With Buf") {
        auto ids = new int64_t[nq * k];
        auto dist = new float[nq * k];
        auto res = knowhere::BruteForce::SearchWithBuf<knowhere::bin1>(train_ds, query_ds, ids, dist, conf, nullptr);
        REQUIRE(res == knowhere::Status::success);
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(ids[i * k] == i);
            REQUIRE(dist[i * k] == 0);
        }
        delete[] ids;
        delete[] dist;
    }

    SECTION("Test Range Search") {
        if (metric == knowhere::metric::SUPERSTRUCTURE || metric == knowhere::metric::SUBSTRUCTURE) {
            return;
        }

        // set radius for different metric type
        auto cfg = conf;
        cfg[knowhere::meta::RADIUS] = radius_map[metric];

        auto res = knowhere::BruteForce::RangeSearch<knowhere::bin1>(train_ds, query_ds, cfg, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        auto lims = res.value()->GetLims();
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(lims[i] == (size_t)i);
            REQUIRE(ids[i] == i);
            REQUIRE(dist[i] == 0);
        }
    }
}

TEST_CASE("Brute Force preserves trailing empty embedding lists", "[emb_list][trailing_empty]") {
    constexpr int64_t dim = 2;
    constexpr int64_t topk = 1;

    std::array<float, 2> base_tensor = {1.0f, 0.0f};
    std::array<size_t, 4> base_lims = {0, 1, 1, 1};
    auto base = knowhere::GenDataSet(1, dim, base_tensor.data());
    base->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(base_lims.data()));
    base->Set(knowhere::meta::EMB_LIST_COUNT, int64_t{3});

    std::array<float, 2> query_tensor = {1.0f, 0.0f};
    std::array<size_t, 2> query_lims = {0, 1};
    auto query = knowhere::GenDataSet(1, dim, query_tensor.data());
    query->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(query_lims.data()));
    query->Set(knowhere::meta::EMB_LIST_COUNT, int64_t{1});
    query->Set(knowhere::meta::NQ, int64_t{1});

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::TOPK] = topk;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_L2;

    auto result = knowhere::BruteForce::Search<knowhere::fp32>(base, query, conf, nullptr);

    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetRows() == 1);
    REQUIRE(knowhere::EmbListOffset(base_lims.data(), 1, knowhere::GetEmbListCount(base)).num_el() == 3);
}

TEST_CASE("Test Brute Force with input ids", "[float vector]") {
    using Catch::Approx;
    const int64_t nb = 1000;
    const int64_t nq = 10;
    const int64_t dim = 128;
    const int64_t k = 10;
    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
    };
    check_search_with_out_ids<knowhere::fp32>(nb, nq, dim, k, metric, conf);
    check_search_with_out_ids<knowhere::fp16>(nb, nq, dim, k, metric, conf);
    check_search_with_out_ids<knowhere::bf16>(nb, nq, dim, k, metric, conf);
    check_search_with_out_ids<knowhere::int8>(nb, nq, dim, k, metric, conf);
}

TEST_CASE("Brute Force filters internal rows through out ids", "[float vector][id_map]") {
    constexpr int64_t nb = 3;
    constexpr int64_t nq = 1;
    constexpr int64_t dim = 2;
    constexpr int64_t k = 2;

    const std::vector<float> base = {
        2.0F, 0.0F,  // internal row 0 -> out id 1
        0.0F, 0.0F,  // internal row 1 -> out id 3, filtered
        4.0F, 0.0F,  // internal row 2 -> out id 5
    };
    const std::vector<float> query = {0.0F, 0.0F};
    const std::vector<int32_t> out_ids = {1, 3, 5};
    auto out_id_array = MakeIdArray(out_ids);
    auto filter_bits = MakeBitset(6, {3});
    knowhere::BitsetView bitset(filter_bits.data(), 6);
    bitset.set_out_ids(out_id_array, out_ids.size());
    bitset.set_vector_count(out_ids.size());
    bitset.set_filter_count(1);

    auto base_ds = knowhere::GenDataSet(nb, dim, base.data());
    auto query_ds = knowhere::GenDataSet(nq, dim, query.data());
    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, k},
    };

    SECTION("SearchWithBuf returns logical out ids") {
        std::vector<int64_t> ids(nq * k, -1);
        std::vector<float> distances(nq * k, 0.0F);
        auto status = knowhere::BruteForce::SearchWithBuf<knowhere::fp32>(base_ds, query_ds, ids.data(),
                                                                          distances.data(), conf, bitset, nullptr);
        REQUIRE(status == knowhere::Status::success);
        REQUIRE(ids == std::vector<int64_t>{1, 5});
        REQUIRE(distances[0] == Catch::Approx(4.0F));
        REQUIRE(distances[1] == Catch::Approx(16.0F));
    }

    SECTION("Search returns logical out ids") {
        auto result = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, conf, bitset, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(std::vector<int64_t>(result.value()->GetIds(), result.value()->GetIds() + nq * k) ==
                std::vector<int64_t>{1, 5});
    }

    SECTION("RangeSearch returns logical out ids") {
        auto range_conf = conf;
        range_conf[knowhere::meta::RADIUS] = 20.0F;
        auto result = knowhere::BruteForce::RangeSearch<knowhere::fp32>(base_ds, query_ds, range_conf, bitset, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetLims()[0] == 0);
        REQUIRE(result.value()->GetLims()[1] == 2);
        REQUIRE(std::vector<int64_t>(result.value()->GetIds(), result.value()->GetIds() + 2) ==
                std::vector<int64_t>{1, 5});
    }

    SECTION("AnnIterator returns logical out ids") {
        auto iterators = knowhere::BruteForce::AnnIterator<knowhere::fp32>(base_ds, query_ds, conf, bitset, false);
        REQUIRE(iterators.has_value());
        REQUIRE(iterators.value().size() == 1);
        auto first = iterators.value()[0]->Next();
        REQUIRE(first.has_value());
        REQUIRE(first.value().first == 1);
        REQUIRE(first.value().second == Catch::Approx(4.0F));
    }

    SECTION("Out ids override tensor begin id") {
        auto offset_base_ds = knowhere::GenDataSet(nb, dim, base.data(), 100);
        auto result = knowhere::BruteForce::Search<knowhere::fp32>(offset_base_ds, query_ds, conf, bitset, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(std::vector<int64_t>(result.value()->GetIds(), result.value()->GetIds() + nq * k) ==
                std::vector<int64_t>{1, 5});
    }

    SECTION("Out ids use the active bitset window") {
        const std::vector<float> window_base = {
            2.0F, 0.0F,  // internal row 0 -> out id 1, filtered
            4.0F, 0.0F,  // internal row 1 -> out id 5
        };
        const std::vector<int32_t> window_out_ids = {0, 1, 5};
        auto window_out_id_array = MakeIdArray(window_out_ids);
        auto window_filter_bits = MakeBitset(6, {1});
        knowhere::BitsetView window_bitset(window_filter_bits.data(), 6);
        window_bitset.set_id_offset(1);
        window_bitset.set_out_ids(window_out_id_array, window_out_ids.size());
        window_bitset.set_vector_count(window_base.size() / dim);
        window_bitset.set_filter_count(1);

        auto window_base_ds = knowhere::GenDataSet(2, dim, window_base.data(), 1);
        auto result =
            knowhere::BruteForce::Search<knowhere::fp32>(window_base_ds, query_ds, conf, window_bitset, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(std::vector<int64_t>(result.value()->GetIds(), result.value()->GetIds() + nq * k) ==
                std::vector<int64_t>{5, -1});
    }

    SECTION("Chunk search slices out ids by chunk") {
        const std::vector<float> chunk0 = {
            2.0F, 0.0F,  // internal row 0 -> out id 1
            0.0F, 0.0F,  // internal row 1 -> out id 3, filtered
        };
        const std::vector<float> chunk1 = {
            4.0F, 0.0F,  // internal row 2 -> out id 5
            6.0F, 0.0F,  // internal row 3 -> out id 7
        };
        const std::array<const float*, 2> chunks = {chunk0.data(), chunk1.data()};
        const std::array<size_t, 3> chunk_lims = {0, 2, 4};
        const std::vector<int32_t> chunk_out_ids = {1, 3, 5, 7};
        auto chunk_out_id_array = MakeIdArray(chunk_out_ids);
        auto chunk_filter_bits = MakeBitset(8, {3});
        knowhere::BitsetView chunk_bitset(chunk_filter_bits.data(), 8);
        chunk_bitset.set_out_ids(chunk_out_id_array, chunk_out_ids.size());

        auto chunk_ds = knowhere::GenDataSet(4, dim, chunks.data());
        chunk_ds->SetIsChunk(true);
        chunk_ds->SetNumChunk(static_cast<int64_t>(chunks.size()));
        chunk_ds->Set(knowhere::meta::EMB_LIST_OFFSET, chunk_lims.data());

        auto result = knowhere::BruteForce::Search<knowhere::fp32>(chunk_ds, query_ds, conf, chunk_bitset, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(std::vector<int64_t>(result.value()->GetIds(), result.value()->GetIds() + nq * k) ==
                std::vector<int64_t>{1, 5});
    }

    SECTION("Chunk search slices concurrent out ids by chunk") {
        const std::vector<float> chunk0 = {
            2.0F, 0.0F,  // internal row 0 -> out id 1
            0.0F, 0.0F,  // internal row 1 -> out id 3, filtered
        };
        const std::vector<float> chunk1 = {
            4.0F, 0.0F,  // internal row 2 -> out id 5
            6.0F, 0.0F,  // internal row 3 -> out id 7
        };
        const std::array<const float*, 2> chunks = {chunk0.data(), chunk1.data()};
        const std::array<size_t, 3> chunk_lims = {0, 2, 4};
        const std::vector<int32_t> values = {1, 3, 5, 7};
        auto chunk_out_id_array = MakeIdArray(values, knowhere::IdArray::Type::APPEND_ARRAY);
        auto chunk_filter_bits = MakeBitset(8, {3});
        knowhere::BitsetView chunk_bitset(chunk_filter_bits.data(), 8);
        chunk_bitset.set_out_ids(chunk_out_id_array, values.size());

        auto chunk_ds = knowhere::GenDataSet(4, dim, chunks.data());
        chunk_ds->SetIsChunk(true);
        chunk_ds->SetNumChunk(static_cast<int64_t>(chunks.size()));
        chunk_ds->Set(knowhere::meta::EMB_LIST_OFFSET, chunk_lims.data());

        auto result = knowhere::BruteForce::Search<knowhere::fp32>(chunk_ds, query_ds, conf, chunk_bitset, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(std::vector<int64_t>(result.value()->GetIds(), result.value()->GetIds() + nq * k) ==
                std::vector<int64_t>{1, 5});
    }

    SECTION("Chunk iterator slices out ids by chunk") {
        const std::vector<float> chunk0 = {
            2.0F, 0.0F,  // internal row 0 -> out id 1
            0.0F, 0.0F,  // internal row 1 -> out id 3, filtered
        };
        const std::vector<float> chunk1 = {
            4.0F, 0.0F,  // internal row 2 -> out id 5
            6.0F, 0.0F,  // internal row 3 -> out id 7
        };
        const std::array<const float*, 2> chunks = {chunk0.data(), chunk1.data()};
        const std::array<size_t, 3> chunk_lims = {0, 2, 4};
        const std::vector<int32_t> chunk_out_ids = {1, 3, 5, 7};
        auto chunk_out_id_array = MakeIdArray(chunk_out_ids);
        auto chunk_filter_bits = MakeBitset(8, {3});
        knowhere::BitsetView chunk_bitset(chunk_filter_bits.data(), 8);
        chunk_bitset.set_out_ids(chunk_out_id_array, chunk_out_ids.size());

        auto chunk_ds = knowhere::GenDataSet(4, dim, chunks.data());
        chunk_ds->SetIsChunk(true);
        chunk_ds->SetNumChunk(static_cast<int64_t>(chunks.size()));
        chunk_ds->Set(knowhere::meta::EMB_LIST_OFFSET, chunk_lims.data());

        auto iterators =
            knowhere::BruteForce::AnnIterator<knowhere::fp32>(chunk_ds, query_ds, conf, chunk_bitset, false);
        REQUIRE(iterators.has_value());
        REQUIRE(iterators.value().size() == 1);
        auto first = iterators.value()[0]->Next();
        REQUIRE(first.has_value());
        REQUIRE(first.value().first == 1);
        auto second = iterators.value()[0]->Next();
        REQUIRE(second.has_value());
        REQUIRE(second.value().first == 5);
    }
}

TEST_CASE("Sparse Brute Force maps logical out ids", "[sparse][id_map]") {
    constexpr int64_t nq = 1;
    constexpr int64_t dim = 8;
    constexpr int64_t k = 2;

    const std::vector<std::map<int32_t, float>> base = {
        {{1, 2.0F}},  // internal row 0 -> out id 1, filtered
        {{1, 1.0F}},  // internal row 1 -> out id 3
        {{1, 0.5F}},  // internal row 2 -> out id 5
    };
    const std::vector<std::map<int32_t, float>> query = {{{1, 1.0F}}};
    const std::vector<int32_t> out_ids = {1, 3, 5};
    auto out_id_array = MakeIdArray(out_ids);
    auto filter_bits = MakeBitset(6, {1});
    knowhere::BitsetView bitset(filter_bits.data(), 6);
    bitset.set_out_ids(out_id_array, out_ids.size());
    bitset.set_vector_count(out_ids.size());
    bitset.set_filter_count(1);

    auto base_ds = GenSparseDataSet(base, dim);
    auto query_ds = GenSparseDataSet(query, dim);
    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, k},
    };

    SECTION("SearchSparseWithBuf returns logical out ids") {
        std::vector<knowhere::sparse::label_t> ids(nq * k, -1);
        std::vector<float> distances(nq * k, 0.0F);
        auto status = knowhere::BruteForce::SearchSparseWithBuf(base_ds, query_ds, ids.data(), distances.data(), conf,
                                                                bitset, nullptr);
        REQUIRE(status == knowhere::Status::success);
        REQUIRE(ids == std::vector<knowhere::sparse::label_t>{3, 5});
        REQUIRE(distances[0] == Catch::Approx(1.0F));
        REQUIRE(distances[1] == Catch::Approx(0.5F));
    }

    SECTION("SearchSparse returns logical out ids") {
        auto result = knowhere::BruteForce::SearchSparse(base_ds, query_ds, conf, bitset, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(std::vector<int64_t>(result.value()->GetIds(), result.value()->GetIds() + nq * k) ==
                std::vector<int64_t>{3, 5});
    }

    SECTION("RangeSearch returns logical out ids") {
        auto range_conf = conf;
        range_conf[knowhere::meta::RADIUS] = 0.0F;
        range_conf[knowhere::meta::RANGE_FILTER] = 2.0F;
        auto result = knowhere::BruteForce::RangeSearch<knowhere::sparse::SparseRow<float>>(
            base_ds, query_ds, range_conf, bitset, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetLims()[0] == 0);
        REQUIRE(result.value()->GetLims()[1] == 2);
        REQUIRE(std::vector<int64_t>(result.value()->GetIds(), result.value()->GetIds() + 2) ==
                std::vector<int64_t>{3, 5});
    }

    SECTION("AnnIterator returns logical out ids") {
        auto iterators = knowhere::BruteForce::AnnIterator<knowhere::sparse::SparseRow<float>>(base_ds, query_ds, conf,
                                                                                               bitset, false);
        REQUIRE(iterators.has_value());
        REQUIRE(iterators.value().size() == 1);
        auto first = iterators.value()[0]->Next();
        REQUIRE(first.has_value());
        REQUIRE(first.value().first == 3);
        REQUIRE(first.value().second == Catch::Approx(1.0F));
        auto second = iterators.value()[0]->Next();
        REQUIRE(second.has_value());
        REQUIRE(second.value().first == 5);
        REQUIRE(second.value().second == Catch::Approx(0.5F));
    }
}
