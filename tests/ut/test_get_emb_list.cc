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

#include <iostream>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_static.h"
#include "utils.h"

#define REQUIRE_HAS_VALUE(result)                                                                                      \
    do {                                                                                                               \
        if (!(result).has_value()) {                                                                                   \
            std::cerr << "GetEmbListByIds failed: " << (result).what() << " (status: " << (int)(result).error() << ")" \
                      << std::endl;                                                                                    \
        }                                                                                                              \
        REQUIRE((result).has_value());                                                                                 \
    } while (0)

#ifndef KNOWHERE_WITH_CARDINAL

TEST_CASE("Test GetEmbListByIds Basic", "[GetEmbListByIds]") {
    const int64_t dim = 4;
    const int64_t each_el_len = 10;
    const int64_t nb = 256;                                       // total vectors
    const int64_t num_el = (nb + each_el_len - 1) / each_el_len;  // number of embedding lists

    auto version = GenTestEmbListVersionList();

    // Build HNSW FLAT index with emb_list data
    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    conf[knowhere::indexparam::HNSW_M] = 16;
    conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    conf[knowhere::indexparam::EF] = 64;

    auto train_ds = GenEmbListDataSet(nb, dim, 42, each_el_len);
    auto original_data = (const float*)train_ds->GetTensor();
    auto original_offsets = train_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);

    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    auto res = idx.Build(train_ds, conf);
    REQUIRE(res == knowhere::Status::success);

    // Serialize and deserialize to ensure emb_list_offset_ is properly restored
    knowhere::BinarySet bs;
    idx.Serialize(bs);
    auto idx_loaded =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    idx_loaded.Deserialize(bs, conf);

    if (!idx_loaded.HasRawData(conf[knowhere::meta::METRIC_TYPE])) {
        SKIP("Index does not support raw data retrieval");
    }

    SECTION("Retrieve single embedding list") {
        int64_t el_id = 3;
        auto ids_ds = knowhere::GenIdsDataSet(1, &el_id);

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == 1);
        REQUIRE(result_ds->GetDim() == dim);

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);
        REQUIRE(result_offsets[0] == 0);
        REQUIRE(result_offsets[1] == each_el_len);

        // Verify data matches original
        auto result_data = (const float*)result_ds->GetTensor();
        size_t orig_vec_start = original_offsets[el_id];
        for (int64_t v = 0; v < each_el_len; v++) {
            for (int64_t d = 0; d < dim; d++) {
                REQUIRE(result_data[v * dim + d] == original_data[(orig_vec_start + v) * dim + d]);
            }
        }
    }

    SECTION("Retrieve multiple embedding lists") {
        std::vector<int64_t> el_ids = {0, 5, num_el - 1};
        auto ids_ds = knowhere::GenIdsDataSet(el_ids.size(), el_ids.data());

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == (int64_t)el_ids.size());
        REQUIRE(result_ds->GetDim() == dim);

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);
        REQUIRE(result_offsets[0] == 0);

        // Verify each embedding list
        auto result_data = (const float*)result_ds->GetTensor();
        for (size_t i = 0; i < el_ids.size(); i++) {
            auto el_id = el_ids[i];
            size_t orig_vec_start = original_offsets[el_id];
            size_t el_len = original_offsets[el_id + 1] - original_offsets[el_id];
            REQUIRE(result_offsets[i + 1] - result_offsets[i] == el_len);

            for (size_t v = 0; v < el_len; v++) {
                for (int64_t d = 0; d < dim; d++) {
                    REQUIRE(result_data[(result_offsets[i] + v) * dim + d] ==
                            original_data[(orig_vec_start + v) * dim + d]);
                }
            }
        }
    }

    SECTION("Retrieve all embedding lists") {
        std::vector<int64_t> el_ids(num_el);
        for (int64_t i = 0; i < num_el; i++) {
            el_ids[i] = i;
        }
        auto ids_ds = knowhere::GenIdsDataSet(el_ids.size(), el_ids.data());

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == num_el);
        REQUIRE(result_ds->GetDim() == dim);

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);
        REQUIRE(result_offsets[num_el] == (size_t)nb);

        // Verify all data matches original
        auto result_data = (const float*)result_ds->GetTensor();
        for (int64_t i = 0; i < nb * dim; i++) {
            REQUIRE(result_data[i] == original_data[i]);
        }
    }

    SECTION("Duplicate el_ids") {
        std::vector<int64_t> el_ids = {2, 2, 5, 5};
        auto ids_ds = knowhere::GenIdsDataSet(el_ids.size(), el_ids.data());

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == (int64_t)el_ids.size());

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);
        REQUIRE(result_offsets[4] == 4 * each_el_len);

        // Verify duplicates return same data
        auto result_data = (const float*)result_ds->GetTensor();
        for (int64_t v = 0; v < each_el_len * dim; v++) {
            REQUIRE(result_data[v] == result_data[each_el_len * dim + v]);
        }
    }
}

TEST_CASE("Test GetEmbListByIds with variable-length embedding lists", "[GetEmbListByIds]") {
    const int64_t dim = 4;
    const int64_t each_el_len = 10;
    const int64_t nb = 256;

    auto version = GenTestEmbListVersionList();

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    conf[knowhere::indexparam::HNSW_M] = 16;
    conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    conf[knowhere::indexparam::EF] = 64;

    // GenEmbListDataSetWithSomeEmpty creates offsets [0,0,20,20,40,40,...] so even-indexed el_ids are empty
    auto train_ds = GenEmbListDataSetWithSomeEmpty(nb, dim, 42, each_el_len);
    auto original_data = (const float*)train_ds->GetTensor();
    auto original_offsets = train_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);

    // Count actual number of embedding lists
    size_t num_el = 0;
    while (original_offsets[num_el] < (size_t)nb) {
        num_el++;
    }
    // original_offsets[num_el] == nb

    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    auto res = idx.Build(train_ds, conf);
    REQUIRE(res == knowhere::Status::success);

    knowhere::BinarySet bs;
    idx.Serialize(bs);
    auto idx_loaded =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    idx_loaded.Deserialize(bs, conf);

    if (!idx_loaded.HasRawData(conf[knowhere::meta::METRIC_TYPE])) {
        SKIP("Index does not support raw data retrieval");
    }

    SECTION("Retrieve non-empty embedding list") {
        // el_id 1 is non-empty: offset[1]=0, offset[2]=20, so it has 20 vectors
        int64_t el_id = 1;
        auto ids_ds = knowhere::GenIdsDataSet(1, &el_id);

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        size_t el_len = result_offsets[1] - result_offsets[0];
        REQUIRE(el_len == (original_offsets[el_id + 1] - original_offsets[el_id]));

        auto result_data = (const float*)result_ds->GetTensor();
        for (size_t v = 0; v < el_len; v++) {
            for (int64_t d = 0; d < dim; d++) {
                REQUIRE(result_data[v * dim + d] == original_data[(original_offsets[el_id] + v) * dim + d]);
            }
        }
    }

    SECTION("Retrieve empty embedding list") {
        // GenEmbListDataSetWithSomeEmpty produces offsets [0, 0, 20, 20, 40, 40, ...].
        // Even-indexed el_ids (0, 2, 4, ...) are empty; odd-indexed ones absorb the vectors.
        int64_t el_id = 0;  // el_id=0 is empty: offset[0]==offset[1]==0
        auto ids_ds = knowhere::GenIdsDataSet(1, &el_id);

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets[0] == 0);
        REQUIRE(result_offsets[1] == 0);
    }

    SECTION("Retrieve mix of empty and non-empty") {
        // el_ids 0, 2 are empty; 1, 3 are non-empty
        std::vector<int64_t> el_ids = {0, 1, 2, 3};
        auto ids_ds = knowhere::GenIdsDataSet(el_ids.size(), el_ids.data());

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == (int64_t)el_ids.size());

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);

        auto result_data = (const float*)result_ds->GetTensor();
        for (size_t i = 0; i < el_ids.size(); i++) {
            auto el_id = el_ids[i];
            size_t el_len = original_offsets[el_id + 1] - original_offsets[el_id];
            REQUIRE(result_offsets[i + 1] - result_offsets[i] == el_len);

            size_t orig_vec_start = original_offsets[el_id];
            for (size_t v = 0; v < el_len; v++) {
                for (int64_t d = 0; d < dim; d++) {
                    REQUIRE(result_data[(result_offsets[i] + v) * dim + d] ==
                            original_data[(orig_vec_start + v) * dim + d]);
                }
            }
        }
    }
}

TEST_CASE("Test GetEmbListByIds error cases", "[GetEmbListByIds]") {
    const int64_t dim = 4;
    const int64_t each_el_len = 10;
    const int64_t nb = 256;
    const int64_t num_el = (nb + each_el_len - 1) / each_el_len;

    auto version = GenTestEmbListVersionList();

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    conf[knowhere::indexparam::HNSW_M] = 16;
    conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    conf[knowhere::indexparam::EF] = 64;

    SECTION("Error when no emb_list_offset (non-emblist index)") {
        // Build a normal (non-emblist) index
        auto train_ds = GenDataSet(nb, dim);
        knowhere::Json normal_conf;
        normal_conf[knowhere::meta::DIM] = dim;
        normal_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
        normal_conf[knowhere::meta::TOPK] = 10;
        normal_conf[knowhere::indexparam::HNSW_M] = 16;
        normal_conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
        normal_conf[knowhere::indexparam::EF] = 64;

        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = idx.Build(train_ds, normal_conf);
        REQUIRE(res == knowhere::Status::success);

        int64_t el_id = 0;
        auto ids_ds = knowhere::GenIdsDataSet(1, &el_id);
        auto result = idx.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE(!result.has_value());
    }

    SECTION("Error when el_id out of range") {
        auto train_ds = GenEmbListDataSet(nb, dim, 42, each_el_len);
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = idx.Build(train_ds, conf);
        REQUIRE(res == knowhere::Status::success);

        int64_t boundary_el_id = num_el;
        auto ids_ds = knowhere::GenIdsDataSet(1, &boundary_el_id);
        auto result = idx.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE(!result.has_value());
    }

    SECTION("Error when el_id is negative") {
        auto train_ds = GenEmbListDataSet(nb, dim, 42, each_el_len);
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = idx.Build(train_ds, conf);
        REQUIRE(res == knowhere::Status::success);

        int64_t bad_el_id = -1;
        auto ids_ds = knowhere::GenIdsDataSet(1, &bad_el_id);
        auto result = idx.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE(!result.has_value());
    }

    SECTION("Error when metric_type is invalid") {
        auto train_ds = GenEmbListDataSet(nb, dim, 42, each_el_len);
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = idx.Build(train_ds, conf);
        REQUIRE(res == knowhere::Status::success);

        int64_t el_id = 0;
        auto ids_ds = knowhere::GenIdsDataSet(1, &el_id);
        // COSINE is not an emblist metric, get_sub_metric_type returns nullopt
        auto result = idx.GetEmbListByIds(ids_ds, knowhere::metric::COSINE);
        REQUIRE(!result.has_value());
    }
}

TEST_CASE("Test GetEmbListByIds after serialize/deserialize", "[GetEmbListByIds]") {
    const int64_t dim = 4;
    const int64_t each_el_len = 10;
    const int64_t nb = 256;
    const int64_t num_el = (nb + each_el_len - 1) / each_el_len;

    auto version = GenTestEmbListVersionList();

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    conf[knowhere::indexparam::HNSW_M] = 16;
    conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    conf[knowhere::indexparam::EF] = 64;

    auto train_ds = GenEmbListDataSet(nb, dim, 42, each_el_len);
    auto original_data = (const float*)train_ds->GetTensor();
    auto original_offsets = train_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);

    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    auto res = idx.Build(train_ds, conf);
    REQUIRE(res == knowhere::Status::success);

    if (!idx.HasRawData(conf[knowhere::meta::METRIC_TYPE])) {
        SKIP("Index does not support raw data retrieval");
    }

    // Get results from original index
    std::vector<int64_t> el_ids = {0, 3, num_el - 1};
    auto ids_ds = knowhere::GenIdsDataSet(el_ids.size(), el_ids.data());
    auto result_before = idx.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
    REQUIRE_HAS_VALUE(result_before);

    // Serialize and deserialize
    knowhere::BinarySet bs;
    idx.Serialize(bs);
    auto idx_loaded =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    idx_loaded.Deserialize(bs, conf);

    // Get results from deserialized index
    auto result_after = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
    REQUIRE_HAS_VALUE(result_after);

    // Compare: both should return identical data
    auto ds_before = result_before.value();
    auto ds_after = result_after.value();
    REQUIRE(ds_before->GetRows() == ds_after->GetRows());
    REQUIRE(ds_before->GetDim() == ds_after->GetDim());

    auto offsets_before = ds_before->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
    auto offsets_after = ds_after->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
    auto data_before = (const float*)ds_before->GetTensor();
    auto data_after = (const float*)ds_after->GetTensor();

    for (size_t i = 0; i <= el_ids.size(); i++) {
        REQUIRE(offsets_before[i] == offsets_after[i]);
    }

    size_t total_vectors = offsets_before[el_ids.size()];
    for (size_t i = 0; i < total_vectors * dim; i++) {
        REQUIRE(data_before[i] == data_after[i]);
    }

    // Also verify against original training data
    for (size_t i = 0; i < el_ids.size(); i++) {
        auto el_id = el_ids[i];
        size_t orig_vec_start = original_offsets[el_id];
        size_t el_len = original_offsets[el_id + 1] - original_offsets[el_id];
        for (size_t v = 0; v < el_len; v++) {
            for (int64_t d = 0; d < dim; d++) {
                REQUIRE(data_after[(offsets_after[i] + v) * dim + d] == original_data[(orig_vec_start + v) * dim + d]);
            }
        }
    }
}

TEST_CASE("Test GetEmbListByIds with TokenANN + HNSW_SQ", "[GetEmbListByIds]") {
    const int64_t dim = 4;
    const int64_t each_el_len = 10;
    const int64_t nb = 256;
    const int64_t num_el = (nb + each_el_len - 1) / each_el_len;

    auto version = GenTestEmbListVersionList();

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW_SQ;
    conf[knowhere::indexparam::HNSW_M] = 16;
    conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    conf[knowhere::indexparam::EF] = 64;
    conf[knowhere::indexparam::SQ_TYPE] = "SQ8";
    conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 3.0f;

    auto train_ds = GenEmbListDataSet(nb, dim, 42, each_el_len);
    auto original_data = (const float*)train_ds->GetTensor();
    auto original_offsets = train_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);

    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    auto res = idx.Build(train_ds, conf);
    REQUIRE(res == knowhere::Status::success);

    knowhere::BinarySet bs;
    idx.Serialize(bs);
    auto idx_loaded =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    idx_loaded.Deserialize(bs, conf);

    if (!idx_loaded.HasRawData(conf[knowhere::meta::METRIC_TYPE])) {
        SKIP("Index does not support raw data retrieval");
    }

    SECTION("Retrieve single embedding list") {
        int64_t el_id = 3;
        auto ids_ds = knowhere::GenIdsDataSet(1, &el_id);

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == 1);
        REQUIRE(result_ds->GetDim() == dim);

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);
        REQUIRE(result_offsets[0] == 0);
        REQUIRE(result_offsets[1] == each_el_len);

        auto result_data = (const float*)result_ds->GetTensor();
        size_t orig_vec_start = original_offsets[el_id];
        for (int64_t v = 0; v < each_el_len; v++) {
            for (int64_t d = 0; d < dim; d++) {
                REQUIRE(result_data[v * dim + d] == original_data[(orig_vec_start + v) * dim + d]);
            }
        }
    }

    SECTION("Retrieve multiple embedding lists") {
        std::vector<int64_t> el_ids = {0, 5, num_el - 1};
        auto ids_ds = knowhere::GenIdsDataSet(el_ids.size(), el_ids.data());

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == (int64_t)el_ids.size());

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);

        auto result_data = (const float*)result_ds->GetTensor();
        for (size_t i = 0; i < el_ids.size(); i++) {
            auto el_id = el_ids[i];
            size_t orig_vec_start = original_offsets[el_id];
            size_t el_len = original_offsets[el_id + 1] - original_offsets[el_id];
            REQUIRE(result_offsets[i + 1] - result_offsets[i] == el_len);

            for (size_t v = 0; v < el_len; v++) {
                for (int64_t d = 0; d < dim; d++) {
                    REQUIRE(result_data[(result_offsets[i] + v) * dim + d] ==
                            original_data[(orig_vec_start + v) * dim + d]);
                }
            }
        }
    }
}

TEST_CASE("Test GetEmbListByIds with MUVERA + HNSW_FLAT", "[GetEmbListByIds]") {
    const int64_t dim = 4;
    const int64_t each_el_len = 10;
    const int64_t nb = 256;
    const int64_t num_el = (nb + each_el_len - 1) / each_el_len;

    auto version = GenTestEmbListVersionList();

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    conf[knowhere::indexparam::HNSW_M] = 16;
    conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    conf[knowhere::indexparam::EF] = 64;
    conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 3.0f;
    conf["emb_list_strategy"] = "muvera";
    conf["muvera_num_projections"] = 3;
    conf["muvera_num_repeats"] = 5;
    conf["muvera_seed"] = 42;

    auto train_ds = GenEmbListDataSet(nb, dim, 42, each_el_len);
    auto original_data = (const float*)train_ds->GetTensor();
    auto original_offsets = train_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);

    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    auto res = idx.Build(train_ds, conf);
    REQUIRE(res == knowhere::Status::success);

    knowhere::BinarySet bs;
    idx.Serialize(bs);
    auto idx_loaded =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    idx_loaded.Deserialize(bs, conf);

    SECTION("Retrieve single embedding list") {
        int64_t el_id = 3;
        auto ids_ds = knowhere::GenIdsDataSet(1, &el_id);

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == 1);
        REQUIRE(result_ds->GetDim() == dim);

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);
        REQUIRE(result_offsets[0] == 0);
        REQUIRE(result_offsets[1] == each_el_len);

        // MUVERA stores raw vectors in emb_list_raw_index_, verify data matches original
        auto result_data = (const float*)result_ds->GetTensor();
        size_t orig_vec_start = original_offsets[el_id];
        for (int64_t v = 0; v < each_el_len; v++) {
            for (int64_t d = 0; d < dim; d++) {
                REQUIRE(result_data[v * dim + d] == original_data[(orig_vec_start + v) * dim + d]);
            }
        }
    }

    SECTION("Retrieve multiple embedding lists") {
        std::vector<int64_t> el_ids = {0, 5, num_el - 1};
        auto ids_ds = knowhere::GenIdsDataSet(el_ids.size(), el_ids.data());

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == (int64_t)el_ids.size());

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);

        auto result_data = (const float*)result_ds->GetTensor();
        for (size_t i = 0; i < el_ids.size(); i++) {
            auto el_id = el_ids[i];
            size_t orig_vec_start = original_offsets[el_id];
            size_t el_len = original_offsets[el_id + 1] - original_offsets[el_id];
            REQUIRE(result_offsets[i + 1] - result_offsets[i] == el_len);

            for (size_t v = 0; v < el_len; v++) {
                for (int64_t d = 0; d < dim; d++) {
                    REQUIRE(result_data[(result_offsets[i] + v) * dim + d] ==
                            original_data[(orig_vec_start + v) * dim + d]);
                }
            }
        }
    }
}

TEST_CASE("Test GetEmbListByIds with LEMUR + HNSW_FLAT", "[GetEmbListByIds]") {
    const int64_t dim = 4;
    const int64_t each_el_len = 10;
    // LEMUR needs num_docs >= hidden_dim, use larger dataset
    const int64_t nb = 512;
    const int64_t num_el = (nb + each_el_len - 1) / each_el_len;

    auto version = GenTestEmbListVersionList();

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::meta::ROWS] = nb;
    conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    conf[knowhere::indexparam::HNSW_M] = 16;
    conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    conf[knowhere::indexparam::EF] = 64;
    conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 3.0f;
    conf["emb_list_strategy"] = "lemur";
    conf["lemur_hidden_dim"] = 32;
    conf["lemur_num_train_samples"] = 1000;
    conf["lemur_num_epochs"] = 2;
    conf["lemur_batch_size"] = 16;
    conf["lemur_learning_rate"] = 0.001f;
    conf["lemur_seed"] = 42;
    conf["lemur_num_layers"] = 1;

    auto train_ds = GenEmbListDataSet(nb, dim, 42, each_el_len);
    auto original_data = (const float*)train_ds->GetTensor();
    auto original_offsets = train_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);

    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    auto res = idx.Build(train_ds, conf);
    REQUIRE(res == knowhere::Status::success);

    knowhere::BinarySet bs;
    idx.Serialize(bs);
    auto idx_loaded =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    idx_loaded.Deserialize(bs, conf);

    SECTION("Retrieve single embedding list") {
        int64_t el_id = 3;
        auto ids_ds = knowhere::GenIdsDataSet(1, &el_id);

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == 1);
        REQUIRE(result_ds->GetDim() == dim);

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);
        REQUIRE(result_offsets[0] == 0);
        REQUIRE(result_offsets[1] == each_el_len);

        // LEMUR stores raw vectors in emb_list_raw_index_, verify data matches original
        auto result_data = (const float*)result_ds->GetTensor();
        size_t orig_vec_start = original_offsets[el_id];
        for (int64_t v = 0; v < each_el_len; v++) {
            for (int64_t d = 0; d < dim; d++) {
                REQUIRE(result_data[v * dim + d] == original_data[(orig_vec_start + v) * dim + d]);
            }
        }
    }

    SECTION("Retrieve multiple embedding lists") {
        std::vector<int64_t> el_ids = {0, 5, num_el - 1};
        auto ids_ds = knowhere::GenIdsDataSet(el_ids.size(), el_ids.data());

        auto result = idx_loaded.GetEmbListByIds(ids_ds, knowhere::metric::MAX_SIM_COSINE);
        REQUIRE_HAS_VALUE(result);

        auto result_ds = result.value();
        REQUIRE(result_ds->GetRows() == (int64_t)el_ids.size());

        auto result_offsets = result_ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(result_offsets != nullptr);

        auto result_data = (const float*)result_ds->GetTensor();
        for (size_t i = 0; i < el_ids.size(); i++) {
            auto el_id = el_ids[i];
            size_t orig_vec_start = original_offsets[el_id];
            size_t el_len = original_offsets[el_id + 1] - original_offsets[el_id];
            REQUIRE(result_offsets[i + 1] - result_offsets[i] == el_len);

            for (size_t v = 0; v < el_len; v++) {
                for (int64_t d = 0; d < dim; d++) {
                    REQUIRE(result_data[(result_offsets[i] + v) * dim + d] ==
                            original_data[(orig_vec_start + v) * dim + d]);
                }
            }
        }
    }
}

TEST_CASE("Test MUVERA/LEMUR reject non-fp32 data types", "[GetEmbListByIds]") {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    std::string msg;

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = 4;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::indexparam::HNSW_M] = 16;
    conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    conf[knowhere::indexparam::EF] = 64;

    SECTION("MUVERA with fp16 should fail ConfigCheck") {
        conf["emb_list_strategy"] = "muvera";
        conf["muvera_num_projections"] = 3;
        conf["muvera_num_repeats"] = 5;
        conf["muvera_seed"] = 42;

        auto status = knowhere::IndexStaticFaced<knowhere::fp16>::ConfigCheck(knowhere::IndexEnum::INDEX_HNSW, version,
                                                                              conf, msg);
        REQUIRE(status != knowhere::Status::success);
        REQUIRE(msg.find("fp32") != std::string::npos);
    }

    SECTION("MUVERA with bf16 should fail ConfigCheck") {
        conf["emb_list_strategy"] = "muvera";
        conf["muvera_num_projections"] = 3;
        conf["muvera_num_repeats"] = 5;
        conf["muvera_seed"] = 42;

        auto status = knowhere::IndexStaticFaced<knowhere::bf16>::ConfigCheck(knowhere::IndexEnum::INDEX_HNSW, version,
                                                                              conf, msg);
        REQUIRE(status != knowhere::Status::success);
        REQUIRE(msg.find("fp32") != std::string::npos);
    }

    SECTION("LEMUR with fp16 should fail ConfigCheck") {
        conf["emb_list_strategy"] = "lemur";
        conf["lemur_hidden_dim"] = 32;

        auto status = knowhere::IndexStaticFaced<knowhere::fp16>::ConfigCheck(knowhere::IndexEnum::INDEX_HNSW, version,
                                                                              conf, msg);
        REQUIRE(status != knowhere::Status::success);
        REQUIRE(msg.find("fp32") != std::string::npos);
    }

    SECTION("MUVERA with fp32 should pass ConfigCheck") {
        conf["emb_list_strategy"] = "muvera";
        conf["muvera_num_projections"] = 3;
        conf["muvera_num_repeats"] = 5;
        conf["muvera_seed"] = 42;

        auto status = knowhere::IndexStaticFaced<knowhere::fp32>::ConfigCheck(knowhere::IndexEnum::INDEX_HNSW, version,
                                                                              conf, msg);
        REQUIRE(status == knowhere::Status::success);
    }

    SECTION("TokenANN with fp16 should pass ConfigCheck") {
        conf["emb_list_strategy"] = "tokenann";

        auto status = knowhere::IndexStaticFaced<knowhere::fp16>::ConfigCheck(knowhere::IndexEnum::INDEX_HNSW, version,
                                                                              conf, msg);
        REQUIRE(status == knowhere::Status::success);
    }

    SECTION("No strategy (default tokenann) with fp16 should pass ConfigCheck") {
        auto status = knowhere::IndexStaticFaced<knowhere::fp16>::ConfigCheck(knowhere::IndexEnum::INDEX_HNSW, version,
                                                                              conf, msg);
        REQUIRE(status == knowhere::Status::success);
    }
}

TEST_CASE("Test DiskANN rejects MUVERA/LEMUR strategies", "[GetEmbListByIds]") {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    std::string msg;

    knowhere::Json conf;
    conf[knowhere::meta::DIM] = 4;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::MAX_SIM_COSINE;
    conf[knowhere::meta::TOPK] = 10;
    conf[knowhere::indexparam::MAX_DEGREE] = 48;
    conf[knowhere::indexparam::SEARCH_LIST_SIZE] = 128;

    SECTION("MUVERA with fp32 should fail ConfigCheck for DiskANN") {
        conf["emb_list_strategy"] = "muvera";
        conf["muvera_num_projections"] = 3;
        conf["muvera_num_repeats"] = 5;
        conf["muvera_seed"] = 42;

        auto status = knowhere::IndexStaticFaced<knowhere::fp32>::ConfigCheck(knowhere::IndexEnum::INDEX_DISKANN,
                                                                              version, conf, msg);
        REQUIRE(status != knowhere::Status::success);
        REQUIRE(msg.find("TokenANN") != std::string::npos);
    }

    SECTION("LEMUR with fp32 should fail ConfigCheck for DiskANN") {
        conf["emb_list_strategy"] = "lemur";
        conf["lemur_hidden_dim"] = 32;

        auto status = knowhere::IndexStaticFaced<knowhere::fp32>::ConfigCheck(knowhere::IndexEnum::INDEX_DISKANN,
                                                                              version, conf, msg);
        REQUIRE(status != knowhere::Status::success);
        REQUIRE(msg.find("TokenANN") != std::string::npos);
    }

    SECTION("TokenANN with fp32 should pass ConfigCheck for DiskANN") {
        conf["emb_list_strategy"] = "tokenann";

        auto status = knowhere::IndexStaticFaced<knowhere::fp32>::ConfigCheck(knowhere::IndexEnum::INDEX_DISKANN,
                                                                              version, conf, msg);
        REQUIRE(status == knowhere::Status::success);
    }

    SECTION("No strategy (default tokenann) with fp32 should pass ConfigCheck for DiskANN") {
        auto status = knowhere::IndexStaticFaced<knowhere::fp32>::ConfigCheck(knowhere::IndexEnum::INDEX_DISKANN,
                                                                              version, conf, msg);
        REQUIRE(status == knowhere::Status::success);
    }
}

#endif  // !KNOWHERE_WITH_CARDINAL
