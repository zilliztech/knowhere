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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <faiss/cppcontrib/knowhere/utils/distances_if.h>
#include <faiss/cppcontrib/knowhere/utils/distances_typed.h>
#include "knowhere/operands.h"
#include "simd/distances_ref.h"
#include "simd/hook.h"

namespace {

template <typename DataType, typename ByIdxDistance, typename DirectDistance>
void
CheckByIdxDistancePositions(ByIdxDistance by_idx_distance, DirectDistance direct_distance) {
    constexpr size_t dim = 3;
    constexpr size_t database_size = 32;
    const std::array<int64_t, 5> selected_ids = {5, 9, 17, 23, 31};
    const std::array<DataType, dim> query = {DataType(2.0f), DataType(-3.0f), DataType(5.0f)};
    std::vector<DataType> database(database_size * dim);
    for (size_t row = 0; row < database_size; ++row) {
        database[row * dim] = DataType(static_cast<float>(row % 11) - 5.0f);
        database[row * dim + 1] = DataType(static_cast<float>((row * 3) % 13) - 6.0f);
        database[row * dim + 2] = DataType(static_cast<float>((row * 5) % 17) - 8.0f);
    }

    std::vector<int64_t> result_ids;
    std::vector<float> result_distances;
    by_idx_distance(
            query.data(),
            database.data(),
            selected_ids.data(),
            dim,
            selected_ids.size(),
            [](size_t) -> std::optional<bool> { return true; },
            [&](float distance, int64_t id) {
                result_ids.push_back(id);
                result_distances.push_back(distance);
            });

    REQUIRE(result_ids == std::vector<int64_t>{0, 1, 2, 3, 4});
    REQUIRE(result_distances.size() == selected_ids.size());
    for (size_t i = 0; i < selected_ids.size(); ++i) {
        const auto id = selected_ids[i];
        const float expected = direct_distance(query.data(), database.data() + id * dim, dim);
        REQUIRE_THAT(result_distances[i], Catch::Matchers::WithinAbs(expected, 0.001f));
    }
}

struct RecordingDistanceComputer : faiss::DistanceComputer {
    void
    set_query(const float*) override {
    }

    float
    operator()(faiss::idx_t id) override {
        return static_cast<float>(id);
    }

    float
    symmetric_dis(faiss::idx_t, faiss::idx_t) override {
        return 0.0f;
    }
};

}  // namespace

TEST_CASE("Test Distance Compute", "[distance]") {
    std::mt19937 rng;
    std::uniform_int_distribution<> distrib(1, 100000);
    std::uniform_real_distribution<float> fill_distrib(1, 1000000);
    std::string ins;
    faiss::cppcontrib::knowhere::fvec_hook(ins);

    using std::make_tuple;
    SECTION("Test Binary Distance Compute") {
        typedef float (*FUNC)(const float*, const float*, size_t);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::cppcontrib::knowhere::fvec_L1, faiss::cppcontrib::knowhere::fvec_L1_ref),
            make_tuple(faiss::cppcontrib::knowhere::fvec_L2sqr, faiss::cppcontrib::knowhere::fvec_L2sqr_ref),
            make_tuple(faiss::cppcontrib::knowhere::fvec_Linf, faiss::cppcontrib::knowhere::fvec_Linf_ref),
            make_tuple(faiss::cppcontrib::knowhere::fvec_inner_product,
                       faiss::cppcontrib::knowhere::fvec_inner_product_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }
            REQUIRE_THAT(real_func(a.data(), b.data(), len),
                         Catch::Matchers::WithinRel(gold_func(a.data(), b.data(), len), 0.001f));
        }
    }

    SECTION("Test Normal Compute") {
        typedef float (*FUNC)(const float*, size_t);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::cppcontrib::knowhere::fvec_norm_L2sqr, faiss::cppcontrib::knowhere::fvec_norm_L2sqr_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
            }
            REQUIRE_THAT(real_func(a.data(), len), Catch::Matchers::WithinRel(gold_func(a.data(), len), 0.001f));
        }
    }

    SECTION("Test Madd and Argmin") {
        typedef int (*FUNC)(size_t, const float*, float, const float*, float*);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::cppcontrib::knowhere::fvec_madd_and_argmin,
                       faiss::cppcontrib::knowhere::fvec_madd_and_argmin_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }

            std::vector<float> c(len);
            std::vector<float> c_gold(len);
            float pf = fill_distrib(rng);
            REQUIRE(real_func(len, a.data(), pf, b.data(), c.data()) ==
                    gold_func(len, a.data(), pf, b.data(), c_gold.data()));

            for (int i = 0; i < len; ++i) {
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(c_gold[i], 0.001f));
            }
        }
    }

    SECTION("Test Madd") {
        typedef void (*FUNC)(size_t, const float*, float, const float*, float*);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::cppcontrib::knowhere::fvec_madd, faiss::cppcontrib::knowhere::fvec_madd_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }

            std::vector<float> c(len);
            std::vector<float> c_gold(len);
            float pf = fill_distrib(rng);
            real_func(len, a.data(), pf, b.data(), c.data());
            gold_func(len, a.data(), pf, b.data(), c_gold.data());

            for (int i = 0; i < len; ++i) {
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(c_gold[i], 0.001f));
            }
        }
    }
}

TEST_CASE("Test typed by-idx distance callbacks use positions", "[distance][typed][by_idx]") {
    using namespace faiss::cppcontrib::knowhere;

    SECTION("INT8 inner product") {
        CheckByIdxDistancePositions<::knowhere::int8>(
                [](auto... args) { int8_vec_inner_products_ny_by_idx_if(args...); },
                int8_vec_inner_product);
    }

    SECTION("INT8 L2") {
        CheckByIdxDistancePositions<::knowhere::int8>(
                [](auto... args) { int8_vec_L2sqr_ny_by_idx_if(args...); },
                int8_vec_L2sqr);
    }

    SECTION("FP16 inner product") {
        CheckByIdxDistancePositions<::knowhere::fp16>(
                [](auto... args) { fp16_vec_inner_products_ny_by_idx_if(args...); },
                fp16_vec_inner_product);
    }

    SECTION("FP16 L2") {
        CheckByIdxDistancePositions<::knowhere::fp16>(
                [](auto... args) { fp16_vec_L2sqr_ny_by_idx_if(args...); },
                fp16_vec_L2sqr);
    }

    SECTION("BF16 inner product") {
        CheckByIdxDistancePositions<::knowhere::bf16>(
                [](auto... args) { bf16_vec_inner_products_ny_by_idx_if(args...); },
                bf16_vec_inner_product);
    }

    SECTION("BF16 L2") {
        CheckByIdxDistancePositions<::knowhere::bf16>(
                [](auto... args) { bf16_vec_L2sqr_ny_by_idx_if(args...); },
                bf16_vec_L2sqr);
    }
}

TEST_CASE("Test generic by-idx distance callbacks use positions", "[distance][by_idx]") {
    const std::array<faiss::idx_t, 3> ids = {5, 9, 17};
    RecordingDistanceComputer distance_computer;
    std::vector<faiss::idx_t> callback_positions;
    std::vector<float> distances;

    faiss::cppcontrib::knowhere::distance_compute_by_idx_if(
            ids.data(),
            ids.size(),
            &distance_computer,
            [](size_t) -> std::optional<bool> { return true; },
            [&](float distance, faiss::idx_t position) {
                distances.push_back(distance);
                callback_positions.push_back(position);
            });

    REQUIRE(callback_positions == std::vector<faiss::idx_t>{0, 1, 2});
    REQUIRE(distances == std::vector<float>{5.0f, 9.0f, 17.0f});
}

TEST_CASE("Test typed selector results use database IDs", "[distance][typed][selector]") {
    using namespace faiss::cppcontrib::knowhere;
    constexpr size_t dim = 3;
    constexpr size_t database_size = 18;
    const std::array<int64_t, 3> selected_ids = {5, 9, 17};
    const std::array<::knowhere::int8, dim> query = {2, -3, 5};
    std::vector<::knowhere::int8> database(database_size * dim);
    for (size_t row = 0; row < database_size; ++row) {
        database[row * dim] = static_cast<::knowhere::int8>(row % 7 + 1);
        database[row * dim + 1] = static_cast<::knowhere::int8>(row % 5 + 2);
        database[row * dim + 2] = static_cast<::knowhere::int8>(row % 3 + 3);
    }
    faiss::IDSelectorArray selector(selected_ids.size(), selected_ids.data());

    auto check_results = [&](const std::array<float, 3>& distances,
                             const std::array<int64_t, 3>& result_ids,
                             auto expected_distance) {
        auto sorted_result_ids = result_ids;
        std::sort(sorted_result_ids.begin(), sorted_result_ids.end());
        REQUIRE(sorted_result_ids == selected_ids);
        for (size_t i = 0; i < result_ids.size(); ++i) {
            REQUIRE_THAT(distances[i], Catch::Matchers::WithinAbs(expected_distance(result_ids[i]), 0.001f));
        }
    };

    SECTION("inner product") {
        std::array<float, 3> distances;
        std::array<int64_t, 3> result_ids;
        knn_inner_product_typed(
                query.data(), database.data(), dim, 1, database_size, 3, distances.data(), result_ids.data(), &selector);
        check_results(distances, result_ids, [&](int64_t id) {
            return int8_vec_inner_product(query.data(), database.data() + id * dim, dim);
        });
    }

    SECTION("L2") {
        std::array<float, 3> distances;
        std::array<int64_t, 3> result_ids;
        knn_L2sqr_typed(query.data(),
                        database.data(),
                        dim,
                        1,
                        database_size,
                        3,
                        distances.data(),
                        result_ids.data(),
                        nullptr,
                        &selector);
        check_results(distances, result_ids, [&](int64_t id) {
            return int8_vec_L2sqr(query.data(), database.data() + id * dim, dim);
        });
    }

    SECTION("cosine") {
        std::vector<float> y_inv_norms(database_size);
        for (size_t id = 0; id < database_size; ++id) {
            y_inv_norms[id] = 1.0f / sqrtf(int8_vec_norm_L2sqr(database.data() + id * dim, dim));
        }
        const float x_inv_norm = 1.0f / sqrtf(int8_vec_norm_L2sqr(query.data(), dim));
        std::array<float, 3> distances;
        std::array<int64_t, 3> result_ids;
        knn_cosine_typed(query.data(),
                         database.data(),
                         y_inv_norms.data(),
                         dim,
                         1,
                         database_size,
                         3,
                         distances.data(),
                         result_ids.data(),
                         &selector);
        check_results(distances, result_ids, [&](int64_t id) {
            return int8_vec_inner_product(query.data(), database.data() + id * dim, dim) * x_inv_norm *
                    y_inv_norms[id];
        });
    }
}
