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

#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "simd/instruction_set.h"
#include "simd/sparse_simd.h"

using namespace knowhere::sparse;

// Helper function to generate random sparse posting list data for testing
struct PostingListTestData {
    std::vector<uint32_t> doc_ids;
    std::vector<float> doc_vals;
    size_t n_docs;

    PostingListTestData(size_t posting_list_size, size_t n_docs_total, int seed = 12345) : n_docs(n_docs_total) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
        std::uniform_int_distribution<uint32_t> doc_dist(0, n_docs_total - 1);

        // Generate unique sorted document IDs
        std::set<uint32_t> unique_ids;
        while (unique_ids.size() < posting_list_size && unique_ids.size() < n_docs_total) {
            unique_ids.insert(doc_dist(gen));
        }

        for (uint32_t doc_id : unique_ids) {
            doc_ids.push_back(doc_id);
            doc_vals.push_back(val_dist(gen));
        }
    }
};

// Scalar reference implementation for testing
void
accumulate_posting_list_ip_scalar_ref(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                                      float* scores) {
    for (size_t i = 0; i < list_size; ++i) {
        scores[doc_ids[i]] += q_weight * doc_vals[i];
    }
}

static std::vector<uint32_t>
membership_reference(const std::vector<uint32_t>& terms, const std::vector<uint32_t>& query) {
    std::vector<uint32_t> positions(query.size(), std::numeric_limits<uint32_t>::max());
    for (size_t i = 0; i < query.size(); ++i) {
        const auto it = std::lower_bound(terms.begin(), terms.end(), query[i]);
        if (it != terms.end() && *it == query[i]) {
            positions[i] = static_cast<uint32_t>(it - terms.begin());
        }
    }
    return positions;
}

TEST_CASE("DSP hybrid SIMD membership matches scalar", "[sparse simd avx512][dsp]") {
#if defined(__x86_64__) || defined(_M_X64)
    if (!faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512BW()) {
        SKIP("AVX512BW not available on this CPU");
    }

    auto check = [](const std::vector<uint32_t>& terms, const std::vector<uint32_t>& query) {
        const auto expected = membership_reference(terms, query);
        std::vector<uint32_t> actual(query.size());
        find_terms_hybrid_avx512(terms.data(), static_cast<uint32_t>(terms.size()), query.data(),
                                 static_cast<uint32_t>(query.size()), actual.data());
        REQUIRE(actual == expected);
    };

    SECTION("edge cases") {
        check({}, {});
        check({}, {1, 2, 3});

        std::vector<uint32_t> terms(80);
        std::iota(terms.begin(), terms.end(), 100);
        check(terms, terms);                 // all match
        check(terms, {164, 165, 178, 179});  // all hits live in the final partial chunk

        terms[15] = 115;
        terms[16] = 115;  // duplicate spanning a chunk boundary
        check(terms, {114, 115, 115, 116});
    }

    SECTION("randomized sorted intersections") {
        std::mt19937 rng(20260720);
        for (int iteration = 0; iteration < 500; ++iteration) {
            const uint32_t term_count = 64 + rng() % 512;
            const uint32_t query_count = rng() % 17;
            std::set<uint32_t> term_set;
            while (term_set.size() < term_count) term_set.insert(rng() % 100000);
            std::set<uint32_t> query_set;
            while (query_set.size() < query_count) query_set.insert(rng() % 100000);
            check(std::vector<uint32_t>(term_set.begin(), term_set.end()),
                  std::vector<uint32_t>(query_set.begin(), query_set.end()));
        }
    }
#else
    SKIP("Test only runs on x86_64 platforms");
#endif
}

TEST_CASE("Test Sparse SIMD AVX512 - Basic Correctness", "[sparse simd avx512]") {
#if defined(__x86_64__) || defined(_M_X64)
    if (!faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512F()) {
        SKIP("AVX512 not available on this CPU");
    }

    const float tolerance = 0.0001f;
    const size_t n_docs = 1000;
    const float q_weight = 2.5f;

    SECTION("Various posting list sizes") {
        // Test different posting list sizes to cover SIMD boundaries
        auto plist_size = GENERATE(0, 1, 7, 15, 16, 17, 31, 32, 33, 47, 48, 49, 64, 100, 256, 1000);

        PostingListTestData test_data(plist_size, n_docs);

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        accumulate_posting_list_ip_scalar_ref(test_data.doc_ids.data(), test_data.doc_vals.data(),
                                              test_data.doc_ids.size(), q_weight, ref_scores.data());

        accumulate_posting_list_ip_avx512(test_data.doc_ids.data(), test_data.doc_vals.data(), test_data.doc_ids.size(),
                                          q_weight, avx512_scores.data());

        REQUIRE(avx512_scores.size() == ref_scores.size());
        for (size_t i = 0; i < ref_scores.size(); ++i) {
            if (std::abs(ref_scores[i]) < 1e-6f && std::abs(avx512_scores[i]) < 1e-6f) {
                // Both are effectively zero
                continue;
            }
            REQUIRE_THAT(avx512_scores[i], Catch::Matchers::WithinRel(ref_scores[i], tolerance));
        }
    }
#else
    SKIP("Test only runs on x86_64 platforms");
#endif
}

TEST_CASE("Test Sparse SIMD AVX512 - SIMD Boundary Cases", "[sparse simd avx512]") {
#if defined(__x86_64__) || defined(_M_X64)
    if (!faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512F()) {
        SKIP("AVX512 not available on this CPU");
    }

    const float tolerance = 0.0001f;
    const size_t n_docs = 500;
    const float q_weight = 1.5f;

    SECTION("Exactly at SIMD width boundaries") {
        // Test exact multiples of 16 (SIMD_WIDTH)
        auto plist_size = GENERATE(16, 32, 48, 64, 80, 96, 112, 128);

        PostingListTestData test_data(plist_size, n_docs, 54321);

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        accumulate_posting_list_ip_scalar_ref(test_data.doc_ids.data(), test_data.doc_vals.data(),
                                              test_data.doc_ids.size(), q_weight, ref_scores.data());

        accumulate_posting_list_ip_avx512(test_data.doc_ids.data(), test_data.doc_vals.data(), test_data.doc_ids.size(),
                                          q_weight, avx512_scores.data());

        for (size_t i = 0; i < ref_scores.size(); ++i) {
            if (std::abs(ref_scores[i]) < 1e-6f && std::abs(avx512_scores[i]) < 1e-6f) {
                continue;
            }
            REQUIRE_THAT(avx512_scores[i], Catch::Matchers::WithinRel(ref_scores[i], tolerance));
        }
    }

    SECTION("One element before/after SIMD boundaries") {
        // Test sizes around SIMD boundaries to ensure tail handling works
        auto plist_size = GENERATE(15, 17, 31, 33, 47, 49, 63, 65);

        PostingListTestData test_data(plist_size, n_docs, 98765);

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        accumulate_posting_list_ip_scalar_ref(test_data.doc_ids.data(), test_data.doc_vals.data(),
                                              test_data.doc_ids.size(), q_weight, ref_scores.data());

        accumulate_posting_list_ip_avx512(test_data.doc_ids.data(), test_data.doc_vals.data(), test_data.doc_ids.size(),
                                          q_weight, avx512_scores.data());

        for (size_t i = 0; i < ref_scores.size(); ++i) {
            if (std::abs(ref_scores[i]) < 1e-6f && std::abs(avx512_scores[i]) < 1e-6f) {
                continue;
            }
            REQUIRE_THAT(avx512_scores[i], Catch::Matchers::WithinRel(ref_scores[i], tolerance));
        }
    }
#else
    SKIP("Test only runs on x86_64 platforms");
#endif
}

TEST_CASE("Test Sparse SIMD AVX512 - Edge Cases", "[sparse simd avx512]") {
#if defined(__x86_64__) || defined(_M_X64)
    if (!faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512F()) {
        SKIP("AVX512 not available on this CPU");
    }

    const float tolerance = 0.0001f;
    const size_t n_docs = 100;
    const float q_weight = 3.0f;

    SECTION("Empty posting list") {
        PostingListTestData test_data(0, n_docs);

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        accumulate_posting_list_ip_scalar_ref(test_data.doc_ids.data(), test_data.doc_vals.data(),
                                              test_data.doc_ids.size(), q_weight, ref_scores.data());

        accumulate_posting_list_ip_avx512(test_data.doc_ids.data(), test_data.doc_vals.data(), test_data.doc_ids.size(),
                                          q_weight, avx512_scores.data());

        for (size_t i = 0; i < n_docs; ++i) {
            REQUIRE(avx512_scores[i] == 0.0f);
        }
    }

    SECTION("Single element posting list") {
        PostingListTestData test_data(1, n_docs, 11111);

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        accumulate_posting_list_ip_scalar_ref(test_data.doc_ids.data(), test_data.doc_vals.data(),
                                              test_data.doc_ids.size(), q_weight, ref_scores.data());

        accumulate_posting_list_ip_avx512(test_data.doc_ids.data(), test_data.doc_vals.data(), test_data.doc_ids.size(),
                                          q_weight, avx512_scores.data());

        for (size_t i = 0; i < ref_scores.size(); ++i) {
            if (std::abs(ref_scores[i]) < 1e-6f && std::abs(avx512_scores[i]) < 1e-6f) {
                continue;
            }
            REQUIRE_THAT(avx512_scores[i], Catch::Matchers::WithinRel(ref_scores[i], tolerance));
        }
    }

    SECTION("Very small posting lists (< 16 elements)") {
        auto small_size = GENERATE(2, 3, 5, 7, 11, 13, 15);
        PostingListTestData test_data(small_size, n_docs, 22222);

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        accumulate_posting_list_ip_scalar_ref(test_data.doc_ids.data(), test_data.doc_vals.data(),
                                              test_data.doc_ids.size(), q_weight, ref_scores.data());

        accumulate_posting_list_ip_avx512(test_data.doc_ids.data(), test_data.doc_vals.data(), test_data.doc_ids.size(),
                                          q_weight, avx512_scores.data());

        for (size_t i = 0; i < ref_scores.size(); ++i) {
            if (std::abs(ref_scores[i]) < 1e-6f && std::abs(avx512_scores[i]) < 1e-6f) {
                continue;
            }
            REQUIRE_THAT(avx512_scores[i], Catch::Matchers::WithinRel(ref_scores[i], tolerance));
        }
    }
#else
    SKIP("Test only runs on x86_64 platforms");
#endif
}

TEST_CASE("Test Sparse SIMD AVX512 - Special Values", "[sparse simd avx512]") {
#if defined(__x86_64__) || defined(_M_X64)
    if (!faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512F()) {
        SKIP("AVX512 not available on this CPU");
    }

    const float tolerance = 0.0001f;
    const size_t n_docs = 200;

    SECTION("Zero query weight") {
        PostingListTestData test_data(64, n_docs, 33333);
        const float q_weight = 0.0f;

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        accumulate_posting_list_ip_scalar_ref(test_data.doc_ids.data(), test_data.doc_vals.data(),
                                              test_data.doc_ids.size(), q_weight, ref_scores.data());

        accumulate_posting_list_ip_avx512(test_data.doc_ids.data(), test_data.doc_vals.data(), test_data.doc_ids.size(),
                                          q_weight, avx512_scores.data());

        for (size_t i = 0; i < n_docs; ++i) {
            REQUIRE(std::abs(avx512_scores[i]) < tolerance);
        }
    }

    SECTION("Large posting lists (stress test)") {
        // Test with very large posting lists to stress the 2x unrolled loop
        auto large_size = GENERATE(500, 1000);
        PostingListTestData test_data(large_size, n_docs, 55555);
        const float q_weight = 1.8f;

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        accumulate_posting_list_ip_scalar_ref(test_data.doc_ids.data(), test_data.doc_vals.data(),
                                              test_data.doc_ids.size(), q_weight, ref_scores.data());

        accumulate_posting_list_ip_avx512(test_data.doc_ids.data(), test_data.doc_vals.data(), test_data.doc_ids.size(),
                                          q_weight, avx512_scores.data());

        for (size_t i = 0; i < ref_scores.size(); ++i) {
            if (std::abs(ref_scores[i]) < 1e-6f && std::abs(avx512_scores[i]) < 1e-6f) {
                continue;
            }
            REQUIRE_THAT(avx512_scores[i], Catch::Matchers::WithinRel(ref_scores[i], tolerance));
        }
    }
#else
    SKIP("Test only runs on x86_64 platforms");
#endif
}

TEST_CASE("Test DSP Superblock-Major UB Accumulation", "[sparse simd avx512][dsp]") {
#if defined(__x86_64__) || defined(_M_X64)
    if (!faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512BW()) {
        SKIP("AVX512BW not available on this CPU");
    }

    constexpr uint32_t stride = 64;
    constexpr uint32_t n_superblocks = 3;
    constexpr uint32_t n_terms = 7;
    const std::vector<uint32_t> surviving_superblocks = {0, 2};

    std::mt19937 generator(24680);
    std::uniform_int_distribution<uint32_t> max_distribution(0, 255);
    std::uniform_int_distribution<uint32_t> weight_distribution(1, 255);
    std::vector<std::vector<uint8_t>> rows(n_terms, std::vector<uint8_t>(stride * n_superblocks));
    std::vector<const uint8_t*> row_pointers;
    std::vector<uint16_t> weights(n_terms);
    for (uint32_t term = 0; term < n_terms; ++term) {
        for (auto& value : rows[term]) {
            value = static_cast<uint8_t>(max_distribution(generator));
        }
        row_pointers.push_back(rows[term].data());
        weights[term] = static_cast<uint16_t>(weight_distribution(generator));
    }

    for (const uint16_t threshold : {uint16_t{0}, uint16_t{1234}, uint16_t{30000}, uint16_t{65534}, uint16_t{65535}}) {
        CAPTURE(threshold);
        std::vector<uint16_t> expected(stride * n_superblocks, 1234);
        std::vector<uint16_t> actual = expected;
        std::vector<uint64_t> expected_masks(n_superblocks, 0xdeadbeef);
        std::vector<uint64_t> actual_masks = expected_masks;
        accumulate_dense_block_ubs_scalar(expected.data(), expected_masks.data(), threshold, row_pointers.data(),
                                          weights.data(), n_terms, surviving_superblocks.data(),
                                          surviving_superblocks.size(), stride);
        accumulate_dense_block_ubs_avx512(actual.data(), actual_masks.data(), threshold, row_pointers.data(),
                                          weights.data(), n_terms, surviving_superblocks.data(),
                                          surviving_superblocks.size(), stride);

        REQUIRE(actual == expected);
        REQUIRE(actual_masks == expected_masks);
        for (uint32_t spb : surviving_superblocks) {
            const auto stripe_begin = expected.begin() + spb * stride;
            uint64_t expected_mask = 0;
            for (uint32_t lane = 0; lane < stride; ++lane) {
                expected_mask |= static_cast<uint64_t>(stripe_begin[lane] > threshold) << lane;
            }
            REQUIRE(expected_masks[spb] == expected_mask);
        }
        REQUIRE(actual_masks[1] == 0xdeadbeef);
        for (uint32_t lane = stride; lane < 2 * stride; ++lane) {
            REQUIRE(actual[lane] == 1234);
        }
    }
#else
    SKIP("Test only runs on x86_64 platforms");
#endif
}

TEST_CASE("Test Sparse SIMD AVX512 - Multiple Accumulations", "[sparse simd avx512]") {
#if defined(__x86_64__) || defined(_M_X64)
    if (!faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512F()) {
        SKIP("AVX512 not available on this CPU");
    }

    const float tolerance = 0.0001f;
    const size_t n_docs = 1000;

    SECTION("Accumulate multiple posting lists") {
        // Test accumulating contributions from multiple posting lists (simulating multiple query terms)
        std::vector<PostingListTestData> posting_lists;
        std::vector<float> query_weights = {2.5f, -1.3f, 0.8f, 3.2f, -0.5f};

        for (size_t i = 0; i < query_weights.size(); ++i) {
            posting_lists.emplace_back(64 + i * 10, n_docs, 10000 + i * 1000);
        }

        std::vector<float> ref_scores(n_docs, 0.0f);
        std::vector<float> avx512_scores(n_docs, 0.0f);

        for (size_t i = 0; i < posting_lists.size(); ++i) {
            accumulate_posting_list_ip_scalar_ref(posting_lists[i].doc_ids.data(), posting_lists[i].doc_vals.data(),
                                                  posting_lists[i].doc_ids.size(), query_weights[i], ref_scores.data());

            accumulate_posting_list_ip_avx512(posting_lists[i].doc_ids.data(), posting_lists[i].doc_vals.data(),
                                              posting_lists[i].doc_ids.size(), query_weights[i], avx512_scores.data());
        }

        for (size_t i = 0; i < ref_scores.size(); ++i) {
            if (std::abs(ref_scores[i]) < 1e-6f && std::abs(avx512_scores[i]) < 1e-6f) {
                continue;
            }
            REQUIRE_THAT(avx512_scores[i], Catch::Matchers::WithinRel(ref_scores[i], tolerance));
        }
    }
#else
    SKIP("Test only runs on x86_64 platforms");
#endif
}
