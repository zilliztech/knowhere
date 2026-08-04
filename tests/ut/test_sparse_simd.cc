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

#include <random>
#include <set>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "index/sparse/codec/simd_bitpacking_kernel.h"
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

std::vector<uint8_t>
simdcomp_pack_scalar_reference(const uint32_t* input, size_t block_count, uint32_t bits) {
    constexpr size_t values_per_block = 128;
    constexpr size_t lanes = 4;
    constexpr size_t values_per_lane = values_per_block / lanes;
    const size_t words_per_block = static_cast<size_t>(bits) * lanes;
    std::vector<uint32_t> packed_words(block_count * words_per_block, 0);

    for (size_t block = 0; block < block_count; ++block) {
        const size_t input_base = block * values_per_block;
        const size_t output_base = block * words_per_block;
        for (size_t lane = 0; lane < lanes; ++lane) {
            for (size_t value_index = 0; value_index < values_per_lane; ++value_index) {
                const uint32_t value = input[input_base + value_index * lanes + lane];
                const size_t bit_offset = value_index * bits;
                const size_t word_index = bit_offset / 32;
                const uint32_t shift = static_cast<uint32_t>(bit_offset % 32);
                packed_words[output_base + word_index * lanes + lane] |= value << shift;
                if (shift + bits > 32) {
                    packed_words[output_base + (word_index + 1) * lanes + lane] |= value >> (32 - shift);
                }
            }
        }
    }

    std::vector<uint8_t> packed(packed_words.size() * sizeof(uint32_t));
    std::memcpy(packed.data(), packed_words.data(), packed.size());
    return packed;
}

TEST_CASE("Test simdcomp bit-packing kernels", "[sparse][simd][simdcomp]") {
    constexpr uint32_t input_guard = 0xa5a55a5aU;
    constexpr uint32_t output_guard = 0xdeadbeefU;
    constexpr uint8_t byte_guard = 0x6d;
    constexpr size_t values_per_block = 128;
    std::mt19937 rng(20260730);

    for (const size_t block_count : {size_t{1}, size_t{2}, size_t{4}}) {
        const size_t value_count = block_count * values_per_block;
        for (uint32_t bits = 1; bits <= 32; ++bits) {
            CAPTURE(block_count, bits);
            const uint32_t mask = bits == 32 ? std::numeric_limits<uint32_t>::max() : (uint32_t{1} << bits) - 1;
            std::vector<uint32_t> input(value_count + 2, input_guard);
            for (size_t i = 0; i < value_count; ++i) {
                input[i + 1] = rng() & mask;
            }
            input[1] = 0;
            input[2] = mask;
            input[3] = uint32_t{1} << (bits - 1);
            input[value_count] = mask;

            const auto expected = simdcomp_pack_scalar_reference(input.data() + 1, block_count, bits);
            std::vector<uint8_t> packed(expected.size() + 2, byte_guard);
            knowhere_simd_pack_128_blocks(input.data() + 1, packed.data() + 1, block_count, bits);
            REQUIRE(packed.front() == byte_guard);
            REQUIRE(packed.back() == byte_guard);
            REQUIRE(std::equal(expected.begin(), expected.end(), packed.begin() + 1));
            REQUIRE(input.front() == input_guard);
            REQUIRE(input.back() == input_guard);

            std::vector<uint32_t> unpacked(value_count + 2, output_guard);
            knowhere_simd_unpack_128_blocks(packed.data() + 1, unpacked.data() + 1, block_count, bits);
            REQUIRE(std::equal(input.begin() + 1, input.end() - 1, unpacked.begin() + 1));
            REQUIRE(unpacked.front() == output_guard);
            REQUIRE(unpacked.back() == output_guard);

            if (bits < 32) {
                constexpr uint32_t previous_doc_id = 17;
                std::vector<uint32_t> doc_ids(value_count + 2, output_guard);
                knowhere_simd_unpack_d1_128_blocks(packed.data() + 1, doc_ids.data() + 1, block_count, bits,
                                                   previous_doc_id);
                uint32_t expected_doc_id = previous_doc_id;
                for (size_t i = 0; i < value_count; ++i) {
                    expected_doc_id += input[i + 1] + 1;
                    REQUIRE(doc_ids[i + 1] == expected_doc_id);
                }
                REQUIRE(doc_ids.front() == output_guard);
                REQUIRE(doc_ids.back() == output_guard);
            }
        }
    }
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
