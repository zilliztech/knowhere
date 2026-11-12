// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/sparse/sindi_bm25_u4.h"
#include "index/sparse/sindi_filter.h"
#include "index/sparse/sindi_packed12.h"
#include "index/sparse/sindi_simd.h"
#include "index/sparse/sparse_index_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace sindi = knowhere::sparse::inverted::sindi;

TEST_CASE("SINDI packed prefilter preserves search across windows and persistence", "[sparse][sindi][prefilter]") {
    const auto quant = GENERATE(std::string("u4_lut_u12"), std::string("u4_lut_u16"), std::string("e5m7"));
    const bool ip = quant == "e5m7";
    const bool growable = ip ? GENERATE(false, true) : false;
    const bool refine = ip ? GENERATE(false, true) : false;
    const size_t window = quant == "u4_lut_u16" ? 8192 : 4096;
    const size_t rows = 2 * window + 19;
    std::vector<std::map<int32_t, float>> documents(rows);
    for (size_t id = 0; id < rows; ++id) {
        documents[id] = {{0, 1.003f + float(id % 199) / 17}};
        if (id % 3 == 0) {
            documents[id][1] = 1.007f + float(id % 97) / 7;
        }
        if (id % 997 == 5) {
            documents[id][2] = 17.125f;
        }
    }
    // Odd posting counts and a term absent from the second window exercise
    // cross-term packed offsets, sparse window cursors, and partial tails.
    documents[3][3] = 23.75f;
    documents[rows - 1][3] = 31.5f;
    if (!ip) {
        for (auto& document : documents) {
            for (auto& [dim, value] : document) {
                value = std::round(value * 8);
            }
        }
    }
    auto data = GenSparseDataSet(documents, 4);
    auto query =
        GenSparseDataSet(std::vector<std::map<int32_t, float>>{{{0, 1.25f}, {1, 0.875f}, {2, 0.5f}, {3, 0.25f}}}, 4);
    knowhere::Json config = {{"metric_type", ip ? "IP" : "BM25"},
                             {"inverted_index_algo", "SINDI"},
                             {"quant_type", quant},
                             {"sindi_window_size", window},
                             {"dim_max_score_ratio", 1.0f},
                             {"refine", refine},
                             {"bm25_k1", 1.2},
                             {"bm25_b", 0.75},
                             {"bm25_avgdl", 100.0}};
    if (refine) {
        config["refine_k"] = 2.0f;
        config["refine_query_mass_percentage"] = 0.7f;
    }
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(growable ? knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC
                                                                : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
                                                       11)
                     .value();
    if (growable) {
        const auto split = documents.begin() + window - 1;
        auto initial = GenSparseDataSet(std::vector<std::map<int32_t, float>>(documents.begin(), split), 4);
        auto added = GenSparseDataSet(std::vector<std::map<int32_t, float>>(split, documents.end()), 4);
        REQUIRE(index.Build(initial, config) == knowhere::Status::success);
        REQUIRE(index.Add(added, config) == knowhere::Status::success);
    } else {
        REQUIRE(index.Build(data, config) == knowhere::Status::success);
    }
    knowhere::BinarySet serialized;
    if (!growable) {
        REQUIRE(index.Serialize(serialized) == knowhere::Status::success);
    }
    const auto field = ip ? "sindi_ip_filter_threshold" : "sindi_bm25_filter_threshold";
    const float default_threshold = ip ? 1.0f / 512 : 1.0f / 64;
    for (bool restored : {false, true}) {
        if (restored) {
            if (growable) {
                break;
            }
            REQUIRE(index.Deserialize(serialized, config) == knowhere::Status::success);
        }
        // Empty/all-pass/all-rejected masks, sparse survivors, exact dispatch
        // boundaries, offset/mapped IDs, and out-of-range exclusions.
        for (int pattern = 0; pattern < 9; ++pattern) {
            const size_t bit_count = pattern == 6 ? rows + 11 : (pattern == 8 ? rows - 7 : rows);
            std::vector<uint8_t> bits((bit_count + 7) / 8, 0);
            for (size_t id = 0; id < bit_count; ++id) {
                bool excluded = pattern == 2;
                if (pattern >= 3) {
                    excluded = id < window || (id % 997 != 5 && id != rows - 1);
                    if (pattern == 4 || pattern == 5) {
                        // One window at the default limit, then one above it.
                        excluded = id % window >= size_t(window * default_threshold) + (pattern == 5);
                    }
                }
                if (excluded) {
                    bits[id / 8] |= 1u << (id % 8);
                }
            }
            auto bitset = pattern == 0 ? knowhere::BitsetView{} : knowhere::BitsetView(bits.data(), bit_count);
            std::vector<int32_t> mapped_ids(rows);
            knowhere::IdArray mapping;
            if (pattern == 6) {
                bitset.set_id_offset(11);
            } else if (pattern == 7) {
                for (size_t i = 0; i < rows; ++i) {
                    mapped_ids[i] = i % 101 == 0 ? -1 : static_cast<int32_t>(rows - i - 1);
                }
                mapping.Set(mapped_ids.data(), mapped_ids.size());
                bitset.set_out_ids(mapping, mapped_ids.size());
            }
            bitset.set_vector_count(rows);
            // Full scoring without prefiltering supplies a per-ID oracle even
            // when the top-k boundary falls inside a group of quantized ties.
            config[field] = 0.0f;
            config["k"] = rows;
            auto reference = index.Search(query, config, bitset);
            REQUIRE(reference.has_value());
            std::vector<float> reference_scores(rows, std::numeric_limits<float>::quiet_NaN());
            for (size_t i = 0; i < rows; ++i) {
                const auto id = reference.value()->GetIds()[i];
                if (id >= 0) {
                    REQUIRE(static_cast<size_t>(id) < rows);
                    reference_scores[id] = reference.value()->GetDistance()[i];
                }
            }
            for (size_t k : {size_t{1}, size_t{10}, size_t{100}}) {
                config["k"] = k;
                config[field] = 0.0f;
                auto baseline = index.Search(query, config, bitset);
                REQUIRE(baseline.has_value());
                for (float threshold : {default_threshold, 1.0f / 96, 1.0f}) {
                    CAPTURE(quant, growable, refine, restored, pattern, k, threshold);
                    config[field] = threshold;
                    auto result = index.Search(query, config, bitset);
                    REQUIRE(result.has_value());
                    std::set<int64_t> seen;
                    for (size_t i = 0; i < k; ++i) {
                        const auto id = result.value()->GetIds()[i];
                        if (baseline.value()->GetIds()[i] >= 0) {
                            REQUIRE(id >= 0);
                            REQUIRE(static_cast<size_t>(id) < rows);
                            REQUIRE(seen.insert(id).second);
                            REQUIRE((bitset.empty() || !bitset.test(id)));
                            const auto score = result.value()->GetDistance()[i];
                            const auto expected = baseline.value()->GetDistance()[i];
                            // Compaction changes which postings use x86 SIMD
                            // versus scalar tails. Their rounding can reorder
                            // near ties; check rank scores and each ID's score.
                            REQUIRE(std::abs(score - expected) <= 1e-5f * std::max(1.0f, expected));
                            REQUIRE(std::isfinite(reference_scores[id]));
                            REQUIRE(std::abs(score - reference_scores[id]) <=
                                    1e-5f * std::max(1.0f, reference_scores[id]));
                        } else {
                            REQUIRE(id == -1);
                            REQUIRE(std::isnan(result.value()->GetDistance()[i]));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("SINDI U4 prefilter preserves ties between filtered and unfiltered windows", "[sparse][sindi][prefilter]") {
    const auto quant = GENERATE("u4_lut_u12", "u4_lut_u16");
    constexpr size_t window = 4096;
    std::vector<std::map<int32_t, float>> documents(2 * window, {{1, 1.0f}});
    documents[3][0] = documents[window + 3][0] = 1.0f;
    auto data = GenSparseDataSet(documents, 2);
    auto query = GenSparseDataSet(std::vector<std::map<int32_t, float>>{{{0, 1.0f}}}, 2);
    knowhere::Json config = {{"metric_type", "BM25"},
                             {"inverted_index_algo", "SINDI"},
                             {"quant_type", quant},
                             {"sindi_window_size", window},
                             {"k", 1},
                             {"bm25_k1", 1.2f},
                             {"bm25_b", 0.75f},
                             {"bm25_avgdl", 100.0f}};
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, 11)
                     .value();
    REQUIRE(index.Build(data, config) == knowhere::Status::success);
    std::vector<uint8_t> bits(2 * window / 8, 0);
    std::fill_n(bits.begin(), window / 8, 0xff);
    bits[0] &= ~(1u << 3);
    knowhere::BitsetView bitset(bits.data(), 2 * window);
    config["sindi_bm25_filter_threshold"] = 0.0f;
    auto baseline = index.Search(query, config, bitset);
    REQUIRE(baseline.has_value());
    // TF=1 and length=2 expose the denominator grouping difference between
    // SVE's packed U4 and unpacked U8 kernels. Only the first window is sparse.
    for (float threshold : {1.0f / 64, 1.0f}) {
        config["sindi_bm25_filter_threshold"] = threshold;
        auto result = index.Search(query, config, bitset);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] == baseline.value()->GetIds()[0]);
        REQUIRE(result.value()->GetDistance()[0] == baseline.value()->GetDistance()[0]);
    }
}

TEST_CASE("SINDI encoded selection decodes only surviving values", "[sparse][sindi][prefilter]") {
    const bool u16_ids = GENERATE(false, true);
    const size_t window = u16_ids ? 65535 : 4096;
    const size_t start = GENERATE(0, 1, 7);
    const size_t byte_offset = GENERATE(0, 1);
    std::vector<uint8_t> bits((window + 7) / 8, 0xff);
    for (size_t id = 5; id < window; id += 997) {
        bits[id / 8] &= ~(1u << (id % 8));
    }
    knowhere::BitsetView bitset(bits.data(), window);
    const auto check = [&]<typename Value>() {
        sindi::PostingFilter<Value> filter;
        REQUIRE(filter.prepare(bitset, 0, window, window / 64));
        // Exercise binary seeking, scanning, scratch reuse, odd packed phases,
        // unaligned serialized data, and the last partial byte without padding.
        for (size_t count : {window, size_t{0}, size_t{1}, size_t{7}, size_t{16}, size_t{31}, size_t{128}}) {
            CAPTURE(u16_ids, start, byte_offset, count);
            std::vector<uint8_t> id_bytes(byte_offset +
                                          (u16_ids ? 2 * (start + count) : sindi::packed12_bytes(start + count)));
            std::vector<uint8_t> value_bytes(byte_offset + (std::is_same_v<Value, knowhere::fp16>
                                                                ? sindi::packed12_bytes(start + count)
                                                                : (start + count + 1) / 2));
            auto* ids = id_bytes.data() + byte_offset;
            auto* values = value_bytes.data() + byte_offset;
            std::vector<uint16_t> expected_ids;
            std::vector<Value> expected_values;
            const uint8_t lut[16] = {0, 1, 3, 7, 11, 17, 23, 31, 41, 53, 67, 89, 113, 149, 191, 255};
            for (size_t i = 0; i < count; ++i) {
                const uint16_t id = i * window / count;
                if (u16_ids) {
                    std::memcpy(ids + 2 * (start + i), &id, sizeof(id));
                } else {
                    sindi::pack12(ids, start + i, id);
                }
                Value value;
                if constexpr (std::is_same_v<Value, knowhere::fp16>) {
                    // Multiples of 1/16 in this range are exactly representable.
                    value = knowhere::fp16(float(1 + i % 127) / 16);
                    sindi::pack12(values, start + i, sindi::encode_e5m7(value));
                } else {
                    const uint8_t code = i % 16;
                    values[(start + i) / 2] |= code << (4 * ((start + i) & 1));
                    value = lut[code];
                }
                if (!bitset.test(id)) {
                    expected_ids.push_back(id);
                    expected_values.push_back(value);
                }
            }
            const auto load_id = [&](size_t i) {
                REQUIRE(i < count);
                return u16_ids ? sindi::unpack_bm25_u16_id(ids, start + i) : sindi::unpack12(ids, start + i);
            };
            size_t decoded = 0;
            const auto selected = filter.select_decoded(bitset, count, load_id, [&](size_t i) -> Value {
                REQUIRE_FALSE(bitset.test(load_id(i)));
                ++decoded;
                if constexpr (std::is_same_v<Value, knowhere::fp16>) {
                    return sindi::decode_e5m7_half(sindi::unpack12(values, start + i));
                } else {
                    return lut[sindi::unpack_bm25_u4(values, start + i)];
                }
            });
            REQUIRE(decoded == expected_ids.size());
            REQUIRE(selected.count == expected_ids.size());
            for (size_t i = 0; i < selected.count; ++i) {
                REQUIRE(selected.ids[i] == expected_ids[i]);
                REQUIRE(selected.values[i] == expected_values[i]);
            }
        }
    };
    check.template operator()<uint8_t>();
    if (!u16_ids) {
        check.template operator()<knowhere::fp16>();
    }
}

TEST_CASE("SINDI posting filter thresholds are configurable search parameters", "[sparse][sindi][prefilter][config]") {
    knowhere::Json json = {{"metric_type", "IP"}, {"k", 10}};
    knowhere::SparseInvertedIndexConfig defaults;
    REQUIRE(knowhere::Config::Load(defaults, json, knowhere::SEARCH) == knowhere::Status::success);
    REQUIRE(defaults.sindi_bm25_filter_threshold.value() == 1.0f / 64);
    REQUIRE(defaults.sindi_ip_filter_threshold.value() == 1.0f / 512);

    const auto field = GENERATE("sindi_bm25_filter_threshold", "sindi_ip_filter_threshold");
    for (float threshold : {0.0f, 1.0f / 96, 0.5f, 1.0f}) {
        CAPTURE(field, threshold);
        json[field] = threshold;
        knowhere::SparseInvertedIndexConfig config;
        REQUIRE(knowhere::Config::Load(config, json, knowhere::SEARCH) == knowhere::Status::success);
        REQUIRE(config.sindi_bm25_filter_threshold.value() == json.value("sindi_bm25_filter_threshold", 1.0f / 64));
        REQUIRE(config.sindi_ip_filter_threshold.value() == json.value("sindi_ip_filter_threshold", 1.0f / 512));
    }
    for (float threshold : {-0.01f, 1.01f, std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        CAPTURE(field, threshold);
        json[field] = threshold;
        knowhere::SparseInvertedIndexConfig config;
        REQUIRE(knowhere::Config::Load(config, json, knowhere::SEARCH) != knowhere::Status::success);
    }
}

TEST_CASE("SINDI compaction preserves eligible posting IDs and values", "[sparse][sindi][prefilter]") {
    const size_t window_size = GENERATE(63, 64, 65, 1024, 4096, 65535);
    const size_t start = GENERATE(0, 7, 65535);
    const bool unaligned = GENERATE(false, true);
    const auto mapping = GENERATE(0, 1, 2, 3);  // direct, offset, mapped, short bitset
    const size_t offset = mapping == 1 ? 11 : 0;
    const size_t bits_count = start + offset + window_size - (mapping == 3 ? 3 : 0);
    std::vector<uint8_t> bits((bits_count + 7) / 8, 0xff);
    for (size_t id = 0; id < bits_count; ++id) {
        if (id % 37 == 5) {
            bits[id / 8] &= ~(1u << (id % 8));
        }
    }
    knowhere::BitsetView bitset(bits.data(), bits_count);
    bitset.set_id_offset(offset);
    bitset.set_vector_count(start + window_size);
    std::vector<int32_t> mapped_ids(bits_count);
    knowhere::IdArray mapping_array;
    if (mapping == 2) {
        for (size_t i = 0; i < bits_count; ++i) {
            mapped_ids[i] = i % 101 == 0 ? -1 : static_cast<int32_t>(bits_count - i - 1);
        }
        mapping_array.Set(mapped_ids.data(), mapped_ids.size());
        bitset.set_out_ids(mapping_array, mapped_ids.size());
    }

    sindi::PostingFilter<uint16_t> filter;
    REQUIRE(filter.prepare(bitset, start, window_size, window_size / 8));
    // Exercise both long lists and short/tail lists, including a second term
    // reusing the scratch after a much larger first term.
    for (size_t count : {window_size, size_t{0}, size_t{1}, size_t{7}, size_t{16}, size_t{31}}) {
        CAPTURE(window_size, start, mapping, count);
        std::vector<uint16_t> ids(count), values(count);
        std::vector<uint16_t> expected_ids, expected_values;
        for (size_t i = 0; i < count; ++i) {
            ids[i] = static_cast<uint16_t>(i * window_size / count);
            values[i] = static_cast<uint16_t>(i * 17 + 1);
            if (!bitset.test(start + ids[i])) {
                expected_ids.push_back(ids[i]);
                expected_values.push_back(values[i]);
            }
        }
        std::vector<uint8_t> id_bytes(count * sizeof(uint16_t) + 1);
        std::vector<uint8_t> value_bytes(count * sizeof(uint16_t) + 1);
        if (count != 0) {
            std::memcpy(id_bytes.data() + 1, ids.data(), count * sizeof(uint16_t));
            std::memcpy(value_bytes.data() + 1, values.data(), count * sizeof(uint16_t));
        }
        const auto* id_ptr = unaligned ? reinterpret_cast<const uint16_t*>(id_bytes.data() + 1) : ids.data();
        const auto* value_ptr = unaligned ? reinterpret_cast<const uint16_t*>(value_bytes.data() + 1) : values.data();
        const auto original = filter.select(bitset, {id_ptr, value_ptr, count}, false);
        REQUIRE(original.ids == id_ptr);
        REQUIRE(original.values == value_ptr);
        REQUIRE(original.count == count);
        const auto selected = filter.select(bitset, original, true);
        REQUIRE(selected.count == expected_ids.size());
        for (size_t i = 0; i < selected.count; ++i) {
            REQUIRE(selected.ids[i] == expected_ids[i]);
            REQUIRE(selected.values[i] == expected_values[i]);
        }
    }
}

TEST_CASE("SINDI prefilter selects paths from each window", "[sparse][sindi][prefilter]") {
    constexpr size_t window = 1024;
    const size_t divisor = GENERATE(64, 512);
    const size_t max_eligible = window / divisor;
    std::vector<uint8_t> bits(3 * window / 8, 0);
    std::fill_n(bits.begin(), window / 8, 0xff);
    bits[0] &= ~1u;
    knowhere::BitsetView bitset(bits.data(), 3 * window);
    sindi::PostingFilter<uint8_t> filter;
    REQUIRE(filter.prepare(bitset, 0, window, max_eligible));
    REQUIRE_FALSE(filter.prepare(bitset, window, window, max_eligible));
    REQUIRE_FALSE(filter.prepare({}, 0, window, max_eligible));
    // A wholly rejected window remains empty, even after previous scratch use.
    std::fill(bits.begin(), bits.end(), 0xff);
    REQUIRE(filter.prepare(bitset, 2 * window, window, max_eligible));
    const uint16_t id = 0;
    const uint8_t value = 255;
    REQUIRE(filter.select(bitset, {&id, &value, 1}, true).empty());
    // Disabling selection must still return the original view after filtering
    // produced an empty view; it must not expose the previous scratch state.
    const auto original = filter.select({}, {&id, &value, 1}, false);
    REQUIRE(original.ids == &id);
    REQUIRE(original.values == &value);
    REQUIRE(original.count == 1);

    // The exact limit takes the filtered path; one more survivor takes SIMD
    // directly. Spread survivors across the window to exercise word counting.
    for (size_t i = 0; i < max_eligible; ++i) {
        const auto allowed = i * divisor;
        bits[allowed / 8] &= ~(1u << (allowed % 8));
    }
    REQUIRE(filter.prepare(bitset, 0, window, max_eligible));
    bits[(window - 1) / 8] &= ~(1u << ((window - 1) % 8));
    REQUIRE_FALSE(filter.prepare(bitset, 0, window, max_eligible));
}

TEST_CASE("SINDI compacted kernels only accumulate eligible postings", "[sparse][sindi][prefilter][simd]") {
    constexpr size_t window = 1024;
    const size_t count = GENERATE(1, 7, 8, 15, 16, 31, 63, 128, 1024);
    std::vector<uint8_t> bits(window / 8, 0xff);
    for (size_t i = 3; i < window; i += 19) {
        bits[i / 8] &= ~(1u << (i % 8));
    }
    knowhere::BitsetView bitset(bits.data(), window);
    std::vector<uint16_t> ids(count);
    std::vector<float> row_sums(window);
    for (size_t i = 0; i < window; ++i) {
        row_sums[i] = 300 + i;
    }
    for (size_t i = 0; i < count; ++i) {
        ids[i] = i * window / count;
    }
    const auto check = [&]<typename Value>(auto accumulate) {
        std::vector<Value> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<Value>(1 + i % 251);
        }
        sindi::PostingFilter<Value> filter;
        REQUIRE(filter.prepare(bitset, 0, window, window / 8));
        const auto selected = filter.select(bitset, {ids.data(), values.data(), count}, true);
        std::vector<float> actual(window, 0.25f);
        const float maximum = accumulate(selected.values, selected.ids, selected.count, actual.data());
        float expected_max = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto id = ids[i];
            float expected = 0.25f;
            if (!bitset.test(id)) {
                if constexpr (std::is_same_v<Value, knowhere::fp16>) {
                    expected += 1.25f * static_cast<float>(values[i]);
                } else {
                    const float tf = static_cast<float>(values[i]);
                    expected += 1.25f * 2.2f * tf / (tf + 0.3f + 0.003f * row_sums[id]);
                }
                expected_max = std::max(expected_max, expected);
            }
            REQUIRE(std::abs(actual[id] - expected) <= 1e-5f * std::max(1.0f, expected));
        }
        REQUIRE(std::abs(maximum - expected_max) <= 1e-5f * std::max(1.0f, expected_max));
    };
    check.template operator()<knowhere::fp16>([&](auto values, auto ids, auto count, auto out) {
        return sindi::get_ip_kernels().accumulate(1.25f, values, ids, count, out);
    });
    check.template operator()<uint16_t>([&](auto values, auto ids, auto count, auto out) {
        return sindi::get_bm25_kernels().accumulate(1.25f, values, ids, count, out, 1.2f, 0.75f, 300, row_sums.data());
    });
    check.template operator()<uint8_t>([&](auto values, auto ids, auto count, auto out) {
        return sindi::get_bm25_u8_kernels().accumulate(1.25f, values, ids, count, out, 1.2f, 0.75f, 300,
                                                       row_sums.data());
    });
}

TEST_CASE("SINDI filtered search matches stored-value scores across windows", "[sparse][sindi][prefilter]") {
    const auto quant = GENERATE(std::string("fp16"), std::string("e5m7"), std::string("u16"), std::string("u8"));
    const bool ip = quant == "fp16" || quant == "e5m7";
    const bool growable = quant == "u8" ? false : GENERATE(false, true);
    // Exercise the default, disabled, custom and always-prefilter paths on the
    // same stored-value oracle, including growing and deserialized indexes.
    const float filter_threshold = GENERATE(-1.0f, 0.0f, 1.0f / 96, 1.0f);
    const size_t window = quant == "e5m7" ? 4096 : 1024;
    const size_t rows = 3 * window + 19;
    std::vector<std::map<int32_t, float>> documents(rows);
    std::vector<float> oracle(rows);
    for (size_t id = 0; id < rows; ++id) {
        documents[id] = {{0, 1.003f + float(id % 97) * 0.137f}, {1, 2.7f + float(id % 59) * 0.413f}};
        if (id % 11 == 0) {
            documents[id][1] = ip ? 31.007f : 256.7f + id % 701;
        }
        if (id == window + 5) {
            documents[id][1] = ip ? 32.3f : 66000.4f;
        }
        if (id % 4 == 0) {
            documents[id][2] = 4.3f;
        }
        // Only one window contains dimension 3, exercising sparse window NNZ
        // encoding and a U8 sidecar entry after other terms' posting ranges.
        if (id == 1041) {
            documents[id][3] = ip ? 8.125f : 701.2f;
        }
        float length = 0;
        for (auto [dim, value] : documents[id]) {
            length += value;
        }
        for (auto [dim, value] : documents[id]) {
            const float weight = dim == 0 ? 1.25f : (dim == 1 ? 0.875f : (dim == 2 ? 0.5f : 0.25f));
            if (ip) {
                auto represented = knowhere::fp16(value);
                if (quant == "e5m7") {
                    // Independent stored-value oracle: E5M7 discards the low
                    // three fraction bits of the nonnegative FP16 value.
                    const auto bits = std::bit_cast<uint16_t>(represented);
                    represented = std::bit_cast<knowhere::fp16>(uint16_t(bits & 0xfff8));
                }
                oracle[id] += weight * float(represented);
            } else {
                const float tf = static_cast<uint16_t>(std::min(value, 65535.0f));
                oracle[id] += weight * 2.2f * tf / (tf + 0.3f + 0.009f * length);
            }
        }
    }
    auto data = GenSparseDataSet(documents, 4);
    auto query =
        GenSparseDataSet(std::vector<std::map<int32_t, float>>{{{0, 1.25f}, {1, 0.875f}, {2, 0.5f}, {3, 0.25f}}}, 4);
    knowhere::Json config = {{"metric_type", ip ? "IP" : "BM25"},
                             {"inverted_index_algo", "SINDI"},
                             {"quant_type", quant},
                             {"sindi_window_size", window},
                             {"dim_max_score_ratio", 1.05},
                             {"bm25_k1", 1.2},
                             {"bm25_b", 0.75},
                             {"bm25_avgdl", 100.0}};
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(growable ? knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC
                                                                : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
                                                       11)
                     .value();
    if (growable) {
        const auto split = documents.begin() + window + 7;
        auto initial = GenSparseDataSet(std::vector<std::map<int32_t, float>>(documents.begin(), split), 4);
        auto added = GenSparseDataSet(std::vector<std::map<int32_t, float>>(split, documents.end()), 4);
        REQUIRE(index.Build(initial, config) == knowhere::Status::success);
        REQUIRE(index.Add(added, config) == knowhere::Status::success);
    } else {
        REQUIRE(index.Build(data, config) == knowhere::Status::success);
    }
    knowhere::BinarySet serialized;
    if (!growable) {
        REQUIRE(index.Serialize(serialized) == knowhere::Status::success);
    }
    if (filter_threshold >= 0) {
        config[ip ? "sindi_ip_filter_threshold" : "sindi_bm25_filter_threshold"] = filter_threshold;
    }

    for (bool restored : {false, true}) {
        if (restored) {
            if (growable) {
                break;  // Growable indexes do not support serialization.
            }
            REQUIRE(index.Deserialize(serialized, config) == knowhere::Status::success);
        }
        for (int pattern = 0; pattern < 8; ++pattern) {
            const size_t bit_count = pattern == 5 ? rows + 11 : (pattern == 7 ? rows - 7 : rows);
            std::vector<uint8_t> bits((bit_count + 7) / 8, 0);
            for (size_t id = 0; id < bit_count; ++id) {
                const bool sparse_excluded = id % 997 != 5 && (ip || id != window + 5) && id != 1041 && id != rows - 2;
                const bool excluded =
                    pattern == 1 || (pattern == 2 && sparse_excluded) ||
                    (pattern == 3 && (id < window || (id < 2 * window ? sparse_excluded : id % 4 == 0))) ||
                    (pattern >= 5 && sparse_excluded);
                if (excluded) {
                    bits[id / 8] |= 1u << (id % 8);
                }
            }
            knowhere::BitsetView bitset =
                pattern == 4 ? knowhere::BitsetView{} : knowhere::BitsetView(bits.data(), bit_count);
            std::vector<int32_t> mapped_ids(rows);
            knowhere::IdArray mapping_array;
            if (pattern == 5) {
                bitset.set_id_offset(11);
            } else if (pattern == 6) {
                for (size_t i = 0; i < rows; ++i) {
                    mapped_ids[i] = i % 101 == 0 ? -1 : static_cast<int32_t>(rows - i - 1);
                }
                mapping_array.Set(mapped_ids.data(), mapped_ids.size());
                bitset.set_out_ids(mapping_array, mapped_ids.size());
            }
            bitset.set_vector_count(rows);
            std::vector<float> expected;
            for (size_t id = 0; id < rows; ++id) {
                if (bitset.empty() || !bitset.test(id)) {
                    expected.push_back(oracle[id]);
                }
            }
            std::sort(expected.begin(), expected.end(), std::greater<float>());
            // Iterator scoring uses get_all_distances(), which shares query
            // cursor initialization with top-k search but not its prefilter.
            if (filter_threshold < 0) {
                CAPTURE(quant, growable, restored, pattern);
                auto iterators = index.AnnIterator(query, config, bitset);
                REQUIRE(iterators.has_value());
                REQUIRE(iterators.value().size() == 1);
                auto& iterator = iterators.value().front();
                std::set<int64_t> seen;
                while (iterator->HasNext().value()) {
                    const auto [id, score] = iterator->Next().value();
                    REQUIRE(id >= 0);
                    REQUIRE(static_cast<size_t>(id) < rows);
                    REQUIRE((bitset.empty() || !bitset.test(id)));
                    REQUIRE(seen.insert(id).second);
                    REQUIRE(std::abs(score - oracle[id]) <= 1e-5f * std::max(1.0f, oracle[id]));
                }
                REQUIRE(seen.size() == expected.size());
            }
            for (const size_t k : {size_t{10}, size_t{100}, rows + 3}) {
                CAPTURE(quant, growable, restored, pattern, k, filter_threshold);
                config["k"] = k;
                auto result = index.Search(query, config, bitset);
                REQUIRE(result.has_value());
                std::set<int64_t> seen;
                for (size_t i = 0; i < k; ++i) {
                    const auto id = result.value()->GetIds()[i];
                    const auto score = result.value()->GetDistance()[i];
                    if (i >= expected.size()) {
                        REQUIRE(id == -1);
                        REQUIRE(std::isnan(score));
                        continue;
                    }
                    REQUIRE(id >= 0);
                    REQUIRE(static_cast<size_t>(id) < rows);
                    REQUIRE((bitset.empty() || !bitset.test(id)));
                    REQUIRE(seen.insert(id).second);
                    REQUIRE(std::abs(score - oracle[id]) <= 1e-5f * std::max(1.0f, oracle[id]));
                    REQUIRE(std::abs(score - expected[i]) <= 1e-5f * std::max(1.0f, expected[i]));
                }
            }
        }
    }
}
