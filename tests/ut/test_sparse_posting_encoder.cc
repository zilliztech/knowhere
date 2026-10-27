// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "index/sparse/block_inverted_index.h"
#include "index/sparse/codec/adaptive.h"
#include "index/sparse/codec/maskedvbyte.h"
#include "index/sparse/codec/streamvbyte.h"

namespace sparse = knowhere::sparse::inverted;

namespace {

std::array<sparse::BlockCodecPtr, 3>
Codecs() {
    return {std::make_shared<sparse::StreamVByteBlockCodec>(), std::make_shared<sparse::MaskedVByteBlockCodec>(),
            std::make_shared<sparse::AdaptiveBlockCodec>()};
}

template <sparse::IndexScorerType Metric, typename Value>
void
RequireRoundTrip(const sparse::BlockCodecPtr& codec, std::span<const uint32_t> ids, std::span<const Value> values,
                 sparse::PostingEncodeWorkspace& workspace) {
    CAPTURE(codec->get_name(), ids.size());
    sparse::EncodePostingList<Metric, Value>(codec, ids, values, workspace);
    const auto first_encoding = workspace.encoded;
    sparse::EncodePostingList<Metric, Value>(codec, ids, values, workspace);
    REQUIRE(workspace.encoded == first_encoding);
    workspace.encoded.resize(workspace.encoded.size() + 16, 0);  // Ordinary decoder's SIMD tail guard.
    using DecodedValue = std::conditional_t<Metric == sparse::IndexScorerType::IP, Value, uint32_t>;
    const uint32_t universe = ids.back() + 1;
    sparse::BlockInvertedIndexCursor<DecodedValue> cursor(codec, workspace.encoded.data(), universe, {});
    for (size_t i = 0; i < ids.size(); ++i) {
        REQUIRE(cursor.vec_id() == ids[i]);
        REQUIRE(cursor.val() == static_cast<DecodedValue>(values[i]));
        cursor.next();
    }
    REQUIRE(cursor.vec_id() == universe);
}

}  // namespace

TEST_CASE("Sparse shared posting encoder round trips block boundaries and reuses workspace",
          "[sparse][posting-encoder]") {
    for (const auto& codec : Codecs()) {
        sparse::PostingEncodeWorkspace workspace;
        // The final singleton also exercises clearing a previously larger output.
        for (const uint32_t count : {1U, 2U, 255U, 256U, 257U, 512U, 513U, 1U}) {
            std::vector<uint32_t> ids(count);
            std::vector<float> floats(count);
            std::vector<knowhere::fp16> halves(count);
            std::vector<uint16_t> shorts(count);
            std::vector<uint32_t> integers(count);
            constexpr std::array<uint32_t, 8> tfs{1,        65535,    16777215,       16777216,
                                                  16777217, 16777218, UINT32_MAX - 1, UINT32_MAX};
            for (size_t i = 0; i < count; ++i) {
                ids[i] = static_cast<uint32_t>(7 + i * 3);
                floats[i] = 0.25F * (i + 1);
                halves[i] = static_cast<knowhere::fp16>(floats[i]);
                shorts[i] = i % 2 == 0 ? 1 : UINT16_MAX;
                integers[i] = tfs[i % tfs.size()];
            }
            RequireRoundTrip<sparse::IndexScorerType::IP, float>(codec, ids, floats, workspace);
            RequireRoundTrip<sparse::IndexScorerType::IP, knowhere::fp16>(codec, ids, halves, workspace);
            RequireRoundTrip<sparse::IndexScorerType::BM25, uint16_t>(codec, ids, shorts, workspace);
            RequireRoundTrip<sparse::IndexScorerType::BM25, uint32_t>(codec, ids, integers, workspace);
        }
        const std::array<uint32_t, 1> ids{7};
        for (const uint32_t tf : {1U, 16777216U, 16777217U, 16777218U, UINT32_MAX - 1, UINT32_MAX}) {
            const std::array<uint32_t, 1> values{tf};
            RequireRoundTrip<sparse::IndexScorerType::BM25, uint32_t>(codec, ids, values, workspace);
        }
    }
}

TEST_CASE("Sparse shared posting encoder preserves singleton wire bytes", "[sparse][posting-encoder]") {
    const auto codec = std::make_shared<sparse::AdaptiveBlockCodec>();
    const std::array<uint32_t, 1> ids{7};
    sparse::PostingEncodeWorkspace workspace;
    const std::array<float, 1> ip{3.5F};
    sparse::EncodePostingList<sparse::IndexScorerType::IP, float>(codec, ids, ip, workspace);
    REQUIRE(workspace.encoded == std::vector<uint8_t>{0x80, 7, 0, 0, 0, 0, 0, 0x60, 0x40});
    for (const auto tf : {1U, 2U, 128U}) {
        const std::array<uint32_t, 1> values{tf};
        sparse::EncodePostingList<sparse::IndexScorerType::BM25, uint32_t>(codec, ids, values, workspace);
        REQUIRE(workspace.encoded == std::vector<uint8_t>{0x80, 7, 0, 0, 0, static_cast<uint8_t>(0x80 | (tf - 1))});
    }
}

TEST_CASE("Sparse shared posting encoder preserves BM25 U16 clamping", "[sparse][posting-encoder]") {
    // U16 TF - 1 promotes to signed int. Preserve the old clamp of -1 to zero,
    // which decodes to TF == 1, without routing the subtraction through float.
    for (const auto& codec : Codecs()) {
        for (const uint32_t count : {1U, 257U}) {
            CAPTURE(codec->get_name(), count);
            std::vector<uint32_t> ids(count);
            for (uint32_t i = 0; i < count; ++i) {
                ids[i] = i;
            }
            const std::vector<uint16_t> values(count, 0);
            sparse::PostingEncodeWorkspace workspace;
            sparse::EncodePostingList<sparse::IndexScorerType::BM25, uint16_t>(codec, ids, values, workspace);
            workspace.encoded.resize(workspace.encoded.size() + 16, 0);
            sparse::BlockInvertedIndexCursor<uint32_t> cursor(codec, workspace.encoded.data(), count, {});
            for (uint32_t i = 0; i < count; ++i) {
                REQUIRE(cursor.vec_id() == i);
                REQUIRE(cursor.val() == 1);
                cursor.next();
            }
            REQUIRE(cursor.vec_id() == count);
        }
    }
}

TEST_CASE("Sparse ordinary BM25 build preserves quantized U32 term frequencies", "[sparse][posting-encoder]") {
    // Each input is representable as f32. Re-converting integer TF - 1 through f32
    // loses one at 16777218 and can wrap UINT32_MAX back to zero during decoding.
    constexpr std::array<float, 5> raw_tfs{1.0F, 65535.0F, 16777216.0F, 16777218.0F, 4294967296.0F};
    for (const auto& codec : Codecs()) {
        for (const uint32_t count : {1U, 257U}) {
            for (const float tf : raw_tfs) {
                CAPTURE(codec->get_name(), count, tf);
                std::vector<knowhere::sparse::SparseRow<float>> rows;
                for (uint32_t i = 0; i < count; ++i) {
                    rows.emplace_back(std::vector<std::pair<uint32_t, float>>{{7, tf}});
                }
                sparse::BlockInvertedIndex<float, uint32_t, sparse::IndexScorerType::BM25> index(codec);
                sparse::IndexScorerConfig scorer{};
                scorer.scorer_type = sparse::IndexScorerType::BM25;
                scorer.scorer_params.bm25 = {1.2F, 0.75F, tf};
                index.set_build_scorer(scorer);
                REQUIRE(index.add(rows.data(), rows.size(), 8) == knowhere::Status::success);
                const auto expected_tf = sparse::get_quant_val<float, uint32_t>(tf);
                auto cursor = index.get_dim_plist_cursor(0, {});
                for (uint32_t i = 0; i < count; ++i) {
                    REQUIRE(cursor.vec_id() == i);
                    REQUIRE(cursor.val() == expected_tf);
                    cursor.next();
                }
                REQUIRE(cursor.vec_id() == count);
            }
        }
    }
}
