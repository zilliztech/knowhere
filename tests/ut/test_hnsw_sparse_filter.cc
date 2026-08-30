// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "index/hnsw/impl/IndexBruteForceWrapper.h"
#include "index/hnsw/impl/IndexConditionalWrapper.h"
#include "knowhere/bitsetview.h"
#include "knowhere/bitsetview_idselector.h"

TEST_CASE("Sparse filtered BF enumerates unaligned bitset windows exactly", "[faiss_hnsw][sparse_filter_bf]") {
    constexpr faiss::idx_t nb = 131;
    constexpr faiss::idx_t dim = 4;
    constexpr faiss::idx_t nq = 2;
    constexpr faiss::idx_t k = 5;

    std::vector<float> base(nb * dim);
    for (faiss::idx_t i = 0; i < nb; ++i) {
        base[i * dim] = static_cast<float>(i);
        base[i * dim + 1] = static_cast<float>((i * 7) % 19);
        base[i * dim + 2] = static_cast<float>((i * 11) % 23);
        base[i * dim + 3] = static_cast<float>((i * 13) % 29);
    }
    const std::vector<float> queries = {
        64.5f, 2.0f, 3.0f, 4.0f,
        128.0f, 5.0f, 7.0f, 11.0f,
    };
    const std::vector<faiss::idx_t> accepted_ids = {0, 7, 31, 63, 64, 65, 96, 127, 130};

    faiss::IndexFlatL2 flat(dim);
    flat.add(nb, base.data());
    knowhere::IndexBruteForceWrapper wrapper(&flat);

    for (const size_t id_offset : {size_t{0}, size_t{17}}) {
        const size_t public_bits = id_offset + static_cast<size_t>(nb);
        std::vector<uint8_t> bits((public_bits + 7) / 8, 0xff);
        for (const auto id : accepted_ids) {
            const size_t public_id = id_offset + static_cast<size_t>(id);
            bits[public_id >> 3] &= static_cast<uint8_t>(~(1U << (public_id & 7)));
        }

        knowhere::BitsetView bitset(bits.data(), public_bits, nb - accepted_ids.size());
        bitset.set_id_offset(id_offset);
        bitset.set_vector_count(nb);
        bitset.set_filter_count(nb - accepted_ids.size());
        knowhere::BitsetViewIDSelector selector(bitset);
        faiss::SearchParameters params;
        params.sel = &selector;

        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        wrapper.search(nq, queries.data(), k, distances.data(), labels.data(), &params);

        for (faiss::idx_t qi = 0; qi < nq; ++qi) {
            std::vector<std::pair<float, faiss::idx_t>> exact;
            for (const auto id : accepted_ids) {
                float distance = 0.0f;
                for (faiss::idx_t d = 0; d < dim; ++d) {
                    const float delta = queries[qi * dim + d] - base[id * dim + d];
                    distance += delta * delta;
                }
                exact.emplace_back(distance, id);
            }
            std::sort(exact.begin(), exact.end());
            for (faiss::idx_t rank = 0; rank < k; ++rank) {
                INFO("offset=" << id_offset << " query=" << qi << " rank=" << rank);
                REQUIRE(labels[qi * k + rank] == exact[rank].second);
                REQUIRE(distances[qi * k + rank] == Catch::Approx(exact[rank].first));
            }
        }
    }
}

TEST_CASE("Filtered HNSW planner scales exact fallback with population and ef", "[faiss_hnsw][sparse_filter_bf]") {
    constexpr size_t nb = 1'000'000;
    std::vector<uint8_t> bits((nb + 7) / 8, 0xff);

    faiss::IndexFlatL2 flat(4);
    flat.ntotal = nb;
    knowhere::FaissHnswConfig config;
    config.k = 10;
    config.ef = 128;

    SECTION("two percent pass stays on KAlpha") {
        const knowhere::BitsetView bitset(bits.data(), nb, 980'000);
        const auto decision = knowhere::WhetherPerformBruteForceSearch(&flat, config, bitset);
        REQUIRE(decision == std::optional<bool>(false));
    }

    SECTION("one percent pass uses sparse exact search at ef 128") {
        const knowhere::BitsetView bitset(bits.data(), nb, 990'000);
        const auto decision = knowhere::WhetherPerformBruteForceSearch(&flat, config, bitset);
        REQUIRE(decision == std::optional<bool>(true));
    }

    SECTION("lower ef keeps the latency-oriented approximate path") {
        config.ef = 32;
        const knowhere::BitsetView bitset(bits.data(), nb, 990'000);
        const auto decision = knowhere::WhetherPerformBruteForceSearch(&flat, config, bitset);
        REQUIRE(decision == std::optional<bool>(false));
    }

    SECTION("half percent pass uses sparse exact search even at ef 32") {
        config.ef = 32;
        const knowhere::BitsetView bitset(bits.data(), nb, 995'000);
        const auto decision = knowhere::WhetherPerformBruteForceSearch(&flat, config, bitset);
        REQUIRE(decision == std::optional<bool>(true));
    }
}
