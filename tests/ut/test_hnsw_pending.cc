// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/cppcontrib/knowhere/impl/HnswSearcher.h>
#include <faiss/cppcontrib/knowhere/utils/Bitset.h>

#include <set>

#include "catch2/catch_test_macros.hpp"
#include "index/hnsw/impl/DummyVisitor.h"

TEST_CASE("HNSW retains filtered bridges hidden by a tighter result threshold", "[hnsw_pending]") {
    namespace fk = faiss::cppcontrib::knowhere;
    fk::NeighborSetDoublePopList candidates(1);
    fk::IteratorMinHeap pending;
    candidates.insert({0, 10, fk::Neighbor::kValid}, &pending);
    REQUIRE(candidates.pop().id == 0);
    candidates.insert({1, 5, fk::Neighbor::kInvalid}, &pending);
    candidates.insert({2, 1, fk::Neighbor::kValid}, &pending);
    REQUIRE(candidates.pop().id == 2);
    REQUIRE_FALSE(candidates.has_next());
    candidates.save_pending(pending);
    REQUIRE(pending.top().id == 1);
}

TEST_CASE("HNSW pending traversal reaches a vertex behind a filtered bridge", "[hnsw_pending]") {
    namespace fk = faiss::cppcontrib::knowhere;
    // 0 -> [1(filtered), 2], 1 -> 3. After accepting 2 the threshold drops
    // below 1's distance. Vertex 3 is reachable only through the saved bridge.
    fk::HNSW graph(2);
    graph.entry_point = 0;
    graph.max_level = 0;
    graph.levels.assign(4, 1);
    graph.offsets = {0, 4, 8, 12, 16};
    graph.neighbors.resize(16);
    std::fill(graph.neighbors.data(), graph.neighbors.data() + 16, -1);
    graph.neighbors[0] = 1;
    graph.neighbors[1] = 2;
    graph.neighbors[4] = 3;
    struct Distances : faiss::DistanceComputer {
        float
        operator()(faiss::idx_t id) override {
            const float d[] = {10, 5, 1, 0.5f};
            return d[id];
        }
        void
        set_query(const float*) override {
        }
        float
        symmetric_dis(faiss::idx_t, faiss::idx_t) override {
            return 0;
        }
    } dc;
    struct Filter {
        bool
        is_member(faiss::idx_t id) const {
            return id != 1;
        }
    } filter;
    knowhere::DummyVisitor visitor;
    auto visited = fk::Bitset::create_cleared(4);
    visited.set(0);
    fk::v2_hnsw_searcher<faiss::DistanceComputer, knowhere::DummyVisitor, fk::Bitset, Filter> searcher(
        graph, dc, visitor, visited, filter, 1.0f, nullptr);
    fk::NeighborSetDoublePopList candidates(1);
    candidates.insert({0, 10, fk::Neighbor::kValid});
    fk::IteratorMinHeap pending;
    searcher.search_on_a_level(candidates, 0, &pending);
    REQUIRE_FALSE(visited.get(3));
    std::set<unsigned> emitted{candidates[0].id};
    float alpha = 1;
    while (!pending.empty()) {
        const auto next = pending.top();
        pending.pop();
        searcher.evaluate_single_node(next.id, 0, alpha, [&](fk::Neighbor item) {
            pending.push(item);
            return true;
        });
        if (filter.is_member(next.id))
            emitted.insert(next.id);
    }
    REQUIRE(visited.get(3));
    REQUIRE(emitted == std::set<unsigned>{0, 2, 3});
}
