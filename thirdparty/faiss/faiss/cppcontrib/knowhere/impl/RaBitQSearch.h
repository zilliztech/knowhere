/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the MIT license in thirdparty/faiss/LICENSE.
 *
 * Port of Faiss #5526 (d8a85956) bounded traversal.
 * Graph adjacency is read from Knowhere without copying or changing the graph.
 * Extended to L2/IP/COSINE, deliberately limited to unfiltered KNN.
 * Callers must dispatch filtered/visitor requests to a compatible searcher.
 */
#pragma once

#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <faiss/utils/distances.h>
#include <cmath>
#include <functional>

namespace faiss::cppcontrib::knowhere::rabitq_search {
struct SearchStats {
    size_t estimate = 0, refine = 0, expanded = 0, upper_full = 0, upper_expanded = 0;
    bool exhausted = false;
};

template <class VT>
SearchStats search_one(const faiss::cppcontrib::knowhere::HNSW& graph,
                  faiss::RaBitQDistanceComputer& rq, VT& vt,
                  faiss::ResultHandler& res, int ef, bool relative,
                  bool similarity = false, const float* norms = nullptr,
                  float query_inverse_norm = 1) {
    SearchStats stats;
    using HC = faiss::CMax<float, int32_t>;
    int32_t nearest = graph.entry_point;
    auto scale = [&](int32_t id) { return norms ? norms[id] * query_inverse_norm : 1.f; };
    auto convert = [&](int32_t id, float raw) { return (similarity ? -raw : raw) * scale(id); };
    float nearest_distance = convert(nearest, rq(nearest));
    ++stats.upper_full;
    for (int level = graph.max_level; level >= 1; --level) {
        for (;;) {
            ++stats.upper_expanded;
            const int32_t previous = nearest;
            size_t begin, end;
            graph.neighbor_range(nearest, level, &begin, &end);
            int32_t ids[4];
            int count = 0;
            auto update = [&](int32_t id, float d) {
                d = convert(id, d);
                if (d < nearest_distance) { nearest = id; nearest_distance = d; }
            };
            for (size_t j = begin; j < end && graph.neighbors[j] >= 0; ++j) {
                ids[count++] = graph.neighbors[j];
                ++stats.upper_full;
                if (count == 4) {
                    float d[4];
                    rq.distances_batch_4(ids[0], ids[1], ids[2], ids[3], d[0], d[1], d[2], d[3]);
                    for (int i = 0; i < 4; ++i) update(ids[i], d[i]);
                    count = 0;
                }
            }
            for (int i = 0; i < count; ++i) update(ids[i], rq(ids[i]));
            if (previous == nearest) break;
        }
    }

    faiss::MinimaxHeapT<HC> candidates(ef);
    candidates.push(nearest, nearest_distance);
    vt.reserve(ef);
    if (nearest_distance < res.threshold) res.add_result(nearest_distance, nearest);
    vt.set(nearest);
    while (candidates.size() > 0) {
        float d0;
        const int32_t node = candidates.pop_min(&d0);
        if (relative && candidates.count_below(d0) >= ef) break;
        size_t begin, end;
        graph.neighbor_range(node, 0, &begin, &end);
        size_t limit = begin;
        for (size_t j = begin; j < end; ++j) {
            if (graph.neighbors[j] < 0) break;
            vt.prefetch(graph.neighbors[j]);
            ++limit;
        }
        int32_t ids[4];
        int count = 0;
        float threshold = res.threshold;
        auto evaluate = [&] {
            for (int i = 0; i < count; ++i) {
                const auto* code = rq.codes + static_cast<size_t>(ids[i]) * rq.code_size;
                const float estimate = rq.distance_to_code_1bit(code);
                ++stats.estimate;
                const auto* factors = reinterpret_cast<const faiss::rabitq_utils::SignBitFactorsWithError*>(
                    code + (rq.d + 7) / 8);
                const float s = scale(ids[i]);
                const float error = factors->f_error * rq.g_error;
                float distance = estimate;
                const bool refine = similarity ? (estimate + error) * s > -threshold
                                               : std::max(0.f, estimate - error) < threshold;
                if (refine) {
                    distance = rq.distance_to_code_full(code);
                    ++stats.refine;
                }
                distance = (similarity ? -distance : distance) * s;
                if (distance < threshold && res.add_result(distance, ids[i])) threshold = res.threshold;
                candidates.push(ids[i], distance);
            }
        };
        for (size_t j = begin; j < limit; ++j) {
            ids[count] = graph.neighbors[j];
            count += vt.set(ids[count]) ? 1 : 0;
            if (count == 4) { evaluate(); count = 0; }
        }
        if (count) evaluate();
        ++stats.expanded;
        if (!relative && stats.expanded > static_cast<size_t>(ef)) break;
    }
    stats.exhausted = candidates.size() == 0;
    return stats;
}

inline void search(const faiss::cppcontrib::knowhere::IndexHNSWRaBitQ& index,
                   faiss::idx_t n, const float* x, faiss::idx_t k, float* distances,
                   faiss::idx_t* labels, int ef, bool relative,
                   const faiss::RaBitQSearchParameters* params = nullptr,
                   const std::function<void(const SearchStats&)>& on_query = {}) {
    FAISS_THROW_IF_NOT(index.metric_type == faiss::METRIC_L2 ||
                      index.metric_type == faiss::METRIC_INNER_PRODUCT);
    const bool similarity = index.metric_type == faiss::METRIC_INNER_PRODUCT;
    const auto* cosine = dynamic_cast<const faiss::cppcontrib::knowhere::IndexHNSWRaBitQCosine*>(&index);
    const float* norms = cosine ? cosine->get_inverse_l2_norms() : nullptr;
    FAISS_THROW_IF_NOT(index.rabitq_index()->rabitq.nb_bits > 1);
    auto raw = std::unique_ptr<faiss::FlatCodesDistanceComputer>(
        params ? index.rabitq_index()->get_quantized_distance_computer(params->qb, params->centered)
               : index.rabitq_index()->get_FlatCodesDistanceComputer());
    auto& rq = dynamic_cast<faiss::RaBitQDistanceComputer&>(*raw);
    // Reuse Faiss's visited table across queries without clearing the full
    // table on each search; advance its generation after processing the query.
    auto& vt = faiss::VisitedTable::get_reusable(index.ntotal);
    faiss::HeapBlockResultHandler<faiss::CMax<float, int64_t>> block(n, distances, labels, k);
    decltype(block)::SingleResultHandler result(block);
    std::vector<float> rotated(index.d);
    for (faiss::idx_t i = 0; i < n; ++i) {
        result.begin(i);
        index.pretransform_index()->chain[0]->apply_noalloc(1, x + i * index.d, rotated.data());
        rq.set_query(rotated.data());
        const float norm2 = norms ? faiss::fvec_norm_L2sqr(x + i * index.d, index.d) : 1.f;
        const float query_inverse_norm = norm2 > 0 ? 1.f / std::sqrt(norm2) : 1.f;
        SearchStats stats;
        if (auto* vector = dynamic_cast<faiss::VisitedTableVector*>(&vt))
            stats = search_one(index.hnsw, rq, *vector, result, std::max<int>(ef, k), relative,
                               similarity, norms, query_inverse_norm);
        else
            stats = search_one(index.hnsw, rq, dynamic_cast<faiss::VisitedTableSet&>(vt), result,
                               std::max<int>(ef, k), relative, similarity, norms, query_inverse_norm);
        result.end();
        vt.advance();
        if (on_query) on_query(stats);
    }
}
} // namespace faiss::cppcontrib::knowhere::rabitq_search
