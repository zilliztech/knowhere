// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

// standard headers
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

// Faiss-specific headers
#include <faiss/Index.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissException.h>
#include <faiss/cppcontrib/knowhere/impl/HNSW.h>
#include <faiss/cppcontrib/knowhere/impl/ResultHandler.h>
#include <faiss/utils/ordered_key_value.h>

// Knowhere-specific headers
#include <faiss/cppcontrib/knowhere/impl/Neighbor.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

namespace {

// whether to track statistics
constexpr bool track_hnsw_stats = true;

} // namespace

// Accomodates all the search logic and variables.
/// * DistanceComputerT is responsible for computing distances
/// * GraphVisitorT records visited edges
/// * VisitedT is responsible for tracking visited nodes
/// * FilterT is resposible for filtering unneeded nodes
/// Interfaces of all templates are tweaked to accept standard Faiss structures
///   with dynamic dispatching. Custom Knowhere structures are also accepted.
template <
        typename DistanceComputerT,
        typename GraphVisitorT,
        typename VisitedT,
        typename FilterT>
struct v2_hnsw_searcher {
    using storage_idx_t = faiss::cppcontrib::knowhere::HNSW::storage_idx_t;
    using idx_t = faiss::idx_t;

    // hnsw structure.
    // the reference is not owned.
    const faiss::cppcontrib::knowhere::HNSW& hnsw;

    // computes distances. it already knows the query vector.
    // the reference is not owned.
    DistanceComputerT& qdis;

    // records visited edges.
    // the reference is not owned.
    GraphVisitorT& graph_visitor;

    // tracks the nodes that have been visited already.
    // the reference is not owned.
    VisitedT& visited_nodes;

    // a filter for disabled nodes.
    // the reference is not owned.
    const FilterT& filter;

    // parameter for the filtering
    const float kAlpha;

    // Whether two groups of four distances should be pipelined for this
    // distance representation and filter-routing density.
    const bool use_distance_pipeline;

    // Whether exact-width 2/3-way distance kernels are available. If not,
    // preserve the original single-distance tail path.
    const bool use_tail_distance_batches;

    // Short-tail batching is representation-sensitive in the compact graph
    // loop: cosine benefits from shared norm/query work, while FlatL2 does not.
    const bool use_compact_tail_distance_batches;

    // Whether two pending groups of four can share one query-vector stream.
    const bool use_distance_batch8;

    // Whether filtered-out graph waypoints can use full-dimensional SQ8
    // scores. Such nodes are ineligible for results and use a separate batch
    // so valid candidates always retain exact fp32 distances.
    const bool use_approximate_routing_distance;

    // Whether graph-offset latency is exposed enough to benefit from a
    // speculative cache-line fetch. High-dimensional distance kernels opt out.
    const bool use_graph_offset_prefetch;

    // Whether to use predicate-aware adaptive local expansion.
    const bool use_adaptive_filter;


    // Per-query scratch space reused by adaptive expansion. Keeping these on
    // the searcher avoids several heap allocations for every popped node.
    std::vector<storage_idx_t> adaptive_pending_ids;
    std::vector<storage_idx_t> adaptive_pending_parents;
    std::vector<storage_idx_t> adaptive_bridges;
    std::vector<std::pair<float, storage_idx_t>> adaptive_ordered_first_hop;

    // custom parameters of HNSW search.
    // the pointer is not owned.
    const faiss::cppcontrib::knowhere::SearchParametersHNSW* params;

    //
    v2_hnsw_searcher(
            const faiss::cppcontrib::knowhere::HNSW& hnsw_,
            DistanceComputerT& qdis_,
            GraphVisitorT& graph_visitor_,
            VisitedT& visited_nodes_,
            const FilterT& filter_,
            const float kAlpha_,
            const faiss::cppcontrib::knowhere::SearchParametersHNSW* params_,
            const bool use_adaptive_filter_ = false)
            : hnsw{hnsw_},
              qdis{qdis_},
              graph_visitor{graph_visitor_},
              visited_nodes{visited_nodes_},
              filter{filter_},
              kAlpha{kAlpha_},
              use_distance_pipeline{qdis_.should_pipeline_distance_batches(kAlpha_)},
              use_tail_distance_batches{qdis_.supports_tail_distance_batches()},
              use_compact_tail_distance_batches{
                      qdis_.prefers_compact_tail_distance_batches()},
              use_distance_batch8{
                      qdis_.supports_distance_batch_8() &&
                      kAlpha_ <=
                              (qdis_.supports_approximate_routing_distance()
                                       ? 0.36f
                                       : 0.50f)},
              use_approximate_routing_distance{
                      qdis_.supports_approximate_routing_distance() &&
                      kAlpha_ >= 0.25f},
              use_graph_offset_prefetch{qdis_.should_prefetch_graph_offsets()},
              use_adaptive_filter{use_adaptive_filter_},
              params{params_} {
        if (use_adaptive_filter) {
            adaptive_pending_ids.reserve(64);
            adaptive_pending_parents.reserve(64);
            adaptive_bridges.reserve(32);
            adaptive_ordered_first_hop.reserve(32);
        }
    }

    v2_hnsw_searcher(const v2_hnsw_searcher&) = delete;
    v2_hnsw_searcher(v2_hnsw_searcher&&) = delete;
    v2_hnsw_searcher& operator=(const v2_hnsw_searcher&) = delete;
    v2_hnsw_searcher& operator=(v2_hnsw_searcher&&) = delete;

    // greedily update a nearest vector at a given level.
    // * the update starts from the value in 'nearest'.
    faiss::cppcontrib::knowhere::HNSWStats greedy_update_nearest(
            const int level,
            storage_idx_t& nearest,
            float& d_nearest) {
        faiss::cppcontrib::knowhere::HNSWStats stats;

        for (;;) {
            storage_idx_t prev_nearest = nearest;

            size_t begin = 0;
            size_t end = 0;
            hnsw.neighbor_range(nearest, level, &begin, &end);

            // prefetch and eval the size
            size_t count = 0;
            for (size_t i = begin; i < end; i++) {
                storage_idx_t v = hnsw.neighbors[i];
                if (v < 0) {
                    break;
                }

                // qdis.prefetch(v);
                count += 1;
            }

            // visit neighbors
            for (size_t i = begin; i < begin + count; i++) {
                storage_idx_t v = hnsw.neighbors[i];

                // compute the distance
                const float dis = qdis(v);

                // record a traversed edge
                graph_visitor.visit_edge(level, prev_nearest, nearest, dis);

                // check if an update is needed
                if (dis < d_nearest) {
                    nearest = v;
                    d_nearest = dis;
                }
            }

            // update stats
            if (track_hnsw_stats) {
                stats.ndis += count;
                stats.nhops += 1;
            }

            // we're done if there we no changes
            if (nearest == prev_nearest) {
                return stats;
            }
        }
    }

    template <typename FuncAddCandidate>
    __attribute__((noinline)) void evaluate_compact_eight(
            const idx_t node_id,
            const int level,
            const size_t* const saved_indices,
            const int* const saved_statuses,
            FuncAddCandidate& func_add_candidate) {
        float dis[8] = {};
        qdis.distances_batch_8(
                saved_indices[0],
                saved_indices[1],
                saved_indices[2],
                saved_indices[3],
                saved_indices[4],
                saved_indices[5],
                saved_indices[6],
                saved_indices[7],
                dis[0],
                dis[1],
                dis[2],
                dis[3],
                dis[4],
                dis[5],
                dis[6],
                dis[7]);
        for (size_t lane = 0; lane < 8; ++lane) {
            graph_visitor.visit_edge(
                    level, node_id, saved_indices[lane], dis[lane]);
            func_add_candidate(knowhere::Neighbor(
                    saved_indices[lane], dis[lane], saved_statuses[lane]));
        }
    }

    // Compact exact-distance path for low-dimensional vectors. Keep this
    // routine free of the high-dimensional SQ8/batch-8 machinery: on short
    // vectors, graph/queue latency and instruction-cache pressure dominate.
    template <typename FuncAddCandidate>
    faiss::cppcontrib::knowhere::HNSWStats evaluate_single_node_compact(
            const idx_t node_id,
            const int level,
            float& accumulated_alpha,
            FuncAddCandidate func_add_candidate) {
        faiss::cppcontrib::knowhere::HNSWStats stats;

        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(node_id, level, &begin, &end);
        if (begin < end) {
            __builtin_prefetch(&hnsw.neighbors[begin], 0, 1);
            if (end - begin > 16) {
                __builtin_prefetch(&hnsw.neighbors[begin + 16], 0, 1);
            }
        }

        size_t counter = 0;
        size_t saved_indices[8];
        int saved_statuses[8];

        auto evaluate_four = [&](const size_t offset) {
            float dis[4] = {0, 0, 0, 0};
            qdis.distances_batch_4(
                    saved_indices[offset],
                    saved_indices[offset + 1],
                    saved_indices[offset + 2],
                    saved_indices[offset + 3],
                    dis[0],
                    dis[1],
                    dis[2],
                    dis[3]);

            for (size_t lane = 0; lane < 4; ++lane) {
                const size_t pos = offset + lane;
                graph_visitor.visit_edge(
                        level, node_id, saved_indices[pos], dis[lane]);
                func_add_candidate(knowhere::Neighbor(
                        saved_indices[pos], dis[lane], saved_statuses[pos]));
            }
        };

        size_t ndis = 0;
        for (size_t j = begin; j < end; ++j) {
            const storage_idx_t v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            if (visited_nodes.get(v1)) {
                graph_visitor.visit_edge(level, node_id, v1, -1);
                continue;
            }
            visited_nodes.set(v1);

            int status = knowhere::Neighbor::kValid;
            if (!filter.is_member(v1)) {
                status = knowhere::Neighbor::kInvalid;
                accumulated_alpha += kAlpha;
                if (accumulated_alpha < 1.0f) {
                    continue;
                }
                accumulated_alpha -= 1.0f;
            }

            saved_indices[counter] = v1;
            saved_statuses[counter] = status;
            ++counter;
            ++ndis;

            if (counter == 4) {
                if (use_distance_pipeline) {
                    qdis.prefetch_batch_4(
                            saved_indices[0],
                            saved_indices[1],
                            saved_indices[2],
                            saved_indices[3]);
                } else {
                    evaluate_four(0);
                    counter = 0;
                }
            } else if (counter == 8) {
                qdis.prefetch_batch_4(
                        saved_indices[4],
                        saved_indices[5],
                        saved_indices[6],
                        saved_indices[7]);
                if (use_distance_batch8) {
                    evaluate_compact_eight(
                            node_id,
                            level,
                            saved_indices,
                            saved_statuses,
                            func_add_candidate);
                } else {
                    evaluate_four(0);
                    evaluate_four(4);
                }
                counter = 0;
            }
        }

        size_t processed = 0;
        if (counter >= 4) {
            evaluate_four(0);
            processed = 4;
        }
        const size_t tail = counter - processed;
        if (use_tail_distance_batches && tail >= 2) {
            float dis[3] = {0, 0, 0};
            if (tail == 3) {
                qdis.distances_batch_3(
                        saved_indices[processed],
                        saved_indices[processed + 1],
                        saved_indices[processed + 2],
                        dis[0],
                        dis[1],
                        dis[2]);
            } else {
                qdis.distances_batch_2(
                        saved_indices[processed],
                        saved_indices[processed + 1],
                        dis[0],
                        dis[1]);
            }
            for (size_t lane = 0; lane < tail; ++lane) {
                const size_t pos = processed + lane;
                graph_visitor.visit_edge(
                        level, node_id, saved_indices[pos], dis[lane]);
                func_add_candidate(knowhere::Neighbor(
                        saved_indices[pos], dis[lane], saved_statuses[pos]));
            }
        } else {
            for (size_t i = processed; i < counter; ++i) {
                const float dis = qdis(saved_indices[i]);
                graph_visitor.visit_edge(level, node_id, saved_indices[i], dis);
                func_add_candidate(knowhere::Neighbor(
                        saved_indices[i], dis, saved_statuses[i]));
            }
        }

        if (track_hnsw_stats) {
            stats.ndis = ndis;
            stats.nhops = 1;
        }
        return stats;
    }

    // no loops, just check neighbors of a single node.
    template <
            bool FastLevel0 = false,
            bool ApproximateRouting = false,
            bool BatchEight = false,
            typename FuncAddCandidate>
    faiss::cppcontrib::knowhere::HNSWStats evaluate_single_node(
            const idx_t node_id,
            const int level,
            float& accumulated_alpha,
            FuncAddCandidate func_add_candidate) {
        // // unused
        // bool do_dis_check = params ? params->check_relative_distance
        //                            : hnsw.check_relative_distance;

        faiss::cppcontrib::knowhere::HNSWStats stats;

        size_t begin = 0;
        size_t end = 0;
        if constexpr (FastLevel0) {
            hnsw.neighbor_range_level0(node_id, &begin, &end);
        } else {
            hnsw.neighbor_range(node_id, level, &begin, &end);
        }

        size_t counter = 0;
        size_t saved_indices[8];
        int saved_statuses[8];
        size_t routing_counter = 0;
        size_t routing_indices[4];

        auto evaluate_four = [&](const size_t offset) {
            float dis[4] = {0, 0, 0, 0};
            qdis.distances_batch_4(
                    saved_indices[offset],
                    saved_indices[offset + 1],
                    saved_indices[offset + 2],
                    saved_indices[offset + 3],
                    dis[0],
                    dis[1],
                    dis[2],
                    dis[3]);

            for (size_t lane = 0; lane < 4; ++lane) {
                const size_t pos = offset + lane;
                graph_visitor.visit_edge(
                        level, node_id, saved_indices[pos], dis[lane]);
                func_add_candidate(knowhere::Neighbor(
                        saved_indices[pos], dis[lane], saved_statuses[pos]));
            }
        };

        auto evaluate_eight = [&]() {
            float dis[8] = {};
            qdis.distances_batch_8(
                    saved_indices[0],
                    saved_indices[1],
                    saved_indices[2],
                    saved_indices[3],
                    saved_indices[4],
                    saved_indices[5],
                    saved_indices[6],
                    saved_indices[7],
                    dis[0],
                    dis[1],
                    dis[2],
                    dis[3],
                    dis[4],
                    dis[5],
                    dis[6],
                    dis[7]);
            for (size_t lane = 0; lane < 8; ++lane) {
                graph_visitor.visit_edge(
                        level, node_id, saved_indices[lane], dis[lane]);
                func_add_candidate(knowhere::Neighbor(
                        saved_indices[lane], dis[lane], saved_statuses[lane]));
            }
        };

        auto evaluate_tail = [&](const size_t offset, const size_t width) {
            float dis[3] = {0, 0, 0};
            if (width == 3) {
                qdis.distances_batch_3(
                        saved_indices[offset],
                        saved_indices[offset + 1],
                        saved_indices[offset + 2],
                        dis[0],
                        dis[1],
                        dis[2]);
            } else if (width == 2) {
                qdis.distances_batch_2(
                        saved_indices[offset],
                        saved_indices[offset + 1],
                        dis[0],
                        dis[1]);
            } else {
                dis[0] = qdis(saved_indices[offset]);
            }
            for (size_t lane = 0; lane < width; ++lane) {
                const size_t pos = offset + lane;
                graph_visitor.visit_edge(
                        level, node_id, saved_indices[pos], dis[lane]);
                func_add_candidate(knowhere::Neighbor(
                        saved_indices[pos], dis[lane], saved_statuses[pos]));
            }
        };

        auto evaluate_routing = [&](const size_t width) {
            float dis[4] = {};
            if (width == 4) {
                qdis.routing_distances_batch_4(
                        routing_indices[0],
                        routing_indices[1],
                        routing_indices[2],
                        routing_indices[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);
            } else if (width == 3) {
                qdis.routing_distances_batch_3(
                        routing_indices[0],
                        routing_indices[1],
                        routing_indices[2],
                        dis[0],
                        dis[1],
                        dis[2]);
            } else if (width == 2) {
                qdis.routing_distances_batch_2(
                        routing_indices[0],
                        routing_indices[1],
                        dis[0],
                        dis[1]);
            } else {
                dis[0] = qdis.routing_distance(routing_indices[0]);
            }
            for (size_t lane = 0; lane < width; ++lane) {
                graph_visitor.visit_edge(
                        level, node_id, routing_indices[lane], dis[lane]);
                func_add_candidate(knowhere::Neighbor(
                        routing_indices[lane],
                        dis[lane],
                        knowhere::Neighbor::kInvalid));
            }
        };

        size_t ndis = 0;
        for (size_t j = begin; j < end; j++) {
            const storage_idx_t v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                // no more neighbors
                break;
            }

            // already visited?
            if (visited_nodes.get(v1)) {
                // yes, visited.
                graph_visitor.visit_edge(level, node_id, v1, -1);
                continue;
            }

            // not visited. mark as visited.
            visited_nodes.set(v1);

            // is the node disabled?
            int status = knowhere::Neighbor::kValid;
            if (!filter.is_member(v1)) {
                // yes, disabled
                status = knowhere::Neighbor::kInvalid;

                // sometimes, disabled nodes are allowed to be used
                accumulated_alpha += kAlpha;
                if (accumulated_alpha < 1.0f) {
                    continue;
                }

                accumulated_alpha -= 1.0f;

                if constexpr (ApproximateRouting) {
                    routing_indices[routing_counter++] = v1;
                    ndis += 1;
                    if (routing_counter == 4) {
                        evaluate_routing(4);
                        routing_counter = 0;
                    }
                    continue;
                }

            }

            saved_indices[counter] = v1;
            saved_statuses[counter] = status;
            counter += 1;

            ndis += 1;

            if (counter == 4) {
                if (use_distance_pipeline) {
                    qdis.prefetch_batch_4(
                            saved_indices[0],
                            saved_indices[1],
                            saved_indices[2],
                            saved_indices[3]);
                } else {
                    evaluate_four(0);
                    counter = 0;
                }
            } else if (counter == 8) {
                qdis.prefetch_batch_4(
                        saved_indices[4],
                        saved_indices[5],
                        saved_indices[6],
                        saved_indices[7]);
                if constexpr (BatchEight) {
                    evaluate_eight();
                } else {
                    evaluate_four(0);
                    evaluate_four(4);
                }
                counter = 0;
            }
        }

        // process leftovers
        size_t processed = 0;
        if (counter >= 4) {
            evaluate_four(0);
            processed = 4;
        }
        if (use_tail_distance_batches && processed < counter) {
            evaluate_tail(processed, counter - processed);
        } else {
            for (size_t id4 = processed; id4 < counter; ++id4) {
                const float dis = qdis(saved_indices[id4]);
                graph_visitor.visit_edge(
                        level, node_id, saved_indices[id4], dis);
                func_add_candidate(knowhere::Neighbor(
                        saved_indices[id4], dis, saved_statuses[id4]));
            }
        }

        if constexpr (ApproximateRouting) {
            if (routing_counter != 0) {
                evaluate_routing(routing_counter);
            }
        }

        // update stats
        if (track_hnsw_stats) {
            stats.ndis = ndis;
            stats.nhops = 1;
        }

        // done
        return stats;
    }

    // NaviX-style adaptive-local expansion. It inspects filter bits before
    // computing distances, and uses one or two graph hops according to the
    // selected fraction in the current neighborhood.
    template <bool FastLevel0 = false, typename FuncAddCandidate>
    faiss::cppcontrib::knowhere::HNSWStats evaluate_single_node_adaptive(
            const idx_t node_id,
            const int level,
            FuncAddCandidate func_add_candidate) {
        faiss::cppcontrib::knowhere::HNSWStats stats;

        size_t begin = 0;
        size_t end = 0;
        if constexpr (FastLevel0) {
            hnsw.neighbor_range_level0(node_id, &begin, &end);
        } else {
            hnsw.neighbor_range(node_id, level, &begin, &end);
        }
        while (end > begin && hnsw.neighbors[end - 1] < 0) {
            --end;
        }
        const size_t total_neighbors = end - begin;
        if (total_neighbors == 0) {
            stats.nhops = 1;
            return stats;
        }

        size_t selected_neighbors = 0;
        for (size_t j = begin; j < end; ++j) {
            selected_neighbors += filter.is_member(hnsw.neighbors[j]) ? 1 : 0;
        }
        const double local_selectivity =
                static_cast<double>(selected_neighbors) / total_neighbors;
        const double estimated_full_two_hop =
                (total_neighbors * selected_neighbors + selected_neighbors) * 0.4;
        const double estimated_directed =
                total_neighbors + (total_neighbors - selected_neighbors);

        // The common high-selectivity path stays allocation-free. Dynamic
        // containers here cost more than the distances saved on low dimensions.
        if (level == 0 && local_selectivity >= 0.5) {
            storage_idx_t ids[4];
            size_t count = 0;
            auto flush = [&](size_t width) {
                float dis[4] = {0, 0, 0, 0};
                if (width == 4) {
                    qdis.distances_batch_4(
                            ids[0], ids[1], ids[2], ids[3],
                            dis[0], dis[1], dis[2], dis[3]);
                } else if (use_tail_distance_batches && width == 3) {
                    qdis.distances_batch_3(
                            ids[0], ids[1], ids[2],
                            dis[0], dis[1], dis[2]);
                } else if (use_tail_distance_batches && width == 2) {
                    qdis.distances_batch_2(
                            ids[0], ids[1], dis[0], dis[1]);
                } else {
                    for (size_t lane = 0; lane < width; ++lane) {
                        dis[lane] = qdis(ids[lane]);
                    }
                }
                for (size_t lane = 0; lane < width; ++lane) {
                    graph_visitor.visit_edge(level, node_id, ids[lane], dis[lane]);
                    func_add_candidate(knowhere::Neighbor(
                            ids[lane], dis[lane], knowhere::Neighbor::kValid));
                }
                stats.ndis += width;
            };
            for (size_t j = begin; j < end; ++j) {
                const storage_idx_t id = hnsw.neighbors[j];
                if (visited_nodes.get(id) || !filter.is_member(id)) {
                    continue;
                }
                visited_nodes.set(id);
                ids[count++] = id;
                if (count == 4) {
                    flush(count);
                    count = 0;
                }
            }
            if (count != 0) {
                flush(count);
            }
            stats.nhops = 1;
            return stats;
        }

        auto& pending_ids = adaptive_pending_ids;
        auto& pending_parents = adaptive_pending_parents;
        pending_ids.clear();
        pending_parents.clear();
        pending_ids.reserve(total_neighbors * 2);
        pending_parents.reserve(total_neighbors * 2);

        auto queue_selected = [&](storage_idx_t id, storage_idx_t parent) {
            if (id < 0 || visited_nodes.get(id) || !filter.is_member(id)) {
                return false;
            }
            visited_nodes.set(id);
            pending_ids.push_back(id);
            pending_parents.push_back(parent);
            return true;
        };

        auto compute_pending = [&]() {
            size_t i = 0;
            for (; i + 4 <= pending_ids.size(); i += 4) {
                float dis[4];
                qdis.distances_batch_4(
                        pending_ids[i],
                        pending_ids[i + 1],
                        pending_ids[i + 2],
                        pending_ids[i + 3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);
                for (size_t lane = 0; lane < 4; ++lane) {
                    graph_visitor.visit_edge(
                            level,
                            pending_parents[i + lane],
                            pending_ids[i + lane],
                            dis[lane]);
                    func_add_candidate(knowhere::Neighbor(
                            pending_ids[i + lane],
                            dis[lane],
                            knowhere::Neighbor::kValid));
                }
            }
            for (; i < pending_ids.size(); ++i) {
                const float dis = qdis(pending_ids[i]);
                graph_visitor.visit_edge(
                        level, pending_parents[i], pending_ids[i], dis);
                func_add_candidate(knowhere::Neighbor(
                        pending_ids[i], dis, knowhere::Neighbor::kValid));
            }
            stats.ndis += pending_ids.size();
            pending_ids.clear();
            pending_parents.clear();
        };

        if (level != 0) {
            for (size_t j = begin; j < end; ++j) {
                queue_selected(hnsw.neighbors[j], node_id);
            }
            compute_pending();
            stats.nhops += 1;
            return stats;
        }

        if (estimated_full_two_hop > estimated_directed) {
            // Directed: pay for every first-hop distance, then inspect second-hop
            // lists in query-distance order until roughly M selected links exist.
            auto& ordered_first_hop = adaptive_ordered_first_hop;
            ordered_first_hop.clear();
            ordered_first_hop.reserve(total_neighbors);
            size_t selected_seen = 0;
            for (size_t j = begin; j < end; ++j) {
                const storage_idx_t v1 = hnsw.neighbors[j];
                if (visited_nodes.get(v1)) {
                    continue;
                }
                const float dis = qdis(v1);
                ++stats.ndis;
                graph_visitor.visit_edge(level, node_id, v1, dis);
                ordered_first_hop.emplace_back(dis, v1);
                if (filter.is_member(v1)) {
                    visited_nodes.set(v1);
                    func_add_candidate(knowhere::Neighbor(
                            v1, dis, knowhere::Neighbor::kValid));
                    ++selected_seen;
                }
            }
            std::sort(ordered_first_hop.begin(), ordered_first_hop.end());

            for (const auto& [unused_distance, v1] : ordered_first_hop) {
                (void)unused_distance;
                if (selected_seen >= total_neighbors) {
                    break;
                }
                if (visited_nodes.get(v1)) {
                    continue;
                }
                visited_nodes.set(v1);
                size_t second_begin = 0;
                size_t second_end = 0;
                hnsw.neighbor_range(v1, 0, &second_begin, &second_end);
                ++stats.nhops;
                for (size_t j = second_begin; j < second_end; ++j) {
                    const storage_idx_t v2 = hnsw.neighbors[j];
                    if (v2 < 0) {
                        break;
                    }
                    if (queue_selected(v2, v1)) {
                        ++selected_seen;
                    }
                }
            }
        } else {
            // Blind two-hop: first admit every selected one-hop neighbor,
            // then bridge through unselected neighbors only until roughly M
            // selected links have been observed.
            auto& bridges = adaptive_bridges;
            bridges.clear();
            bridges.reserve(total_neighbors);
            size_t selected_seen = 0;
            for (size_t j = begin; j < end; ++j) {
                const storage_idx_t v1 = hnsw.neighbors[j];
                if (filter.is_member(v1)) {
                    selected_seen += queue_selected(v1, node_id) ? 1 : 0;
                } else if (!visited_nodes.get(v1)) {
                    bridges.push_back(v1);
                }
            }
            for (const storage_idx_t v1 : bridges) {
                if (selected_seen >= total_neighbors) {
                    break;
                }
                if (visited_nodes.get(v1)) {
                    continue;
                }
                visited_nodes.set(v1);
                size_t second_begin = 0;
                size_t second_end = 0;
                hnsw.neighbor_range(v1, 0, &second_begin, &second_end);
                ++stats.nhops;
                for (size_t k = second_begin; k < second_end; ++k) {
                    const storage_idx_t v2 = hnsw.neighbors[k];
                    if (v2 < 0) {
                        break;
                    }
                    if (queue_selected(v2, v1)) {
                        ++selected_seen;
                    }
                }
            }
        }

        compute_pending();
        stats.nhops += 1;
        return stats;
    }

    // Low-dimensional level-0 loop paired with evaluate_single_node_compact.
    // The offset prefetch is the only graph-latency optimization kept here.
    faiss::cppcontrib::knowhere::HNSWStats search_on_a_level_compact(
            knowhere::NeighborSetDoublePopList& retset,
            const int level,
            knowhere::IteratorMinHeap* const __restrict disqualified = nullptr,
            const float initial_accumulated_alpha = 1.0f) {
        faiss::cppcontrib::knowhere::HNSWStats stats;
        float accumulated_alpha = initial_accumulated_alpha;
        auto add_search_candidate = [&](const knowhere::Neighbor n) {
            const bool inserted = retset.insert(n, disqualified);
            if (inserted) {
                __builtin_prefetch(&hnsw.offsets[n.id], 0, 1);
            }
            return inserted;
        };

        while (retset.has_next()) {
            const knowhere::Neighbor neighbor = retset.pop();
            faiss::cppcontrib::knowhere::HNSWStats local_stats;
            if (use_adaptive_filter && level == 0 &&
                disqualified == nullptr) {
                local_stats = evaluate_single_node_adaptive<false>(
                        neighbor.id, level, add_search_candidate);
            } else {
                local_stats = evaluate_single_node_compact(
                        neighbor.id,
                        level,
                        accumulated_alpha,
                        add_search_candidate);
            }
            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }
        return stats;
    }

    // perform the search on a given level.
    // it is assumed that retset is initialized and contains the initial nodes.
    template <
            bool PrefetchGraphOffsets = false,
            bool ApproximateRouting = false,
            bool BatchEight = false>
    faiss::cppcontrib::knowhere::HNSWStats search_on_a_level(
            knowhere::NeighborSetDoublePopList& retset,
            const int level,
            knowhere::IteratorMinHeap* const __restrict disqualified = nullptr,
            const float initial_accumulated_alpha = 1.0f) {
        faiss::cppcontrib::knowhere::HNSWStats stats;

        //
        float accumulated_alpha = initial_accumulated_alpha;

        // what to do with a accepted candidate
        auto add_search_candidate = [&](const knowhere::Neighbor n) {
            const bool inserted = retset.insert(n, disqualified);
            if constexpr (PrefetchGraphOffsets) {
                // Inserted candidates are likely to be popped after enough
                // intervening distance work to hide this random offset load.
                if (inserted) {
                    __builtin_prefetch(&hnsw.offsets[n.id], 0, 1);
                }
            }
            return inserted;
        };

        // iterate while possible
        while (retset.has_next()) {
            // get a node to be processed
            const knowhere::Neighbor neighbor = retset.pop();

            // analyze its neighbors
            faiss::cppcontrib::knowhere::HNSWStats local_stats;
            if (use_adaptive_filter && level == 0 && disqualified == nullptr) {
                local_stats = evaluate_single_node_adaptive<false>(
                        neighbor.id, level, add_search_candidate);
            } else {
                local_stats = evaluate_single_node<
                        false,
                        ApproximateRouting,
                        BatchEight>(
                        neighbor.id,
                        level,
                        accumulated_alpha,
                        add_search_candidate);
            }

            // update stats
            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }

        // done
        return stats;
    }

    // traverse down to the level 0
    faiss::cppcontrib::knowhere::HNSWStats greedy_search_top_levels(
            storage_idx_t& nearest,
            float& d_nearest) {
        faiss::cppcontrib::knowhere::HNSWStats stats;

        // iterate through upper levels
        for (int level = hnsw.max_level; level >= 1; level--) {
            // update the visitor
            graph_visitor.visit_level(level);

            // alter the value of 'nearest'
            faiss::cppcontrib::knowhere::HNSWStats local_stats =
                    greedy_update_nearest(level, nearest, d_nearest);

            // update stats
            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }

        return stats;
    }

    // perform the search.
    faiss::cppcontrib::knowhere::HNSWStats search(
            const idx_t k,
            float* __restrict distances,
            idx_t* __restrict labels) {
        faiss::cppcontrib::knowhere::HNSWStats stats;

        // is the graph empty?
        if (hnsw.entry_point == -1) {
            return stats;
        }

        // grab some needed parameters
        const int efSearch = (params ? params->efSearch : hnsw.efSearch) +
                ((use_approximate_routing_distance && kAlpha <= 0.36f) ? 1
                                                                      : 0);

        // yes.
        // greedy search on upper levels.

        // initialize the starting point.
        storage_idx_t nearest = hnsw.entry_point;
        float d_nearest = qdis(nearest);

        // iterate through upper levels
        auto bottom_levels_stats = greedy_search_top_levels(nearest, d_nearest);

        // update stats
        if (track_hnsw_stats) {
            stats.combine(bottom_levels_stats);
        }

        // level 0 search

        // update the visitor
        graph_visitor.visit_level(0);

        // initialize the container for candidates
        const idx_t n_candidates = std::max((idx_t)efSearch, k);
        knowhere::NeighborSetDoublePopList retset(n_candidates);

        // initialize retset with a single 'nearest' point
        {
            if (!filter.is_member(nearest)) {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kInvalid));
            } else {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kValid));
            }

            visited_nodes[nearest] = true;
        }

        // Seed a few selected nodes from both ID-space ends. This is not a
        // vector scan: it checks filter bits and computes at most 10 distances.
        // It prevents a highly selective projected graph from starting empty.
        if (use_adaptive_filter && kAlpha >= 0.665f) {
            const size_t ntotal = hnsw.levels.size();
            const size_t seed_target = std::min<size_t>(10, k);
            size_t seeded = 0;
            for (size_t step = 0; step < ntotal && seeded < seed_target; ++step) {
                const storage_idx_t id = static_cast<storage_idx_t>(
                        (step & 1) ? ntotal - 1 - step / 2 : step / 2);
                if (visited_nodes.get(id) || !filter.is_member(id)) {
                    continue;
                }
                const float dis = qdis(id);
                retset.insert(knowhere::Neighbor(
                        id, dis, knowhere::Neighbor::kValid));
                visited_nodes.set(id);
                ++seeded;
                ++stats.ndis;
            }
        }

        // perform the search of the level 0.
        faiss::cppcontrib::knowhere::HNSWStats local_stats;
        if (use_approximate_routing_distance && use_distance_batch8) {
            local_stats = search_on_a_level<false, true, true>(retset, 0);
        } else if (use_approximate_routing_distance) {
            local_stats = search_on_a_level<false, true, false>(retset, 0);
        } else if (use_graph_offset_prefetch) {
            local_stats = search_on_a_level_compact(retset, 0);
        } else if (use_distance_batch8) {
            local_stats = search_on_a_level<false, false, true>(retset, 0);
        } else {
            local_stats = search_on_a_level<false, false, false>(retset, 0);
        }

        // todo: switch to brute-force in case of (retset.size() < k)

        // populate the result
        const idx_t len = std::min((idx_t)retset.size(), k);
        for (idx_t i = 0; i < len; i++) {
            distances[i] = retset[i].distance;
            labels[i] = (idx_t)retset[i].id;
        }
        if (len < k) {
            for (idx_t idx = len; idx < k; idx++) {
                labels[idx] = -1;
                distances[idx] = std::numeric_limits<float>::max();
            }
        }
        // update stats
        if (track_hnsw_stats) {
            stats.combine(local_stats);
        }

        // done
        return stats;
    }

    faiss::cppcontrib::knowhere::HNSWStats range_search(
            const float radius,
            typename faiss::cppcontrib::knowhere::RangeSearchBlockResultHandler<
                    faiss::CMax<float, int64_t>>::
                    SingleResultHandler* const __restrict rres) {
        faiss::cppcontrib::knowhere::HNSWStats stats;

        // is the graph empty?
        if (hnsw.entry_point == -1) {
            return stats;
        }

        // grab some needed parameters
        const int efSearch = (params ? params->efSearch : hnsw.efSearch) +
                ((use_approximate_routing_distance && kAlpha <= 0.36f) ? 1
                                                                      : 0);

        // yes.
        // greedy search on upper levels.

        // initialize the starting point.
        storage_idx_t nearest = hnsw.entry_point;
        float d_nearest = qdis(nearest);

        // iterate through upper levels
        auto bottom_levels_stats = greedy_search_top_levels(nearest, d_nearest);

        // update stats
        if (track_hnsw_stats) {
            stats.combine(bottom_levels_stats);
        }

        // level 0 search

        // update the visitor
        graph_visitor.visit_level(0);

        // initialize the container for candidates
        const idx_t n_candidates = efSearch;
        knowhere::NeighborSetDoublePopList retset(n_candidates);

        // initialize retset with a single 'nearest' point
        {
            if (!filter.is_member(nearest)) {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kInvalid));
            } else {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kValid));
            }

            visited_nodes[nearest] = true;
        }

        // perform the search of the level 0.
        faiss::cppcontrib::knowhere::HNSWStats local_stats;
        if (use_approximate_routing_distance && use_distance_batch8) {
            local_stats = search_on_a_level<false, true, true>(retset, 0);
        } else if (use_approximate_routing_distance) {
            local_stats = search_on_a_level<false, true, false>(retset, 0);
        } else if (use_graph_offset_prefetch) {
            local_stats = search_on_a_level_compact(retset, 0);
        } else if (use_distance_batch8) {
            local_stats = search_on_a_level<false, false, true>(retset, 0);
        } else {
            local_stats = search_on_a_level<false, false, false>(retset, 0);
        }

        // update stats
        if (track_hnsw_stats) {
            stats.combine(local_stats);
        }

        // select candidates that match our criteria
        faiss::cppcontrib::knowhere::HNSWStats pick_stats;

        visited_nodes.clear();

        std::queue<std::pair<float, int64_t>> radius_queue;
        for (size_t i = retset.size(); (i--) > 0;) {
            const auto candidate = retset[i];
            if (candidate.distance < radius) {
                radius_queue.push({candidate.distance, candidate.id});
                rres->add_result(candidate.distance, candidate.id);

                visited_nodes[candidate.id] = true;
            }
        }

        while (!radius_queue.empty()) {
            auto current = radius_queue.front();
            radius_queue.pop();

            size_t id_begin = 0;
            size_t id_end = 0;
            hnsw.neighbor_range(current.second, 0, &id_begin, &id_end);

            for (size_t id = id_begin; id < id_end; id++) {
                const auto ngb = hnsw.neighbors[id];
                if (ngb == -1) {
                    break;
                }

                if (visited_nodes[ngb]) {
                    continue;
                }

                visited_nodes[ngb] = true;

                if (filter.is_member(ngb)) {
                    const float dis = qdis(ngb);
                    if (dis < radius) {
                        radius_queue.push({dis, ngb});
                        rres->add_result(dis, ngb);
                    }

                    if (track_hnsw_stats) {
                        pick_stats.ndis += 1;
                    }
                }
            }
        }

        // update stats
        if (track_hnsw_stats) {
            stats.combine(pick_stats);
        }

        return stats;
    }
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
