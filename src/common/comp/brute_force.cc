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

#include "knowhere/comp/brute_force.h"

#include <vector>

#include "common/metric.h"
#include "faiss/MetricType.h"
#include "faiss/utils/binary_distances.h"
#include "faiss/utils/distances.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/config.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/tracer.h"
#endif

namespace knowhere {

/* knowhere wrapper API to call faiss brute force search for all metric types */

class BruteForceConfig : public BaseConfig {};

template <typename DataType>
expected<DataSetPtr>
BruteForce::Search(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                   const BitsetView& bitset) {
    auto base = ConvertFromDataTypeIfNeeded<DataType>(base_dataset);
    auto query = ConvertFromDataTypeIfNeeded<DataType>(query_dataset);

    auto xb = base->GetTensor();
    auto nb = base->GetRows();
    auto dim = base->GetDim();

    auto xq = query->GetTensor();
    auto nq = query->GetRows();

    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::SEARCH, &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, msg);
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto ctx = tracer::GetTraceCtxFromCfg(&cfg);
        span = tracer::StartSpan("knowhere bf search", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, cfg.metric_type.value());
        span->SetAttribute(meta::TOPK, cfg.k.value());
        span->SetAttribute(meta::ROWS, nb);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
#endif

    std::string metric_str = cfg.metric_type.value();
    auto result = Str2FaissMetricType(metric_str);
    if (result.error() != Status::success) {
        return expected<DataSetPtr>::Err(result.error(), result.what());
    }
    faiss::MetricType faiss_metric_type = result.value();
    bool is_cosine = IsMetricType(metric_str, metric::COSINE);

    int topk = cfg.k.value();
    auto labels = std::make_unique<int64_t[]>(nq * topk);
    auto distances = std::make_unique<float[]>(nq * topk);

    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<Status>> futs;
    futs.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        futs.emplace_back(pool->push([&, index = i, labels_ptr = labels.get(), distances_ptr = distances.get()] {
            ThreadPool::ScopedOmpSetter setter(1);
            auto cur_labels = labels_ptr + topk * index;
            auto cur_distances = distances_ptr + topk * index;

            BitsetViewIDSelector bw_idselector(bitset);
            faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;

            switch (faiss_metric_type) {
                case faiss::METRIC_L2: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::float_maxheap_array_t buf{(size_t)1, (size_t)topk, cur_labels, cur_distances};
                    faiss::knn_L2sqr(cur_query, (const float*)xb, dim, 1, nb, &buf, nullptr, id_selector);
                    break;
                }
                case faiss::METRIC_INNER_PRODUCT: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::float_minheap_array_t buf{(size_t)1, (size_t)topk, cur_labels, cur_distances};
                    if (is_cosine) {
                        auto copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        faiss::knn_cosine(copied_query.get(), (const float*)xb, nullptr, dim, 1, nb, &buf, id_selector);
                    } else {
                        faiss::knn_inner_product(cur_query, (const float*)xb, dim, 1, nb, &buf, id_selector);
                    }
                    break;
                }
                case faiss::METRIC_Jaccard: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::float_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, cur_distances};
                    binary_knn_hc(faiss::METRIC_Jaccard, &res, cur_query, (const uint8_t*)xb, nb, dim / 8, id_selector);
                    break;
                }
                case faiss::METRIC_Hamming: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    std::vector<int32_t> int_distances(topk);
                    faiss::int_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, int_distances.data()};
                    binary_knn_hc(faiss::METRIC_Hamming, &res, (const uint8_t*)cur_query, (const uint8_t*)xb, nb,
                                  dim / 8, id_selector);
                    for (int i = 0; i < topk; ++i) {
                        cur_distances[i] = int_distances[i];
                    }
                    break;
                }
                case faiss::METRIC_Substructure:
                case faiss::METRIC_Superstructure: {
                    // only matched ids will be chosen, not to use heap
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    binary_knn_mc(faiss_metric_type, cur_query, (const uint8_t*)xb, 1, nb, topk, dim / 8, cur_distances,
                                  cur_labels, id_selector);
                    break;
                }
                default: {
                    LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << cfg.metric_type.value();
                    return Status::invalid_metric_type;
                }
            }
            return Status::success;
        }));
    }
    auto ret = WaitAllSuccess(futs);
    if (ret != Status::success) {
        return expected<DataSetPtr>::Err(ret, "failed to brute force search");
    }
    auto res = GenResultDataSet(nq, cfg.k.value(), std::move(labels), std::move(distances));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    if (cfg.trace_id.has_value()) {
        span->End();
    }
#endif

    return res;
}

template <typename DataType>
Status
BruteForce::SearchWithBuf(const DataSetPtr base_dataset, const DataSetPtr query_dataset, int64_t* ids, float* dis,
                          const Json& config, const BitsetView& bitset) {
    auto base = ConvertFromDataTypeIfNeeded<DataType>(base_dataset);
    auto query = ConvertFromDataTypeIfNeeded<DataType>(query_dataset);

    auto xb = base->GetTensor();
    auto nb = base->GetRows();
    auto dim = base->GetDim();

    auto xq = query->GetTensor();
    auto nq = query->GetRows();

    BruteForceConfig cfg;
    RETURN_IF_ERROR(Config::Load(cfg, config, knowhere::SEARCH));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto ctx = tracer::GetTraceCtxFromCfg(&cfg);
        span = tracer::StartSpan("knowhere bf search with buf", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, cfg.metric_type.value());
        span->SetAttribute(meta::TOPK, cfg.k.value());
        span->SetAttribute(meta::ROWS, nb);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
#endif

    std::string metric_str = cfg.metric_type.value();
    auto result = Str2FaissMetricType(cfg.metric_type.value());
    if (result.error() != Status::success) {
        return result.error();
    }
    faiss::MetricType faiss_metric_type = result.value();
    bool is_cosine = IsMetricType(metric_str, metric::COSINE);

    int topk = cfg.k.value();
    auto labels = ids;
    auto distances = dis;

    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<Status>> futs;
    futs.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        futs.emplace_back(pool->push([&, index = i] {
            ThreadPool::ScopedOmpSetter setter(1);
            auto cur_labels = labels + topk * index;
            auto cur_distances = distances + topk * index;

            BitsetViewIDSelector bw_idselector(bitset);
            faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;

            switch (faiss_metric_type) {
                case faiss::METRIC_L2: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::float_maxheap_array_t buf{(size_t)1, (size_t)topk, cur_labels, cur_distances};
                    faiss::knn_L2sqr(cur_query, (const float*)xb, dim, 1, nb, &buf, nullptr, id_selector);
                    break;
                }
                case faiss::METRIC_INNER_PRODUCT: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::float_minheap_array_t buf{(size_t)1, (size_t)topk, cur_labels, cur_distances};
                    if (is_cosine) {
                        auto copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        faiss::knn_cosine(copied_query.get(), (const float*)xb, nullptr, dim, 1, nb, &buf, id_selector);
                    } else {
                        faiss::knn_inner_product(cur_query, (const float*)xb, dim, 1, nb, &buf, id_selector);
                    }
                    break;
                }
                case faiss::METRIC_Jaccard: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::float_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, cur_distances};
                    binary_knn_hc(faiss::METRIC_Jaccard, &res, cur_query, (const uint8_t*)xb, nb, dim / 8, id_selector);
                    break;
                }
                case faiss::METRIC_Hamming: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    std::vector<int32_t> int_distances(topk);
                    faiss::int_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, int_distances.data()};
                    binary_knn_hc(faiss::METRIC_Hamming, &res, (const uint8_t*)cur_query, (const uint8_t*)xb, nb,
                                  dim / 8, id_selector);
                    for (int i = 0; i < topk; ++i) {
                        cur_distances[i] = int_distances[i];
                    }
                    break;
                }
                case faiss::METRIC_Substructure:
                case faiss::METRIC_Superstructure: {
                    // only matched ids will be chosen, not to use heap
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    binary_knn_mc(faiss_metric_type, cur_query, (const uint8_t*)xb, 1, nb, topk, dim / 8, cur_distances,
                                  cur_labels, id_selector);
                    break;
                }
                default: {
                    LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << cfg.metric_type.value();
                    return Status::invalid_metric_type;
                }
            }
            return Status::success;
        }));
    }
    RETURN_IF_ERROR(WaitAllSuccess(futs));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    if (cfg.trace_id.has_value()) {
        span->End();
    }
#endif

    return Status::success;
}

/** knowhere wrapper API to call faiss brute force range search for all metric types
 */
template <typename DataType>
expected<DataSetPtr>
BruteForce::RangeSearch(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                        const BitsetView& bitset) {
    DataSetPtr base(base_dataset);
    DataSetPtr query(query_dataset);
    bool is_sparse = std::is_same<DataType, knowhere::sparse::SparseRow<float>>::value;
    if (!is_sparse) {
        base = ConvertFromDataTypeIfNeeded<DataType>(base_dataset);
        query = ConvertFromDataTypeIfNeeded<DataType>(query_dataset);
    }
    auto xb = base->GetTensor();
    auto nb = base->GetRows();
    auto dim = base->GetDim();

    auto xq = query->GetTensor();
    auto nq = query->GetRows();

    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::RANGE_SEARCH, &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, std::move(msg));
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto ctx = tracer::GetTraceCtxFromCfg(&cfg);
        span = tracer::StartSpan("knowhere bf range search", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, cfg.metric_type.value());
        span->SetAttribute(meta::RADIUS, cfg.radius.value());
        if (cfg.range_filter.value() != defaultRangeFilter) {
            span->SetAttribute(meta::RANGE_FILTER, cfg.range_filter.value());
        }
        span->SetAttribute(meta::ROWS, nb);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
#endif

    std::string metric_str = cfg.metric_type.value();
    auto result = Str2FaissMetricType(metric_str);
    if (result.error() != Status::success) {
        return expected<DataSetPtr>::Err(result.error(), result.what());
    }
    faiss::MetricType faiss_metric_type = result.value();
    if (is_sparse && !IsMetricType(metric_str, metric::IP)) {
        return expected<DataSetPtr>::Err(Status::invalid_metric_type,
                                         "Invalid metric type for sparse float vector: " + metric_str);
    }
    bool is_cosine = IsMetricType(metric_str, metric::COSINE);

    auto radius = cfg.radius.value();
    bool is_ip = false;
    float range_filter = cfg.range_filter.value();

    auto pool = ThreadPool::GetGlobalSearchThreadPool();

    std::vector<std::vector<int64_t>> result_id_array(nq);
    std::vector<std::vector<float>> result_dist_array(nq);

    std::vector<folly::Future<Status>> futs;
    futs.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        futs.emplace_back(pool->push([&, index = i] {
            if (is_sparse) {
                auto cur_query = (const sparse::SparseRow<float>*)xq + index;
                auto xb_sparse = (const sparse::SparseRow<float>*)xb;
                for (int j = 0; j < nb; ++j) {
                    if (!bitset.empty() && bitset.test(j)) {
                        continue;
                    }
                    auto dist = cur_query->dot(xb_sparse[j]);
                    if (dist > radius && dist <= range_filter) {
                        result_id_array[index].push_back(j);
                        result_dist_array[index].push_back(dist);
                    }
                }
                return Status::success;
            }
            // else not sparse:
            ThreadPool::ScopedOmpSetter setter(1);
            faiss::RangeSearchResult res(1);

            BitsetViewIDSelector bw_idselector(bitset);
            faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;

            switch (faiss_metric_type) {
                case faiss::METRIC_L2: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::range_search_L2sqr(cur_query, (const float*)xb, dim, 1, nb, radius, &res, id_selector);
                    break;
                }
                case faiss::METRIC_INNER_PRODUCT: {
                    is_ip = true;
                    auto cur_query = (const float*)xq + dim * index;
                    if (is_cosine) {
                        auto copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        faiss::range_search_cosine(copied_query.get(), (const float*)xb, nullptr, dim, 1, nb, radius,
                                                   &res, id_selector);
                    } else {
                        faiss::range_search_inner_product(cur_query, (const float*)xb, dim, 1, nb, radius, &res,
                                                          id_selector);
                    }
                    break;
                }
                case faiss::METRIC_Jaccard: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::binary_range_search<faiss::CMin<float, int64_t>, float>(faiss::METRIC_Jaccard, cur_query,
                                                                                   (const uint8_t*)xb, 1, nb, radius,
                                                                                   dim / 8, &res, id_selector);
                    break;
                }
                case faiss::METRIC_Hamming: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::binary_range_search<faiss::CMin<int, int64_t>, int>(faiss::METRIC_Hamming, cur_query,
                                                                               (const uint8_t*)xb, 1, nb, (int)radius,
                                                                               dim / 8, &res, id_selector);
                    break;
                }
                default: {
                    LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << cfg.metric_type.value();
                    return Status::invalid_metric_type;
                }
            }
            auto elem_cnt = res.lims[1];
            result_dist_array[index].resize(elem_cnt);
            result_id_array[index].resize(elem_cnt);
            for (size_t j = 0; j < elem_cnt; j++) {
                result_dist_array[index][j] = res.distances[j];
                result_id_array[index][j] = res.labels[j];
            }
            if (cfg.range_filter.value() != defaultRangeFilter) {
                FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, radius,
                                                range_filter);
            }
            return Status::success;
        }));
    }
    auto ret = WaitAllSuccess(futs);
    if (ret != Status::success) {
        return expected<DataSetPtr>::Err(ret, "failed to brute force search");
    }

    auto range_search_result =
        GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius, range_filter);
    auto res = GenResultDataSet(nq, std::move(range_search_result));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    if (cfg.trace_id.has_value()) {
        span->End();
    }
#endif

    return res;
}

Status
BruteForce::SearchSparseWithBuf(const DataSetPtr base_dataset, const DataSetPtr query_dataset, sparse::label_t* labels,
                                float* distances, const Json& config, const BitsetView& bitset) {
    auto base = static_cast<const sparse::SparseRow<float>*>(base_dataset->GetTensor());
    auto rows = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();

    auto xq = static_cast<const sparse::SparseRow<float>*>(query_dataset->GetTensor());
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::SEARCH, &msg);
    if (status != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Failed to load config, msg is: " << msg;
        return status;
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto ctx = tracer::GetTraceCtxFromCfg(&cfg);
        span = tracer::StartSpan("knowhere bf search sparse with buf", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, cfg.metric_type.value());
        span->SetAttribute(meta::TOPK, cfg.k.value());
        span->SetAttribute(meta::ROWS, rows);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
#endif

    std::string metric_str = cfg.metric_type.value();
    auto result = Str2FaissMetricType(metric_str);
    if (result.error() != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << cfg.metric_type.value();
        return result.error();
    }
    if (!IsMetricType(metric_str, metric::IP)) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << cfg.metric_type.value();
        return Status::invalid_metric_type;
    }

    int topk = cfg.k.value();
    std::fill(distances, distances + nq * topk, std::numeric_limits<float>::quiet_NaN());
    std::fill(labels, labels + nq * topk, -1);

    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futs;
    futs.reserve(nq);
    for (int64_t i = 0; i < nq; ++i) {
        futs.emplace_back(pool->push([&, index = i] {
            auto cur_labels = labels + topk * index;
            auto cur_distances = distances + topk * index;

            const auto& row = xq[index];
            if (row.size() == 0) {
                return;
            }
            sparse::MaxMinHeap<float> heap(topk);
            for (int64_t j = 0; j < rows; ++j) {
                if (!bitset.empty() && bitset.test(j)) {
                    continue;
                }
                float dist = row.dot(base[j]);
                if (dist > 0) {
                    heap.push(j, dist);
                }
            }
            int result_size = heap.size();
            for (int j = result_size - 1; j >= 0; --j) {
                cur_labels[j] = heap.top().id;
                cur_distances[j] = heap.top().val;
                heap.pop();
            }
        }));
    }
    WaitAllSuccess(futs);

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    if (cfg.trace_id.has_value()) {
        span->End();
    }
#endif

    return Status::success;
}

expected<DataSetPtr>
BruteForce::SearchSparse(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                         const BitsetView& bitset) {
    auto nq = query_dataset->GetRows();
    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::SEARCH, &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, msg);
    }

    int topk = cfg.k.value();
    auto labels = std::make_unique<sparse::label_t[]>(nq * topk);
    auto distances = std::make_unique<float[]>(nq * topk);

    SearchSparseWithBuf(base_dataset, query_dataset, labels.get(), distances.get(), config, bitset);
    return GenResultDataSet(nq, topk, std::move(labels), std::move(distances));
}

template <typename DataType>
expected<std::vector<IndexNode::IteratorPtr>>
BruteForce::AnnIterator(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                        const BitsetView& bitset) {
    auto base = ConvertFromDataTypeIfNeeded<DataType>(base_dataset);
    auto query = ConvertFromDataTypeIfNeeded<DataType>(query_dataset);

    auto xb = base->GetTensor();
    auto nb = base->GetRows();
    auto dim = base->GetDim();

    auto xq = query->GetTensor();
    auto nq = query->GetRows();
    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::ITERATOR, &msg);
    if (status != Status::success) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(status, msg);
    }
    std::string metric_str = cfg.metric_type.value();
    auto result = Str2FaissMetricType(metric_str);
    if (result.error() != Status::success) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(result.error(), result.what());
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto ctx = tracer::GetTraceCtxFromCfg(&cfg);
        span = tracer::StartSpan("knowhere bf ann iterator initialization", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, cfg.metric_type.value());
        span->SetAttribute(meta::ROWS, nb);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
#endif
    faiss::MetricType faiss_metric_type = result.value();
    bool is_cosine = IsMetricType(metric_str, metric::COSINE);

    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    auto vec = std::vector<IndexNode::IteratorPtr>(nq, nullptr);
    std::vector<folly::Future<Status>> futs;
    futs.reserve(nq);

    for (int i = 0; i < nq; ++i) {
        futs.emplace_back(pool->push([&, index = i] {
            ThreadPool::ScopedOmpSetter setter(1);

            BitsetViewIDSelector bw_idselector(bitset);
            faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;
            auto larger_is_closer = faiss::is_similarity_metric(faiss_metric_type) || is_cosine;
            auto max_dis = larger_is_closer ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
            std::vector<DistId> distances_ids(nb, {-1, max_dis});

            switch (faiss_metric_type) {
                case faiss::METRIC_L2: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::all_L2sqr(cur_query, (const float*)xb, dim, 1, nb, distances_ids, nullptr, id_selector);
                    break;
                }
                case faiss::METRIC_INNER_PRODUCT: {
                    auto cur_query = (const float*)xq + dim * index;
                    if (is_cosine) {
                        auto copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        faiss::all_cosine(copied_query.get(), (const float*)xb, nullptr, dim, 1, nb, distances_ids,
                                          id_selector);
                    } else {
                        faiss::all_inner_product(cur_query, (const float*)xb, dim, 1, nb, distances_ids, id_selector);
                    }
                    break;
                }
                default: {
                    LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << cfg.metric_type.value();
                    return Status::invalid_metric_type;
                }
            }
            vec[index] = std::make_shared<PrecomputedDistanceIterator>(std::move(distances_ids), larger_is_closer);

            return Status::success;
        }));
    }

    auto ret = WaitAllSuccess(futs);
    if (ret != Status::success) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(ret, "failed to brute force search for iterator");
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    if (cfg.trace_id.has_value()) {
        span->End();
    }
#endif
    return vec;
}

template <>
expected<std::vector<IndexNode::IteratorPtr>>
BruteForce::AnnIterator<knowhere::sparse::SparseRow<float>>(const DataSetPtr base_dataset,
                                                            const DataSetPtr query_dataset, const Json& config,
                                                            const BitsetView& bitset) {
    auto base = static_cast<const sparse::SparseRow<float>*>(base_dataset->GetTensor());
    auto rows = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();

    auto xq = static_cast<const sparse::SparseRow<float>*>(query_dataset->GetTensor());
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::ITERATOR, &msg);
    if (status != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Failed to load config: " << msg;
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(
            status, "Failed to brute force search sparse for iterator: failed to load config: " + msg);
    }

    std::string metric_str = cfg.metric_type.value();

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto ctx = tracer::GetTraceCtxFromCfg(&cfg);
        span = tracer::StartSpan("knowhere bf iterator sparse", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, metric_str);
        span->SetAttribute(meta::ROWS, rows);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
#endif

    auto result = Str2FaissMetricType(metric_str);
    if (result.error() != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << metric_str;
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(
            result.error(), "Failed to brute force search sparse for iterator: invalid metric type " + metric_str);
    }
    if (!IsMetricType(metric_str, metric::IP)) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << metric_str;
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(
            Status::invalid_metric_type,
            "Failed to brute force search sparse for iterator: invalid metric type " + metric_str);
    }

    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    auto vec = std::vector<IndexNode::IteratorPtr>(nq, nullptr);
    std::vector<folly::Future<folly::Unit>> futs;
    futs.reserve(nq);
    for (int64_t i = 0; i < nq; ++i) {
        futs.emplace_back(pool->push([&, index = i] {
            const auto& row = xq[index];
            std::vector<DistId> distances_ids;
            if (row.size() > 0) {
                for (int64_t j = 0; j < rows; ++j) {
                    if (!bitset.empty() && bitset.test(j)) {
                        continue;
                    }
                    auto dist = row.dot(base[j]);
                    if (dist > 0) {
                        distances_ids.emplace_back(j, dist);
                    }
                }
            }
            vec[index] = std::make_shared<PrecomputedDistanceIterator>(std::move(distances_ids), true);
        }));
    }
    WaitAllSuccess(futs);

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    if (cfg.trace_id.has_value()) {
        span->End();
    }
#endif

    return vec;
}

}  // namespace knowhere
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::Search<knowhere::fp32>(const knowhere::DataSetPtr base_dataset,
                                             const knowhere::DataSetPtr query_dataset, const knowhere::Json& config,
                                             const knowhere::BitsetView& bitset);
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::Search<knowhere::fp16>(const knowhere::DataSetPtr base_dataset,
                                             const knowhere::DataSetPtr query_dataset, const knowhere::Json& config,
                                             const knowhere::BitsetView& bitset);
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::Search<knowhere::bf16>(const knowhere::DataSetPtr base_dataset,
                                             const knowhere::DataSetPtr query_dataset, const knowhere::Json& config,
                                             const knowhere::BitsetView& bitset);
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::Search<knowhere::bin1>(const knowhere::DataSetPtr base_dataset,
                                             const knowhere::DataSetPtr query_dataset, const knowhere::Json& config,
                                             const knowhere::BitsetView& bitset);
template knowhere::Status
knowhere::BruteForce::SearchWithBuf<knowhere::fp32>(const knowhere::DataSetPtr base_dataset,
                                                    const knowhere::DataSetPtr query_dataset, int64_t* ids, float* dis,
                                                    const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::Status
knowhere::BruteForce::SearchWithBuf<knowhere::fp16>(const knowhere::DataSetPtr base_dataset,
                                                    const knowhere::DataSetPtr query_dataset, int64_t* ids, float* dis,
                                                    const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::Status
knowhere::BruteForce::SearchWithBuf<knowhere::bf16>(const knowhere::DataSetPtr base_dataset,
                                                    const knowhere::DataSetPtr query_dataset, int64_t* ids, float* dis,
                                                    const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::Status
knowhere::BruteForce::SearchWithBuf<knowhere::bin1>(const knowhere::DataSetPtr base_dataset,
                                                    const knowhere::DataSetPtr query_dataset, int64_t* ids, float* dis,
                                                    const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::RangeSearch<knowhere::fp32>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::RangeSearch<knowhere::fp16>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::RangeSearch<knowhere::bf16>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::RangeSearch<knowhere::bin1>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::expected<knowhere::DataSetPtr>
knowhere::BruteForce::RangeSearch<knowhere::sparse::SparseRow<float>>(const knowhere::DataSetPtr base_dataset,
                                                                      const knowhere::DataSetPtr query_dataset,
                                                                      const knowhere::Json& config,
                                                                      const knowhere::BitsetView& bitset);

template knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
knowhere::BruteForce::AnnIterator<knowhere::fp32>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
knowhere::BruteForce::AnnIterator<knowhere::fp16>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset);
template knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
knowhere::BruteForce::AnnIterator<knowhere::bf16>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset);
