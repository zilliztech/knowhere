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
#include "common/range_util.h"
#include "faiss/MetricType.h"
#include "faiss/utils/binary_distances.h"
#include "faiss/utils/distances.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/config.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

/* knowhere wrapper API to call faiss brute force search for all metric types */

class BruteForceConfig : public BaseConfig {};

expected<DataSetPtr>
BruteForce::Search(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                   const BitsetView& bitset) {
    auto xb = base_dataset->GetTensor();
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();

    auto xq = query_dataset->GetTensor();
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::SEARCH, &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, msg);
    }

    std::string metric_str = cfg.metric_type.value();
    auto result = Str2FaissMetricType(metric_str);
    if (result.error() != Status::success) {
        return expected<DataSetPtr>::Err(result.error(), result.what());
    }
    faiss::MetricType faiss_metric_type = result.value();
    bool is_cosine = IsMetricType(metric_str, metric::COSINE);

    int topk = cfg.k.value();
    auto labels = new int64_t[nq * topk];
    auto distances = new float[nq * topk];

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
    for (auto& fut : futs) {
        fut.wait();
        auto ret = fut.result().value();
        if (ret != Status::success) {
            return expected<DataSetPtr>::Err(ret, "failed to brute force search");
        }
    }
    return GenResultDataSet(nq, cfg.k.value(), labels, distances);
}

Status
BruteForce::SearchWithBuf(const DataSetPtr base_dataset, const DataSetPtr query_dataset, int64_t* ids, float* dis,
                          const Json& config, const BitsetView& bitset) {
    auto xb = base_dataset->GetTensor();
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();

    auto xq = query_dataset->GetTensor();
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    RETURN_IF_ERROR(Config::Load(cfg, config, knowhere::SEARCH));

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
    for (auto& fut : futs) {
        fut.wait();
        auto ret = fut.result().value();
        RETURN_IF_ERROR(ret);
    }
    return Status::success;
}

/** knowhere wrapper API to call faiss brute force range search for all metric types
 */
expected<DataSetPtr>
BruteForce::RangeSearch(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                        const BitsetView& bitset) {
    auto xb = base_dataset->GetTensor();
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();

    auto xq = query_dataset->GetTensor();
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::RANGE_SEARCH, &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, std::move(msg));
    }

    std::string metric_str = cfg.metric_type.value();
    auto result = Str2FaissMetricType(metric_str);
    if (result.error() != Status::success) {
        return expected<DataSetPtr>::Err(result.error(), result.what());
    }
    faiss::MetricType faiss_metric_type = result.value();
    bool is_cosine = IsMetricType(metric_str, metric::COSINE);

    auto radius = cfg.radius.value();
    bool is_ip = false;
    float range_filter = cfg.range_filter.value();

    auto pool = ThreadPool::GetGlobalSearchThreadPool();

    std::vector<std::vector<int64_t>> result_id_array(nq);
    std::vector<std::vector<float>> result_dist_array(nq);
    std::vector<size_t> result_size(nq);
    std::vector<size_t> result_lims(nq + 1);
    std::vector<folly::Future<Status>> futs;
    futs.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        futs.emplace_back(pool->push([&, index = i] {
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
            result_size[index] = elem_cnt;
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
    for (auto& fut : futs) {
        fut.wait();
        auto ret = fut.result().value();
        if (ret != Status::success) {
            return expected<DataSetPtr>::Err(ret, "failed to brute force search");
        }
    }

    int64_t* ids = nullptr;
    float* distances = nullptr;
    size_t* lims = nullptr;
    GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius, range_filter, distances, ids, lims);
    return GenResultDataSet(nq, ids, distances, lims);
}
}  // namespace knowhere
