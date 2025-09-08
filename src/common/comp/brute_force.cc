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
#include "faiss/utils/distances_typed.h"
#include "index/minhash/minhash_util.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/task.h"
#include "knowhere/config.h"
#include "knowhere/emb_list_utils.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"
#include "simd/hook.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/tracer.h"
#endif

namespace knowhere {

/* knowhere wrapper API to call faiss brute force search for all metric types */
/* If the ids of base_dataset does not start from 0, the BF functions will filter based on the real ids and return the
 * real ids.*/

class BruteForceConfig : public BaseConfig {};

namespace {

template <typename T>
expected<sparse::DocValueComputer<T>>
GetDocValueComputer(const BruteForceConfig& cfg) {
    if (IsMetricType(cfg.metric_type.value(), metric::IP)) {
        return sparse::GetDocValueOriginalComputer<T>();
    } else if (IsMetricType(cfg.metric_type.value(), metric::BM25)) {
        if (!cfg.bm25_k1.has_value() || !cfg.bm25_b.has_value() || !cfg.bm25_avgdl.has_value()) {
            return expected<sparse::DocValueComputer<T>>::Err(
                Status::invalid_args, "bm25_k1, bm25_b, bm25_avgdl must be set when searching for bm25 metric");
        }
        auto k1 = cfg.bm25_k1.value();
        auto b = cfg.bm25_b.value();
        auto avgdl = cfg.bm25_avgdl.value();
        avgdl = std::max(avgdl, 1.0f);
        return sparse::GetDocValueBM25Computer<T>(k1, b, avgdl);
    } else {
        return expected<sparse::DocValueComputer<T>>::Err(Status::invalid_metric_type,
                                                          "metric type not supported for sparse vector");
    }
}

template <typename DataType>
std::unique_ptr<float[]>
GetVecNorms(const DataSetPtr& base) {
    using NormComputer = float (*)(const DataType*, size_t);
    NormComputer norm_computer;
    if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
        norm_computer = faiss::fvec_norm_L2sqr;
    } else if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
        norm_computer = faiss::fp16_vec_norm_L2sqr;
    } else if constexpr (std::is_same_v<DataType, knowhere::bf16>) {
        norm_computer = faiss::bf16_vec_norm_L2sqr;
    } else {
        return nullptr;
    }
    auto xb = (DataType*)base->GetTensor();
    auto nb = base->GetRows();
    auto dim = base->GetDim();
    auto norms = std::make_unique<float[]>(nb);

    // use build thread pool to compute norms
    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futs;
    constexpr int64_t chunk_size = 8192;
    auto chunk_num = (nb + chunk_size - 1) / chunk_size;
    futs.reserve(chunk_num);
    for (auto i = 0; i < nb; i += chunk_size) {
        auto last = std::min(i + chunk_size, nb);
        futs.emplace_back(pool->push([&, beg_id = i, end_id = last] {
            for (auto j = beg_id; j < end_id; j++) {
                norms[j] = std::sqrt(norm_computer(xb + j * dim, dim));
            }
        }));
    }
    WaitAllSuccess(futs);
    return norms;
}
}  // namespace

template <typename DataType>
expected<DataSetPtr>
BruteForce::Search(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                   const BitsetView& bitset_) {
    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::SEARCH, &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, msg);
    }
    auto nq = query_dataset->GetRows();
    int topk = cfg.k.value();
    auto labels = std::make_unique<int64_t[]>(nq * topk);
    auto distances = std::make_unique<float[]>(nq * topk);

    auto search_status =
        SearchWithBuf<DataType>(base_dataset, query_dataset, labels.get(), distances.get(), config, bitset_);
    if (search_status != Status::success) {
        return expected<DataSetPtr>::Err(search_status, "search with buf failed");
    }
    auto res = GenResultDataSet(nq, cfg.k.value(), std::move(labels), std::move(distances));

    return res;
}

template <typename DataType>
Status
BruteForce::SearchWithBuf(const DataSetPtr base_dataset, const DataSetPtr query_dataset, int64_t* ids, float* dis,
                          const Json& config, const BitsetView& bitset_) {
    auto xb = base_dataset->GetTensor();
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();
    bool is_emb_list = base_dataset->GetLims() != nullptr;
    auto xb_id_offset = base_dataset->GetTensorBeginId();
    BitsetView bitset = bitset_;
    bitset.set_id_offset(xb_id_offset);

    auto xq = query_dataset->GetTensor();
    auto nq = query_dataset->GetRows();
    bool query_is_emb_list = query_dataset->GetLims() != nullptr;
    if (is_emb_list != query_is_emb_list) {
        LOG_KNOWHERE_ERROR_ << "base dataset and query must be both emb_list or not";
        return Status::invalid_args;
    }

    BruteForceConfig cfg;
    RETURN_IF_ERROR(Config::Load(cfg, config, knowhere::SEARCH));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    // LCOV_EXCL_START
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto trace_id_str = tracer::GetIDFromHexStr(cfg.trace_id.value());
        auto span_id_str = tracer::GetIDFromHexStr(cfg.span_id.value());
        auto ctx = tracer::TraceContext{(uint8_t*)trace_id_str.c_str(), (uint8_t*)span_id_str.c_str(),
                                        (uint8_t)cfg.trace_flags.value()};
        span = tracer::StartSpan("knowhere bf search with buf", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, cfg.metric_type.value());
        span->SetAttribute(meta::TOPK, cfg.k.value());
        span->SetAttribute(meta::ROWS, nb);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
    // LCOV_EXCL_STOP
#endif

    std::string metric_str = cfg.metric_type.value();
    auto topk = cfg.k.value();

    if (is_emb_list) {
        auto el_metric_type_or = get_el_metric_type(metric_str);
        if (!el_metric_type_or.has_value()) {
            LOG_KNOWHERE_ERROR_ << "metric type not supported for emb_list: " << metric_str;
            return Status::invalid_metric_type;
        }
        auto el_metric_type = el_metric_type_or.value();
        auto el_agg_func_or = get_emb_list_agg_func(el_metric_type);
        if (!el_agg_func_or.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid emb list aggeration function for metric type: " << el_metric_type;
            return Status::brute_force_inner_error;
        }
        auto el_agg_func = el_agg_func_or.value();
        auto el_sub_metric_type_or = get_sub_metric_type(el_metric_type);
        if (!el_sub_metric_type_or.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid emb list sub metric type for metric type: " << el_metric_type;
            return Status::brute_force_inner_error;
        }
        auto el_sub_metric_type = el_sub_metric_type_or.value();
        bool is_cosine = IsMetricType(el_sub_metric_type, metric::COSINE);
        auto base_el_offset = EmbListOffset(base_dataset->GetLims(), nb);
        auto query_el_offset = EmbListOffset(query_dataset->GetLims(), nq);
        auto num_base_el = base_el_offset.num_el();
        auto num_query_el = query_el_offset.num_el();

        auto pool = ThreadPool::GetGlobalSearchThreadPool();
        std::vector<folly::Future<Status>> futs;
        futs.reserve(num_query_el);

        for (size_t query_el_i = 0; query_el_i < num_query_el; query_el_i++) {
            futs.emplace_back(pool->push([&, query_el_idx = query_el_i] {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                std::priority_queue<DistId, std::vector<DistId>, std::greater<>> minheap;
                for (size_t base_el_idx = 0; base_el_idx < num_base_el; base_el_idx++) {
                    if (bitset.empty() || !bitset.test(base_el_idx)) {
                        auto num_base_vectors =
                            base_el_offset.offset[base_el_idx + 1] - base_el_offset.offset[base_el_idx];
                        assert(num_base_vectors >= 0);
                        auto num_query_vectors =
                            query_el_offset.offset[query_el_idx + 1] - query_el_offset.offset[query_el_idx];
                        assert(num_query_vectors >= 0);
                        auto distances = std::make_unique<float[]>(num_query_vectors * num_base_vectors);

                        auto cur_query = (const DataType*)xq + query_el_offset.offset[query_el_idx] * dim;
                        auto cur_base = (const DataType*)xb + base_el_offset.offset[base_el_idx] * dim;

                        if (is_cosine) {
                            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                                faiss::all_cosine_distances(cur_query, cur_base, nullptr, dim, num_query_vectors,
                                                            num_base_vectors, distances.get(), nullptr);
                            } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                                faiss::all_cosine_distances_typed(cur_query, cur_base, nullptr, dim, num_query_vectors,
                                                                  num_base_vectors, distances.get(), nullptr);
                            } else {
                                LOG_KNOWHERE_ERROR_ << "Metric COSINE not supported for current vector type";
                                return Status::faiss_inner_error;
                            }
                        } else {
                            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                                faiss::all_inner_product_distances(cur_query, cur_base, dim, num_query_vectors,
                                                                   num_base_vectors, distances.get(), nullptr);
                            } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                                faiss::all_inner_product_distances_typed(cur_query, cur_base, dim, num_query_vectors,
                                                                         num_base_vectors, distances.get(), nullptr);
                            } else {
                                LOG_KNOWHERE_ERROR_ << "Metric IP not supported for current vector type";
                                return Status::faiss_inner_error;
                            }
                        }

                        auto score_or = el_agg_func(distances.get(), num_query_vectors, num_base_vectors);
                        if (!score_or.has_value()) {
                            LOG_KNOWHERE_WARNING_ << "get_sum_max_sim failed, num_query_vectors: " << num_query_vectors
                                                  << ", num_base_vectors: " << num_base_vectors;
                            return Status::brute_force_inner_error;
                        }
                        auto score = score_or.value();
                        if (minheap.size() < (size_t)topk) {
                            minheap.emplace((int64_t)base_el_idx + xb_id_offset, score);
                        } else {
                            if (score > minheap.top().val) {
                                minheap.pop();
                                minheap.emplace((int64_t)base_el_idx + xb_id_offset, score);
                            }
                        }
                    }
                }
                auto real_el_k = minheap.size();
                for (size_t j = 0; j < real_el_k; j++) {
                    auto& a = minheap.top();
                    ids[query_el_idx * topk + real_el_k - j - 1] = a.id;
                    dis[query_el_idx * topk + real_el_k - j - 1] = a.val;
                    minheap.pop();
                }
                for (size_t j = real_el_k; j < (size_t)topk; j++) {
                    ids[query_el_idx * topk + j] = -1;
                    dis[query_el_idx * topk + j] = std::numeric_limits<float>::min();
                }

                return Status::success;
            }));
        }

        RETURN_IF_ERROR(WaitAllSuccess(futs));
    } else if (IsMetricType(metric_str, metric::MHJACCARD)) {
        auto labels = ids;
        auto distances = dis;
        size_t mh_vec_size_in_bytes = dim / 8;
        size_t mh_vec_element_size_in_bytes = cfg.mh_element_bit_width.value() / 8;
        size_t mh_vec_dim = mh_vec_size_in_bytes / mh_vec_element_size_in_bytes;
        size_t mh_lsh_band = cfg.mh_lsh_band.value();
        auto [b, r] = minhash::OptimizeMinHashLSHParams(mh_vec_dim, mh_lsh_band);
        mh_lsh_band = b;
        size_t mh_lsh_r = r;
        LOG_KNOWHERE_DEBUG_ << "Search with minhash vector with band_num, band_size = [" << mh_lsh_band << " "
                            << mh_lsh_r << "]" << std::endl;
        bool mh_search_with_jaccard = cfg.mh_search_with_jaccard.value();
        auto mh_valid_stat = minhash::MinhashConfigCheck(dim, datatype_v<DataType>,
                                                         PARAM_TYPE::SEARCH | PARAM_TYPE::TRAIN, &cfg, &bitset);
        if (mh_valid_stat != Status::success) {
            return mh_valid_stat;
        }
        auto search_status = minhash::MinHashVecSearch((const char*)xq, (const char*)xb, mh_vec_size_in_bytes,
                                                       mh_vec_element_size_in_bytes, mh_lsh_band, mh_lsh_r, nq, nb,
                                                       topk, mh_search_with_jaccard, bitset, distances, labels);
        if (search_status != Status::success) {
            return search_status;
        }
        if (xb_id_offset != 0) {
            for (auto i = 0; i < nq * topk; i++) {
                labels[i] = labels[i] == -1 ? -1 : labels[i] + xb_id_offset;
            }
        }
    } else {
        auto result = Str2FaissMetricType(cfg.metric_type.value());
        if (result.error() != Status::success) {
            return result.error();
        }
        faiss::MetricType faiss_metric_type = result.value();
        bool is_cosine = IsMetricType(metric_str, metric::COSINE);

        auto labels = ids;
        auto distances = dis;

        std::unique_ptr<float[]> norms = is_cosine ? GetVecNorms<DataType>(base_dataset) : nullptr;
        auto pool = ThreadPool::GetGlobalSearchThreadPool();
        std::vector<folly::Future<Status>> futs;
        futs.reserve(nq);
        for (int i = 0; i < nq; ++i) {
            futs.emplace_back(pool->push([&, index = i] {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                auto cur_labels = labels + topk * index;
                auto cur_distances = distances + topk * index;

                BitsetViewIDSelector bw_idselector(bitset);
                faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;
                switch (faiss_metric_type) {
                    case faiss::METRIC_L2: {
                        [[maybe_unused]] auto cur_query = (const DataType*)xq + dim * index;
                        if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                            faiss::knn_L2sqr(cur_query, (const float*)xb, dim, 1, nb, topk, cur_distances, cur_labels,
                                             nullptr, id_selector);
                        } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                            faiss::knn_L2sqr_typed(cur_query, (const DataType*)xb, dim, 1, nb, topk, cur_distances,
                                                   cur_labels, nullptr, id_selector);
                        } else {
                            LOG_KNOWHERE_ERROR_ << "Metric L2 not supported for current vector type";
                            return Status::faiss_inner_error;
                        }
                        break;
                    }
                    case faiss::METRIC_INNER_PRODUCT: {
                        [[maybe_unused]] auto cur_query = (const DataType*)xq + dim * index;
                        if (is_cosine) {
                            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                                auto copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                                faiss::knn_cosine(copied_query.get(), (const float*)xb, norms.get(), dim, 1, nb, topk,
                                                  cur_distances, cur_labels, id_selector);
                            } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                                // normalize query vector may cause precision loss, so div query norms in apply
                                // function
                                faiss::knn_cosine_typed(cur_query, (const DataType*)xb, norms.get(), dim, 1, nb, topk,
                                                        cur_distances, cur_labels, id_selector);
                            } else {
                                LOG_KNOWHERE_ERROR_ << "Metric COSINE not supported for current vector type";
                                return Status::faiss_inner_error;
                            }
                        } else {
                            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                                faiss::knn_inner_product(cur_query, (const float*)xb, dim, 1, nb, topk, cur_distances,
                                                         cur_labels, id_selector);
                            } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                                faiss::knn_inner_product_typed(cur_query, (const DataType*)xb, dim, 1, nb, topk,
                                                               cur_distances, cur_labels, id_selector);
                            } else {
                                LOG_KNOWHERE_ERROR_ << "Metric IP not supported for current vector type";
                                return Status::faiss_inner_error;
                            }
                        }
                        break;
                    }
                    case faiss::METRIC_Jaccard: {
                        auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                        faiss::float_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, cur_distances};
                        binary_knn_hc(faiss::METRIC_Jaccard, &res, cur_query, (const uint8_t*)xb, nb, dim / 8,
                                      id_selector);
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
                        binary_knn_mc(faiss_metric_type, cur_query, (const uint8_t*)xb, 1, nb, topk, dim / 8,
                                      cur_distances, cur_labels, id_selector);
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

        if (xb_id_offset != 0) {
            for (auto i = 0; i < nq * topk; i++) {
                labels[i] = labels[i] == -1 ? -1 : labels[i] + xb_id_offset;
            }
        }
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    // LCOV_EXCL_START
    if (cfg.trace_id.has_value()) {
        span->End();
    }
    // LCOV_EXCL_STOP
#endif

    return Status::success;
}

/** knowhere wrapper API to call faiss brute force range search for all metric types
 */
template <typename DataType>
expected<DataSetPtr>
BruteForce::RangeSearch(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                        const BitsetView& bitset_) {
    DataSetPtr query(query_dataset);
    auto xb = base_dataset->GetTensor();
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();
    auto xb_id_offset = base_dataset->GetTensorBeginId();
    BitsetView bitset = bitset_;
    bitset.set_id_offset(xb_id_offset);
    auto xq = query_dataset->GetTensor();
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    std::string msg;
    auto status = Config::Load(cfg, config, knowhere::RANGE_SEARCH, &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, std::move(msg));
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    // LCOV_EXCL_START
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto trace_id_str = tracer::GetIDFromHexStr(cfg.trace_id.value());
        auto span_id_str = tracer::GetIDFromHexStr(cfg.span_id.value());
        auto ctx = tracer::TraceContext{(uint8_t*)trace_id_str.c_str(), (uint8_t*)span_id_str.c_str(),
                                        (uint8_t)cfg.trace_flags.value()};
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
    // LCOV_EXCL_STOP
#endif

    std::string metric_str = cfg.metric_type.value();
    const bool is_bm25 = IsMetricType(metric_str, metric::BM25);

    faiss::MetricType faiss_metric_type;
    sparse::DocValueComputer<float> sparse_computer;
    if constexpr (!std::is_same_v<DataType, knowhere::sparse::SparseRow<float>>) {
        auto result = Str2FaissMetricType(metric_str);
        if (result.error() != Status::success) {
            return expected<DataSetPtr>::Err(result.error(), result.what());
        }
        faiss_metric_type = result.value();
    } else {
        auto computer_or = GetDocValueComputer<float>(cfg);
        if (!computer_or.has_value()) {
            return expected<DataSetPtr>::Err(computer_or.error(), computer_or.what());
        }
        sparse_computer = computer_or.value();
    }

    bool is_cosine = IsMetricType(metric_str, metric::COSINE);

    const bool the_larger_the_closer = IsMetricType(metric_str, metric::IP) ||
                                       IsMetricType(metric_str, metric::COSINE) ||
                                       IsMetricType(metric_str, metric::BM25);
    auto radius = cfg.radius.value();
    float range_filter = cfg.range_filter.value();

    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    // some check for minhash metric
    if (metric_str == metric::MHJACCARD) {
        return expected<DataSetPtr>::Err(Status::not_implemented, "minhash not support range search.");
    }

    std::vector<std::vector<int64_t>> result_id_array(nq);
    std::vector<std::vector<float>> result_dist_array(nq);

    std::unique_ptr<float[]> norms = is_cosine ? GetVecNorms<DataType>(base_dataset) : nullptr;
    std::vector<folly::Future<Status>> futs;
    futs.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        futs.emplace_back(pool->push([&, index = i] {
            if constexpr (std::is_same_v<DataType, knowhere::sparse::SparseRow<float>>) {
                auto cur_query = (const sparse::SparseRow<float>*)xq + index;
                auto xb_sparse = (const sparse::SparseRow<float>*)xb;
                std::set<std::pair<float, int64_t>, std::greater<>> result;
                for (int j = 0; j < nb; ++j) {
                    auto xid = xb_id_offset + j;
                    // bitset has already set the id_offset, so we need to use j instead of xid
                    if (!bitset.empty() && bitset.test(j)) {
                        continue;
                    }
                    float row_sum = 0;
                    if (is_bm25) {
                        for (size_t k = 0; k < xb_sparse[j].size(); ++k) {
                            auto [d, v] = xb_sparse[j][k];
                            row_sum += v;
                        }
                    }
                    auto dist = cur_query->dot(xb_sparse[j], sparse_computer, row_sum);
                    if (dist > radius && dist <= range_filter) {
                        result.insert({dist, xid});
                    }
                }
                result_id_array[index].reserve(result.size());
                result_dist_array[index].reserve(result.size());
                for (auto& [dist, id] : result) {
                    result_id_array[index].push_back(id);
                    result_dist_array[index].push_back(dist);
                }
                return Status::success;
            } else {
                // else not sparse:
                ThreadPool::ScopedSearchOmpSetter setter(1);
                faiss::RangeSearchResult res(1);

                BitsetViewIDSelector bw_idselector(bitset);
                faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;
                switch (faiss_metric_type) {
                    case faiss::METRIC_L2: {
                        [[maybe_unused]] auto cur_query = (const DataType*)xq + dim * index;
                        if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                            faiss::range_search_L2sqr(cur_query, (const float*)xb, dim, 1, nb, radius, &res,
                                                      id_selector);
                        } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                            faiss::range_search_L2sqr_typed(cur_query, (const DataType*)xb, dim, 1, nb, radius, &res,
                                                            id_selector);
                        } else {
                            LOG_KNOWHERE_ERROR_ << "Metric L2 not supported for current vector type";
                            return Status::faiss_inner_error;
                        }
                        break;
                    }
                    case faiss::METRIC_INNER_PRODUCT: {
                        [[maybe_unused]] auto cur_query = (const DataType*)xq + dim * index;
                        if (is_cosine) {
                            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                                auto copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                                faiss::range_search_cosine(copied_query.get(), (const float*)xb, norms.get(), dim, 1,
                                                           nb, radius, &res, id_selector);
                            } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                                // normalize query vector may cause precision loss, so div query norms in apply function
                                faiss::range_search_cosine_typed(cur_query, (const DataType*)xb, norms.get(), dim, 1,
                                                                 nb, radius, &res, id_selector);
                            } else {
                                LOG_KNOWHERE_ERROR_ << "Metric COSINE not supported for current vector type";
                                return Status::faiss_inner_error;
                            }
                        } else {
                            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                                faiss::range_search_inner_product(cur_query, (const DataType*)xb, dim, 1, nb, radius,
                                                                  &res, id_selector);
                            } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                                faiss::range_search_inner_product_typed(cur_query, (const DataType*)xb, dim, 1, nb,
                                                                        radius, &res, id_selector);
                            } else {
                                LOG_KNOWHERE_ERROR_ << "Metric IP not supported for current vector type";
                                return Status::faiss_inner_error;
                            }
                        }
                        break;
                    }
                    case faiss::METRIC_Jaccard: {
                        auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                        faiss::binary_range_search<faiss::CMin<float, int64_t>, float>(
                            faiss::METRIC_Jaccard, cur_query, (const uint8_t*)xb, 1, nb, radius, dim / 8, &res,
                            id_selector);
                        break;
                    }
                    case faiss::METRIC_Hamming: {
                        auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                        faiss::binary_range_search<faiss::CMin<int, int64_t>, int>(
                            faiss::METRIC_Hamming, cur_query, (const uint8_t*)xb, 1, nb, (int)radius, dim / 8, &res,
                            id_selector);
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
                    result_id_array[index][j] = res.labels[j] + xb_id_offset;
                }
                if (cfg.range_filter.value() != defaultRangeFilter) {
                    FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index],
                                                    faiss_metric_type == faiss::METRIC_INNER_PRODUCT, radius,
                                                    range_filter);
                }
                return Status::success;
            }
        }));
    }
    auto ret = WaitAllSuccess(futs);
    if (ret != Status::success) {
        return expected<DataSetPtr>::Err(ret, "failed to brute force search");
    }

    auto range_search_result =
        GetRangeSearchResult(result_dist_array, result_id_array, the_larger_the_closer, nq, radius, range_filter);
    auto res = GenResultDataSet(nq, std::move(range_search_result));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    // LCOV_EXCL_START
    if (cfg.trace_id.has_value()) {
        span->End();
    }
    // LCOV_EXCL_STOP
#endif

    return res;
}

Status
BruteForce::SearchSparseWithBuf(const DataSetPtr base_dataset, const DataSetPtr query_dataset, sparse::label_t* labels,
                                float* distances, const Json& config, const BitsetView& bitset) {
    auto base = static_cast<const sparse::SparseRow<float>*>(base_dataset->GetTensor());
    auto rows = base_dataset->GetRows();
    auto xb_id_offset = base_dataset->GetTensorBeginId();

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
    // LCOV_EXCL_START
    auto dim = base_dataset->GetDim();
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto trace_id_str = tracer::GetIDFromHexStr(cfg.trace_id.value());
        auto span_id_str = tracer::GetIDFromHexStr(cfg.span_id.value());
        auto ctx = tracer::TraceContext{(uint8_t*)trace_id_str.c_str(), (uint8_t*)span_id_str.c_str(),
                                        (uint8_t)cfg.trace_flags.value()};
        span = tracer::StartSpan("knowhere bf search sparse with buf", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, cfg.metric_type.value());
        span->SetAttribute(meta::TOPK, cfg.k.value());
        span->SetAttribute(meta::ROWS, rows);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
    // LCOV_EXCL_STOP
#endif

    std::string metric_str = cfg.metric_type.value();
    const bool is_bm25 = IsMetricType(metric_str, metric::BM25);

    auto computer_or = GetDocValueComputer<float>(cfg);
    if (!computer_or.has_value()) {
        return computer_or.error();
    }
    auto computer = computer_or.value();

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
                auto x_id = j + xb_id_offset;
                if (!bitset.empty() && bitset.test(x_id)) {
                    continue;
                }
                float row_sum = 0;
                if (is_bm25) {
                    for (size_t k = 0; k < base[j].size(); ++k) {
                        auto [d, v] = base[j][k];
                        row_sum += v;
                    }
                }
                float dist = row.dot(base[j], computer, row_sum);
                if (dist > 0) {
                    heap.push(x_id, dist);
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
    // LCOV_EXCL_START
    if (cfg.trace_id.has_value()) {
        span->End();
    }
    // LCOV_EXCL_STOP
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
                        const BitsetView& bitset_, bool use_knowhere_search_pool) {
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();
    auto nq = query_dataset->GetRows();
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

    // some check for minhash metric
    if (metric_str == metric::MHJACCARD) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::not_implemented,
                                                                  "minhash does not support iterator.");
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    // LCOV_EXCL_START
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto trace_id_str = tracer::GetIDFromHexStr(cfg.trace_id.value());
        auto span_id_str = tracer::GetIDFromHexStr(cfg.span_id.value());
        auto ctx = tracer::TraceContext{(uint8_t*)trace_id_str.c_str(), (uint8_t*)span_id_str.c_str(),
                                        (uint8_t)cfg.trace_flags.value()};
        span = tracer::StartSpan("knowhere bf ann iterator initialization", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, cfg.metric_type.value());
        span->SetAttribute(meta::ROWS, nb);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
    // LCOV_EXCL_STOP
#endif
    faiss::MetricType faiss_metric_type = result.value();
    bool is_cosine = IsMetricType(metric_str, metric::COSINE);
    auto larger_is_closer = faiss::is_similarity_metric(faiss_metric_type) || is_cosine;
    auto vec = std::vector<IndexNode::IteratorPtr>(nq, nullptr);
    std::shared_ptr<float[]> norms = GetVecNorms<DataType>(base_dataset);

    try {
        for (int i = 0; i < nq; ++i) {
            // Heavy computations with `compute_dist_func` will be deferred until the first call to 'Iterator->Next()'.
            auto compute_dist_func = [=]() -> std::vector<DistId> {
                auto xb = base_dataset->GetTensor();
                auto xq = query_dataset->GetTensor();
                auto xb_id_offset = base_dataset->GetTensorBeginId();
                BitsetView bitset = bitset_;
                bitset.set_id_offset(xb_id_offset);
                BitsetViewIDSelector bw_idselector(bitset);
                [[maybe_unused]] faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;
                auto max_dis =
                    larger_is_closer ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
                std::vector<DistId> distances_ids(nb, {-1, max_dis});
                [[maybe_unused]] auto cur_query = (const DataType*)xq + dim * i;
                switch (faiss_metric_type) {
                    case faiss::METRIC_L2: {
                        if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                            faiss::all_L2sqr(cur_query, (const float*)xb, dim, 1, nb, distances_ids, nullptr,
                                             id_selector);
                        } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                            faiss::all_L2sqr_typed(cur_query, (const DataType*)xb, dim, 1, nb, distances_ids, nullptr,
                                                   id_selector);
                        } else {
                            std::string err_msg = "Metric L2 not supported for current vector type";
                            LOG_KNOWHERE_ERROR_ << err_msg;
                            KNOWHERE_THROW_MSG(err_msg);
                        }
                        break;
                    }
                    case faiss::METRIC_INNER_PRODUCT: {
                        if (is_cosine) {
                            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                                auto copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                                faiss::all_cosine(copied_query.get(), (const float*)xb, norms.get(), dim, 1, nb,
                                                  distances_ids, id_selector);
                            } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                                // normalize query vector may cause precision loss, so div query norms in apply function
                                faiss::all_cosine_typed(cur_query, (const DataType*)xb, norms.get(), dim, 1, nb,
                                                        distances_ids, id_selector);
                            } else {
                                std::string err_msg = "Metric COSINE not supported for current vector type";
                                LOG_KNOWHERE_ERROR_ << err_msg;
                                KNOWHERE_THROW_MSG(err_msg);
                            }
                        } else {
                            if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
                                faiss::all_inner_product(cur_query, (const float*)xb, dim, 1, nb, distances_ids,
                                                         id_selector);
                            } else if constexpr (KnowhereLowPrecisionTypeCheck<DataType>::value) {
                                faiss::all_inner_product_typed(cur_query, (const DataType*)xb, dim, 1, nb,
                                                               distances_ids, id_selector);
                            } else {
                                std::string err_msg = "Metric IP not supported for current vector type";
                                LOG_KNOWHERE_ERROR_ << err_msg;
                                KNOWHERE_THROW_MSG(err_msg);
                            }
                        }
                        break;
                    }
                    default: {
                        std::string err_msg = "Invalid metric type: " + cfg.metric_type.value();
                        LOG_KNOWHERE_ERROR_ << err_msg;
                        KNOWHERE_THROW_MSG(err_msg);
                    }
                }
                if (xb_id_offset != 0) {
                    for (auto& distances_id : distances_ids) {
                        distances_id.id = distances_id.id == -1 ? -1 : distances_id.id + xb_id_offset;
                    }
                }
                return distances_ids;
            };
            vec[i] = std::make_shared<PrecomputedDistanceIterator>(compute_dist_func, larger_is_closer,
                                                                   use_knowhere_search_pool);
        }
    } catch (const std::exception& e) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::brute_force_inner_error, e.what());
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    // LCOV_EXCL_START
    if (cfg.trace_id.has_value()) {
        span->End();
    }
    // LCOV_EXCL_STOP
#endif
    return vec;
}

template <>
expected<std::vector<IndexNode::IteratorPtr>>
BruteForce::AnnIterator<knowhere::sparse::SparseRow<float>>(const DataSetPtr base_dataset,
                                                            const DataSetPtr query_dataset, const Json& config,
                                                            const BitsetView& bitset, bool use_knowhere_search_pool) {
    auto rows = base_dataset->GetRows();
    auto xb_id_offset = base_dataset->GetTensorBeginId();
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
    // LCOV_EXCL_START
    auto dim = base_dataset->GetDim();
    std::shared_ptr<tracer::trace::Span> span = nullptr;
    if (cfg.trace_id.has_value()) {
        auto trace_id_str = tracer::GetIDFromHexStr(cfg.trace_id.value());
        auto span_id_str = tracer::GetIDFromHexStr(cfg.span_id.value());
        auto ctx = tracer::TraceContext{(uint8_t*)trace_id_str.c_str(), (uint8_t*)span_id_str.c_str(),
                                        (uint8_t)cfg.trace_flags.value()};
        span = tracer::StartSpan("knowhere bf iterator sparse", &ctx);
        span->SetAttribute(meta::METRIC_TYPE, metric_str);
        span->SetAttribute(meta::ROWS, rows);
        span->SetAttribute(meta::DIM, dim);
        span->SetAttribute(meta::NQ, nq);
    }
    // LCOV_EXCL_STOP
#endif

    const bool is_bm25 = IsMetricType(metric_str, metric::BM25);

    auto computer_or = GetDocValueComputer<float>(cfg);
    if (!computer_or.has_value()) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(computer_or.error(), computer_or.what());
    }
    auto computer = computer_or.value();

    auto vec = std::vector<IndexNode::IteratorPtr>(nq, nullptr);
    try {
        for (int64_t i = 0; i < nq; ++i) {
            // Heavy computations with `compute_dist_func` will be deferred until the first call to 'Iterator->Next()'.
            auto compute_dist_func = [=]() -> std::vector<DistId> {
                auto xq = static_cast<const sparse::SparseRow<float>*>(query_dataset->GetTensor());
                auto base = static_cast<const sparse::SparseRow<float>*>(base_dataset->GetTensor());
                const auto& row = xq[i];
                std::vector<DistId> distances_ids;
                if (row.size() > 0) {
                    for (int64_t j = 0; j < rows; ++j) {
                        auto xb_id = j + xb_id_offset;
                        if (!bitset.empty() && bitset.test(xb_id)) {
                            continue;
                        }
                        float row_sum = 0;
                        if (is_bm25) {
                            for (size_t k = 0; k < base[j].size(); ++k) {
                                auto [d, v] = base[j][k];
                                row_sum += v;
                            }
                        }
                        auto dist = row.dot(base[j], computer, row_sum);
                        if (dist > 0) {
                            distances_ids.emplace_back(xb_id, dist);
                        }
                    }
                }
                return distances_ids;
            };

            vec[i] = std::make_shared<PrecomputedDistanceIterator>(compute_dist_func, true, use_knowhere_search_pool);
        }
    } catch (const std::exception& e) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::brute_force_inner_error, e.what());
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    // LCOV_EXCL_START
    if (cfg.trace_id.has_value()) {
        span->End();
    }
    // LCOV_EXCL_STOP
#endif

    return vec;
}

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
knowhere::BruteForce::Search<knowhere::int8>(const knowhere::DataSetPtr base_dataset,
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
knowhere::BruteForce::SearchWithBuf<knowhere::int8>(const knowhere::DataSetPtr base_dataset,
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
knowhere::BruteForce::RangeSearch<knowhere::int8>(const knowhere::DataSetPtr base_dataset,
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
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset,
                                                  bool use_knowhere_search_pool = true);
template knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
knowhere::BruteForce::AnnIterator<knowhere::fp16>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset,
                                                  bool use_knowhere_search_pool = true);
template knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
knowhere::BruteForce::AnnIterator<knowhere::bf16>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset,
                                                  bool use_knowhere_search_pool = true);
template knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
knowhere::BruteForce::AnnIterator<knowhere::int8>(const knowhere::DataSetPtr base_dataset,
                                                  const knowhere::DataSetPtr query_dataset,
                                                  const knowhere::Json& config, const knowhere::BitsetView& bitset,
                                                  bool use_knowhere_search_pool = true);

}  // namespace knowhere
