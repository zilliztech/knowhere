// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// knowhere-specific indices
#pragma once

#include <knowhere/bitsetview.h>
#include <knowhere/range_util.h>

#include <atomic>
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/ResultHandler.h"
#include "faiss/utils/distances_if.h"
#include "index/data_view_dense_index/refine_computer.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/task.h"
#include "knowhere/config.h"
#include "knowhere/context.h"
#include "knowhere/operands.h"
#include "knowhere/range_util.h"
namespace knowhere {
using CMAX = faiss::CMax<float, idx_t>;
using CMIN = faiss::CMin<float, idx_t>;
struct RangeSearchResult;
/*
DataViewIndexBase is is an index base class.
This kind of index will not hold raw data by itself, and it will use ViewDataOp to access raw data.
DataViewIndexBase only keep index meta and data codes(!= raw data) in memory.
*/
class DataViewIndexBase {
 public:
    DataViewIndexBase(idx_t d, DataFormatEnum data_type, MetricType metric_type, ViewDataOp view, bool is_cosine,
                      RefineType refine_type, std::optional<int> build_thread_num)
        : d_(d),
          data_type_(data_type),
          metric_type_(metric_type),
          view_data_(view),
          is_cosine_(is_cosine),
          refine_type_(refine_type),
          build_thread_num_(build_thread_num) {
        if (metric_type != metric::L2 && metric_type != metric::IP) {
            throw std::runtime_error("DataViewIndexBase only support L2 or IP.");
        }
        if (data_type_ == DataFormatEnum::fp32) {
            code_size_ = sizeof(fp32) * d_;
        } else if (data_type_ == DataFormatEnum::fp16) {
            code_size_ = sizeof(fp16) * d_;
        } else if (data_type_ == DataFormatEnum::bf16) {
            code_size_ = sizeof(bf16) * d_;
        } else {
            throw std::runtime_error("data view index only support float data type.");
        }
        // use memory quant data to refine
        if (refine_type_ != RefineType::DATA_VIEW) {
            quant_data_ = std::shared_ptr<QuantRefine>(new QuantRefine(d, data_type_, refine_type_, metric_type_));
            if ((refine_type_ == RefineType::BFLOAT16_QUANT && data_type == DataFormatEnum::fp16) ||
                (refine_type_ == RefineType::FLOAT16_QUANT && data_type == DataFormatEnum::bf16)) {
                throw std::runtime_error(
                    "Type fp16 can't use BFLOAT16_QUANT to refine or bf16 can't use BFLOAT16_QUANT to refine.");
            }
        } else {
            quant_data_ = nullptr;
        }
    }
    virtual ~DataViewIndexBase(){};

    virtual void
    Train(idx_t n, const void* __restrict x, bool use_knowhere_build_pool) = 0;

    virtual void
    Add(idx_t n, const void* __restrict x, const float* __restrict norms_, bool use_knowhere_build_pool) = 0;

    virtual void
    Search(const idx_t n, const void* __restrict x, const idx_t k, float* __restrict distances,
           idx_t* __restrict labels, const BitsetView& bitset, milvus::OpContext* op_context,
           const bool use_quant) const = 0;

    /** Knn Search on set of vectors
     *
     * @param n             nb of vectors to query
     * @param x             query vectors, size nx * d
     * @param ids_num_lims  prefix sum of different selected ids rows , size n + 1
     * @param ids           selected ids for each queries
     * @param k             topk
     * @param out_dist      result ids, size nx * topk
     * @param out_ids       result distances, size nx * topk
     */
    virtual void
    SearchWithIds(const idx_t n, const void* __restrict x, const idx_t* __restrict ids_num_lims,
                  const idx_t* __restrict ids, const idx_t k, float* __restrict out_dist, idx_t* __restrict out_ids,
                  const bool use_quant) const = 0;

    /**
     * @brief Calculate distances between query vectors and reference vectors by IDs
     *
     * @param num_queries  number of query vectors
     * @param queries      query vectors, size num_queries * d
     * @param num_ids      total number of selected ids
     * @param ids          selected ids for each query
     * @param out_dist     result distances, size num_queries * num_ids
     */
    virtual void
    CalcDistByIDs(const idx_t num_queries, const void* __restrict queries, const idx_t num_ids,
                  const idx_t* __restrict ids, float* __restrict out_dist, const bool use_quant) const = 0;

    virtual RangeSearchResult
    RangeSearch(const idx_t n, const void* __restrict x, const float radius, const float range_filter,
                const BitsetView& bitset, milvus::OpContext* op_context, const bool use_quant) const = 0;

    /** Range Search on set of vectors
     *
     * @param n             nb of vectors to query
     * @param x             query vectors, size nx * d
     * @param ids_num_lims  prefix sum of different selected ids rows , size n + 1
     * @param ids           selected ids for each queries
     * @param radius
     * @param range_filter
     */
    virtual RangeSearchResult
    RangeSearchWithIds(const idx_t n, const void* __restrict x, const idx_t* __restrict ids_num_lims,
                       const idx_t* __restrict ids, const float radius, const float range_filter,
                       const bool use_quant) const = 0;

    virtual void
    ComputeDistanceSubset(const void* __restrict x, const idx_t sub_y_n, float* x_y_distances,
                          const idx_t* __restrict x_y_labels, const bool use_quant) const = 0;

    auto
    Dim() const {
        return d_;
    }
    bool
    IsCosine() const {
        return is_cosine_;
    }
    MetricType
    Metric() const {
        return metric_type_;
    }
    DataFormatEnum
    DataFormat() const {
        return data_type_;
    }
    ViewDataOp
    GetViewData() const {
        return view_data_;
    }
    idx_t
    Count() const {
        return ntotal_.load();
    }
    std::shared_ptr<QuantRefine>
    GetQuantData() const {
        return quant_data_;
    }
    std::optional<size_t>
    GetQueryCodeSize() const {
        return code_size_;
    }

 protected:
    int d_;
    DataFormatEnum data_type_;
    MetricType metric_type_;
    ViewDataOp view_data_;
    bool is_cosine_;
    int code_size_;
    std::atomic<idx_t> ntotal_ = 0;
    RefineType refine_type_;
    std::shared_ptr<QuantRefine> quant_data_ = nullptr;
    std::optional<int> build_thread_num_ = std::nullopt;
};

class DataViewIndexFlat : public DataViewIndexBase {
 public:
    DataViewIndexFlat(idx_t d, DataFormatEnum data_type, MetricType metric_type, ViewDataOp view, bool is_cosine,
                      RefineType refine_type, std::optional<int> build_thread_num = std::nullopt)
        : DataViewIndexBase(d, data_type, metric_type, view, is_cosine, refine_type, build_thread_num) {
        this->ntotal_.store(0);
    }
    void
    Train(idx_t n, const void* x, bool use_knowhere_build_pool) override {
        if (quant_data_ != nullptr) {
            auto build_pool_wrapper =
                std::make_shared<ThreadPoolWrapper>(ThreadPool::GetGlobalBuildThreadPool(), use_knowhere_build_pool);
            auto task = build_pool_wrapper
                            ->push([&] {
                                std::unique_ptr<ThreadPool::ScopedBuildOmpSetter> setter;
                                if (build_thread_num_.has_value()) {
                                    setter =
                                        std::make_unique<ThreadPool::ScopedBuildOmpSetter>(build_thread_num_.value());
                                } else {
                                    setter = std::make_unique<ThreadPool::ScopedBuildOmpSetter>();
                                }
                                quant_data_->Train(x, n);
                            })
                            .getTry();
            if (task.hasException()) {
                std::runtime_error(std::string("faiss internal error," + task.exception().what()));
            }
        }
        return;
    }

    void
    Add(idx_t n, const void* x, const float* __restrict in_norms, bool use_knowhere_build_pool) override {
        if (quant_data_ != nullptr) {
            auto build_pool_wrapper =
                std::make_shared<ThreadPoolWrapper>(ThreadPool::GetGlobalBuildThreadPool(), use_knowhere_build_pool);
            auto task = build_pool_wrapper
                            ->push([&] {
                                std::unique_ptr<ThreadPool::ScopedBuildOmpSetter> setter;
                                if (build_thread_num_.has_value()) {
                                    setter =
                                        std::make_unique<ThreadPool::ScopedBuildOmpSetter>(build_thread_num_.value());
                                } else {
                                    setter = std::make_unique<ThreadPool::ScopedBuildOmpSetter>();
                                }
                                std::vector<idx_t> ids(n);
                                std::iota(ids.begin(), ids.end(), ntotal_.load());
                                quant_data_->Add(x, ids.data(), n);
                            })
                            .getTry();
            if (task.hasException()) {
                std::runtime_error(std::string("faiss internal error," + task.exception().what()));
            }
        }
        if (is_cosine_) {
            if (in_norms == nullptr) {
                std::vector<float> l2_norms;
                if (data_type_ == DataFormatEnum::fp32) {
                    l2_norms = GetL2Norms<fp32>((const fp32*)x, d_, n);
                } else if (data_type_ == DataFormatEnum::fp16) {
                    l2_norms = GetL2Norms<fp16>((const fp16*)x, d_, n);
                } else {
                    l2_norms = GetL2Norms<bf16>((const bf16*)x, d_, n);
                }
                std::unique_lock lock(norms_mutex_);
                norms_.insert(norms_.end(), l2_norms.begin(), l2_norms.end());
            } else {
                std::unique_lock lock(norms_mutex_);
                norms_.insert(norms_.end(), in_norms, in_norms + n);
            }
        }
        ntotal_.fetch_add(n);
    }

    void
    Search(const idx_t n, const void* __restrict x, const idx_t k, float* __restrict distances,
           idx_t* __restrict labels, const BitsetView& bitset, milvus::OpContext* op_context,
           const bool use_quant) const override;

    void
    SearchWithIds(const idx_t n, const void* __restrict x, const idx_t* __restrict ids_num_lims,
                  const idx_t* __restrict ids, const idx_t k, float* __restrict out_dist, idx_t* __restrict out_ids,
                  const bool use_quant) const override;

    void
    CalcDistByIDs(const idx_t num_queries, const void* __restrict queries, const idx_t num_ids,
                  const idx_t* __restrict ids, float* __restrict out_dist, const bool use_quant) const override;

    RangeSearchResult
    RangeSearch(const idx_t n, const void* __restrict x, const float radius, const float range_filter,
                const BitsetView& bitset, milvus::OpContext* op_context, const bool use_quant) const override;

    RangeSearchResult
    RangeSearchWithIds(const idx_t n, const void* __restrict x, const idx_t* __restrict ids_num_lims,
                       const idx_t* __restrict ids, const float radius, const float range_filter,
                       const bool use_quant) const override;

    void
    ComputeDistanceSubset(const void* __restrict x, const idx_t sub_y_n, float* __restrict x_y_distances,
                          const idx_t* __restrict x_y_labels, const bool use_quant) const override;

    float
    GetDataNorm(idx_t id) const {
        assert(id < ntotal_);
        idx_t current_norms_size = 0;
        {
            std::shared_lock lock(norms_mutex_);
            current_norms_size = norms_.size();
        }
        if (current_norms_size < id) {  // maybe cosine is false, get norm in place
            auto data = view_data_(id);
            if (data_type_ == DataFormatEnum::fp32) {
                return GetL2Norm<fp32>((const fp32*)data, d_);
            } else if (data_type_ == DataFormatEnum::fp16) {
                return GetL2Norm<fp16>((const fp16*)data, d_);
            } else {
                return GetL2Norm<bf16>((const bf16*)data, d_);
            }
        } else {
            std::shared_lock lock(norms_mutex_);
            return norms_[id];
        }
    }

 protected:
    template <class SingleResultHandler, class SelectorHelper>
    void
    exhaustive_search_in_one_query_impl(const std::unique_ptr<faiss::DistanceComputer>& computer, size_t ny,
                                        SingleResultHandler& resi, const SelectorHelper& selector) const;

 protected:
    std::vector<float> norms_;  // vector norms will be populated if is_cosine_ == true
    mutable std::shared_mutex norms_mutex_;
};

template <class SingleResultHandler, class SelectorHelper>
void
DataViewIndexFlat::exhaustive_search_in_one_query_impl(const std::unique_ptr<faiss::DistanceComputer>& computer,
                                                       size_t ny, SingleResultHandler& resi,
                                                       const SelectorHelper& selector) const {
    auto filter = [&selector](const size_t j) { return selector.is_member(j); };
    if (is_cosine_) {
        std::shared_lock lock(norms_mutex_);
        auto apply = [&](const float in_dis, const idx_t j) {
            float dis = in_dis;
            dis = dis / (norms_[j]);
            resi.add_result(dis, j);
        };
        faiss::distance_compute_if(ny, computer.get(), filter, apply);
    } else {
        auto apply = [&resi](const float dis, const idx_t j) { resi.add_result(dis, j); };
        faiss::distance_compute_if(ny, computer.get(), filter, apply);
    }
}

void
DataViewIndexFlat::Search(const idx_t n, const void* __restrict x, const idx_t k, float* __restrict distances,
                          idx_t* __restrict labels, const BitsetView& bitset, milvus::OpContext* op_context,
                          const bool use_quant) const {
    // todo: need more test to check
    const auto& search_pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futs;
    futs.reserve(n);
    if (k < faiss::distance_compute_min_k_reservoir) {
        if (metric_type_ == metric::L2) {
            faiss::HeapBlockResultHandler<CMAX> res(n, distances, labels, k);
            for (auto i = 0; i < n; i++) {
                futs.emplace_back(search_pool->push([&] {
                    knowhere::checkCancellation(op_context);
                    ThreadPool::ScopedSearchOmpSetter setter(1);
                    faiss::HeapBlockResultHandler<CMAX>::SingleResultHandler resi(res);
                    auto computer = SelectDataViewComputer(view_data_, data_type_, metric_type_, d_, is_cosine_,
                                                           use_quant ? quant_data_ : nullptr);
                    computer->set_query((const float*)((const char*)x + code_size_ * i));
                    resi.begin(i);
                    if (bitset.empty()) {
                        exhaustive_search_in_one_query_impl(computer, n, resi, faiss::IDSelectorAll());
                    } else {
                        exhaustive_search_in_one_query_impl(computer, n, resi, BitsetViewIDSelector(bitset));
                    }
                    resi.end();
                }));
            }
            WaitAllSuccess(futs);
        } else {
            faiss::HeapBlockResultHandler<CMIN> res(n, distances, labels, k);
            for (auto i = 0; i < n; i++) {
                futs.emplace_back(search_pool->push([&] {
                    knowhere::checkCancellation(op_context);
                    ThreadPool::ScopedSearchOmpSetter setter(1);
                    faiss::HeapBlockResultHandler<CMIN>::SingleResultHandler resi(res);
                    auto computer = SelectDataViewComputer(view_data_, data_type_, metric_type_, d_, is_cosine_,
                                                           use_quant ? quant_data_ : nullptr);
                    computer->set_query((const float*)((const char*)x + code_size_ * i));
                    resi.begin(i);
                    if (bitset.empty()) {
                        exhaustive_search_in_one_query_impl(computer, n, resi, faiss::IDSelectorAll());
                    } else {
                        exhaustive_search_in_one_query_impl(computer, n, resi, BitsetViewIDSelector(bitset));
                    }
                    resi.end();
                }));
            }
            WaitAllSuccess(futs);
        }
    } else {
        if (metric_type_ == metric::L2) {
            faiss::ReservoirBlockResultHandler<CMAX> res(n, distances, labels, k);

            for (auto i = 0; i < n; i++) {
                futs.emplace_back(search_pool->push([&] {
                    ThreadPool::ScopedSearchOmpSetter setter(1);
                    faiss::ReservoirBlockResultHandler<CMAX>::SingleResultHandler resi(res);
                    auto computer = SelectDataViewComputer(view_data_, data_type_, metric_type_, d_, is_cosine_,
                                                           use_quant ? quant_data_ : nullptr);
                    computer->set_query((const float*)((const char*)x + code_size_ * i));
                    resi.begin(i);
                    if (bitset.empty()) {
                        exhaustive_search_in_one_query_impl(computer, n, resi, faiss::IDSelectorAll());
                    } else {
                        exhaustive_search_in_one_query_impl(computer, n, resi, BitsetViewIDSelector(bitset));
                    }
                    resi.end();
                }));
            }
            WaitAllSuccess(futs);
        } else {
            faiss::ReservoirBlockResultHandler<CMIN> res(n, distances, labels, k);
            for (auto i = 0; i < n; i++) {
                futs.emplace_back(search_pool->push([&] {
                    ThreadPool::ScopedSearchOmpSetter setter(1);
                    faiss::ReservoirBlockResultHandler<CMIN>::SingleResultHandler resi(res);
                    auto computer = SelectDataViewComputer(view_data_, data_type_, metric_type_, d_, is_cosine_,
                                                           use_quant ? quant_data_ : nullptr);
                    computer->set_query((const float*)((const char*)x + code_size_ * i));
                    resi.begin(i);
                    if (bitset.empty()) {
                        exhaustive_search_in_one_query_impl(computer, n, resi, faiss::IDSelectorAll());
                    } else {
                        exhaustive_search_in_one_query_impl(computer, n, resi, BitsetViewIDSelector(bitset));
                    }
                    resi.end();
                }));
            }
            WaitAllSuccess(futs);
        }
    }
}

void
DataViewIndexFlat::SearchWithIds(const idx_t n, const void* __restrict x, const idx_t* __restrict ids_num_lims,
                                 const idx_t* __restrict ids, const idx_t k, float* __restrict out_dist,
                                 idx_t* __restrict out_ids, const bool use_quant) const {
    const auto& search_pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futs;
    futs.reserve(n);
    for (auto i = 0; i < n; i++) {
        futs.emplace_back(search_pool->push([&, i = i] {
            ThreadPool::ScopedSearchOmpSetter setter(1);
            auto base_ids = ids + ids_num_lims[i];
            auto base_n = ids_num_lims[i + 1] - ids_num_lims[i];
            auto base_dist = std::unique_ptr<float[]>(new float[base_n]);
            if (metric_type_ == metric::L2) {
                std::fill(base_dist.get(), base_dist.get() + base_n, CMAX::neutral());
            } else {
                std::fill(base_dist.get(), base_dist.get() + base_n, CMIN::neutral());
            }
            auto x_i = (const char*)x + code_size_ * i;

            assert(base_n >= k);
            ComputeDistanceSubset(x_i, base_n, base_dist.get(), base_ids, use_quant);
            if (is_cosine_) {
                std::shared_lock lock(norms_mutex_);
                for (auto j = 0; j < base_n; j++) {
                    if (base_ids[j] != -1) {
                        base_dist[j] /= norms_[base_ids[j]];
                    }
                }
            }
            if (metric_type_ == metric::L2) {
                faiss::reorder_2_heaps<CMAX>(1, k, out_ids + i * k, out_dist + i * k, base_n, base_ids,
                                             base_dist.get());
            } else {
                faiss::reorder_2_heaps<CMIN>(1, k, out_ids + i * k, out_dist + i * k, base_n, base_ids,
                                             base_dist.get());
            }
        }));
    }
    WaitAllSuccess(futs);
    return;
}

void
DataViewIndexFlat::CalcDistByIDs(const idx_t num_queries, const void* __restrict queries, const idx_t num_ids,
                                 const idx_t* __restrict ids, float* __restrict out_dist, const bool use_quant) const {
    const auto& search_pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futs;
    futs.reserve(num_queries);
    for (auto i = 0; i < num_queries; i++) {
        futs.emplace_back(search_pool->push([&, idx = i] {
            ThreadPool::ScopedSearchOmpSetter setter(1);
            auto query = (const char*)queries + code_size_ * idx;
            ComputeDistanceSubset(query, num_ids, out_dist + idx * num_ids, ids, use_quant);
            if (is_cosine_) {
                std::shared_lock lock(norms_mutex_);
                for (auto j = 0; j < num_ids; j++) {
                    out_dist[idx * num_ids + j] /= norms_[ids[j]];
                }
            }
        }));
    }
    WaitAllSuccess(futs);
    return;
}

RangeSearchResult
DataViewIndexFlat::RangeSearch(const idx_t n, const void* __restrict x, const float radius, const float range_filter,
                               const BitsetView& bitset, milvus::OpContext* op_context, const bool use_quant) const {
    // todo: need more test to check
    std::vector<std::vector<float>> result_dist_array(n);
    std::vector<std::vector<idx_t>> result_id_array(n);

    std::shared_ptr<float[]> base_norms = nullptr;
    if (is_cosine_) {
        std::shared_lock lock(norms_mutex_);
        base_norms = std::shared_ptr<float[]>(new float[norms_.size()]);
        std::memcpy(base_norms.get(), norms_.data(), sizeof(float) * norms_.size());
    }
    auto is_ip = metric_type_ == metric::IP;

    const auto& search_pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futs;
    futs.reserve(n);
    if (metric_type_ == metric::L2) {
        for (auto i = 0; i < n; i++) {
            futs.emplace_back(search_pool->push([&, i = i] {
                knowhere::checkCancellation(op_context);
                ThreadPool::ScopedSearchOmpSetter setter(1);
                auto computer = SelectDataViewComputer(view_data_, data_type_, metric_type_, d_, is_cosine_,
                                                       use_quant ? quant_data_ : nullptr);
                faiss::RangeSearchResult res(1);
                faiss::RangeSearchBlockResultHandler<CMAX> resh(&res, radius);
                faiss::RangeSearchBlockResultHandler<CMAX>::SingleResultHandler reshi(resh);
                computer->set_query(((const float*)x + code_size_ * i));
                reshi.begin(i);
                if (bitset.empty()) {
                    exhaustive_search_in_one_query_impl(computer, n, reshi, faiss::IDSelectorAll());
                } else {
                    exhaustive_search_in_one_query_impl(computer, n, reshi, BitsetViewIDSelector(bitset));
                }
                reshi.end();
                auto elem_cnt = res.lims[1];
                result_dist_array[i].resize(elem_cnt);
                result_id_array[i].resize(elem_cnt);
                for (size_t j = 0; j < elem_cnt; j++) {
                    result_dist_array[i][j] = res.distances[j];
                    result_id_array[i][j] = res.labels[j];
                }
                if (range_filter != defaultRangeFilter) {
                    FilterRangeSearchResultForOneNq(result_dist_array[i], result_id_array[i], is_ip, radius,
                                                    range_filter);
                }
            }));
        }
        WaitAllSuccess(futs);
    } else {
        for (auto i = 0; i < n; i++) {
            futs.emplace_back(search_pool->push([&, i = i] {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                auto computer = SelectDataViewComputer(view_data_, data_type_, metric_type_, d_, is_cosine_,
                                                       use_quant ? quant_data_ : nullptr);
                faiss::RangeSearchResult res(1);
                faiss::RangeSearchBlockResultHandler<CMIN> resh(&res, radius);
                faiss::RangeSearchBlockResultHandler<CMIN>::SingleResultHandler reshi(resh);
                computer->set_query(((const float*)x + code_size_ * i));
                reshi.begin(i);
                if (bitset.empty()) {
                    exhaustive_search_in_one_query_impl(computer, n, reshi, faiss::IDSelectorAll());
                } else {
                    exhaustive_search_in_one_query_impl(computer, n, reshi, BitsetViewIDSelector(bitset));
                }
                reshi.end();
                auto elem_cnt = res.lims[1];
                result_dist_array[i].resize(elem_cnt);
                result_id_array[i].resize(elem_cnt);
                for (size_t j = 0; j < elem_cnt; j++) {
                    result_dist_array[i][j] = res.distances[j];
                    result_id_array[i][j] = res.labels[j];
                }
                if (range_filter != defaultRangeFilter) {
                    FilterRangeSearchResultForOneNq(result_dist_array[i], result_id_array[i], is_ip, radius,
                                                    range_filter);
                }
            }));
        }
        WaitAllSuccess(futs);
    }
    return GetRangeSearchResult(result_dist_array, result_id_array, is_ip, n, radius, range_filter);
}

RangeSearchResult
DataViewIndexFlat::RangeSearchWithIds(const idx_t n, const void* __restrict x, const idx_t* __restrict ids_num_lims,
                                      const idx_t* __restrict ids, const float radius, const float range_filter,
                                      const bool use_quant) const {
    std::vector<std::vector<float>> result_dist_array(n);
    std::vector<std::vector<idx_t>> result_id_array(n);
    auto is_ip = metric_type_ == metric::IP;
    const auto& search_pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futs;
    futs.reserve(n);
    for (auto i = 0; i < n; i++) {
        futs.emplace_back(search_pool->push([&, i = i] {
            ThreadPool::ScopedSearchOmpSetter setter(1);
            auto base_ids = ids + ids_num_lims[i];
            auto base_n = ids_num_lims[i + 1] - ids_num_lims[i];
            auto base_dist = std::unique_ptr<float[]>(new float[base_n]);
            auto x_i = (const char*)x + code_size_ * i;
            ComputeDistanceSubset((const void*)x_i, base_n, base_dist.get(), base_ids, use_quant);
            if (is_cosine_) {
                std::shared_lock lock(norms_mutex_);
                for (auto j = 0; j < base_n; j++) {
                    base_dist[j] = base_dist[j] / norms_[base_ids[j]];
                }
            }
            for (auto j = 0; j < base_n; j++) {
                if (!is_ip) {
                    if (base_dist[j] < radius) {
                        result_dist_array[i].emplace_back(base_dist[j]);
                        result_id_array[i].emplace_back(base_ids[j]);
                    }
                } else {
                    if (base_dist[j] > radius) {
                        result_dist_array[i].emplace_back(base_dist[j]);
                        result_id_array[i].emplace_back(base_ids[j]);
                    }
                }
            }
            if (range_filter != defaultRangeFilter) {
                FilterRangeSearchResultForOneNq(result_dist_array[i], result_id_array[i], is_ip, radius, range_filter);
            }
        }));
    }
    WaitAllSuccess(futs);
    return GetRangeSearchResult(result_dist_array, result_id_array, is_ip, n, radius, range_filter);
}

void
DataViewIndexFlat::ComputeDistanceSubset(const void* __restrict x, const idx_t sub_y_n, float* x_y_distances,
                                         const idx_t* __restrict x_y_labels, const bool use_quant) const {
    auto computer =
        SelectDataViewComputer(view_data_, data_type_, metric_type_, d_, is_cosine_, use_quant ? quant_data_ : nullptr);

    computer->set_query((const float*)(x));
    const idx_t* __restrict idsj = x_y_labels;
    float* __restrict disj = x_y_distances;

    auto filter = [=](const size_t i) { return (idsj[i] >= 0); };
    auto apply = [=](const float dis, const size_t i) { disj[i] = dis; };
    distance_compute_by_idx_if(idsj, sub_y_n, computer.get(), filter, apply);
}
}  // namespace knowhere
