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

#include <sys/mman.h>

#include <boost/intrusive/pack_options.hpp>
#include <exception>

#include "index/sparse/inverted/flatten_inverted_index.h"
#include "index/sparse/inverted/growable_inverted_index.h"
#include "index/sparse/inverted/inverted_index.h"
#include "index/sparse/sparse_inverted_index_config.h"
#include "io/file_io.h"
#include "io/memory_io.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere {

// Inverted Index impl for sparse vectors.
//
// Not overriding RangeSearch, will use the default implementation in IndexNode.
//
// Thread safety: not thread safe.
template <typename T, bool use_wand>
class SparseInvertedIndexNode : public IndexNode {
    static_assert(std::is_same_v<T, fp32>, "SparseInvertedIndexNode only support float");

 public:
    explicit SparseInvertedIndexNode(const int32_t& /*version*/, const Object& /*object*/)
        : search_pool_(ThreadPool::GetGlobalSearchThreadPool()), build_pool_(ThreadPool::GetGlobalBuildThreadPool()) {
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> config) override {
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        // metric type validation
        if (!IsMetricType(cfg.metric_type.value(), metric::IP) &&
            !IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            LOG_KNOWHERE_ERROR_ << Type() << " only support metric_type IP or BM25";
            return Status::invalid_metric_type;
        }

        if (IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            if (!cfg.bm25_k1.has_value() || !cfg.bm25_b.has_value() || !cfg.bm25_avgdl.has_value()) {
                return Status::invalid_args;
            }
        }

        // create index
        auto index_or = IsMetricType(cfg.metric_type.value(), metric::IP) ? CreateIndex<T, fp16>(cfg, true)
                                                                          : CreateIndex<T, uint16_t>(cfg, true);
        if (!index_or.has_value()) {
            return index_or.error();
        }

        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, old index will be deleted";
        }
        index_ = std::move(index_or.value());

        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> config) override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not add data to uninitialized " << Type() << " index";
            return Status::empty_index;
        }

        auto tryObj = build_pool_
                          ->push([&] {
                              return index_->add(static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor()),
                                                 dataset->GetRows(), dataset->GetDim());
                          })
                          .getTry();
        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "Failed to add data to index " << Type() << ": " << tryObj.exception().what();
            return Status::sparse_inner_error;
        }

        return tryObj.value();
    }

    [[nodiscard]] expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not search uninitialized " << Type() << " index";
            return expected<DataSetPtr>::Err(Status::empty_index, "index is not initialized");
        }

        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        sparse::InvertedIndexSearchParams search_params = {
            .approx =
                {
                    .drop_ratio_search = cfg.drop_ratio_search.value_or(0.0f),
                    .dim_max_score_ratio = cfg.dim_max_score_ratio.value_or(1.05f),
                },
        };

        if (cfg.search_algo.value() == "INHERIT") {
            search_params.algo = index_->build_algo();
        } else if (cfg.search_algo.value() == "DAAT_MAXSCORE") {
            search_params.algo = sparse::InvertedIndexAlgo::DAAT_MAXSCORE;
        } else if (cfg.search_algo.value() == "DAAT_WAND") {
            search_params.algo = sparse::InvertedIndexAlgo::DAAT_WAND;
        } else if (cfg.search_algo.value() == "TAAT_NAIVE") {
            search_params.algo = sparse::InvertedIndexAlgo::TAAT_NAIVE;
        } else {
            return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported search algorithm");
        }

        if (IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            if (!cfg.bm25_k1.has_value() || !cfg.bm25_b.has_value() || !cfg.bm25_avgdl.has_value()) {
                return expected<DataSetPtr>::Err(Status::invalid_args,
                                                 "BM25 parameters k1, b, and avgdl must be set when searching");
            }
            search_params.metric_type = sparse::SparseMetricType::METRIC_BM25;
            search_params.metric_params.bm25 = {
                .k1 = cfg.bm25_k1.value(), .b = cfg.bm25_b.value(), .avgdl = std::max(cfg.bm25_avgdl.value(), 1.0f)};
        } else if (IsMetricType(cfg.metric_type.value(), metric::IP)) {
            search_params.metric_type = sparse::SparseMetricType::METRIC_IP;
        } else {
            return expected<DataSetPtr>::Err(Status::invalid_metric_type, "Unsupported metric type");
        }

        if (auto status = index_->valid_search_check(search_params); status != Status::success) {
            return expected<DataSetPtr>::Err(status, "Invalid search params");
        }

        auto queries = static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor());
        auto nq = dataset->GetRows();
        auto k = cfg.k.value();
        auto p_id = std::make_unique<sparse::label_t[]>(nq * k);
        auto p_dist = std::make_unique<float[]>(nq * k);

        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (int64_t idx = 0; idx < nq; ++idx) {
            futs.emplace_back(search_pool_->push([&, idx = idx, p_id = p_id.get(), p_dist = p_dist.get()]() {
                index_->search(queries[idx], k, p_dist + idx * k, p_id + idx * k, bitset, search_params);
            }));
        }
        WaitAllSuccess(futs);

        return GenResultDataSet(nq, k, p_id.release(), p_dist.release());
    }

    // TODO: for now inverted index and wand use the same impl for AnnIterator.
    [[nodiscard]] expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset,
                bool use_knowhere_search_pool) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "creating iterator on empty index";
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::empty_index, "index not loaded");
        }

        auto nq = dataset->GetRows();

        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        sparse::InvertedIndexSearchParams search_params = {
            .approx =
                {
                    .drop_ratio_search = cfg.drop_ratio_search.value_or(0.0f),
                    .dim_max_score_ratio = cfg.dim_max_score_ratio.value_or(1.05f),
                },
        };

        if (cfg.search_algo.value() == "INHERIT") {
            search_params.algo = index_->build_algo();
        } else if (cfg.search_algo.value() == "DAAT_MAXSCORE") {
            search_params.algo = sparse::InvertedIndexAlgo::DAAT_MAXSCORE;
        } else if (cfg.search_algo.value() == "DAAT_WAND") {
            search_params.algo = sparse::InvertedIndexAlgo::DAAT_WAND;
        } else if (cfg.search_algo.value() == "TAAT_NAIVE") {
            search_params.algo = sparse::InvertedIndexAlgo::TAAT_NAIVE;
        } else {
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::invalid_args,
                                                                      "Unsupported search algorithm");
        }

        if (IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            if (!cfg.bm25_k1.has_value() || !cfg.bm25_b.has_value() || !cfg.bm25_avgdl.has_value()) {
                return expected<std::vector<IndexNode::IteratorPtr>>::Err(
                    Status::invalid_args, "BM25 parameters k1, b, and avgdl must be set when searching");
            }
            search_params.metric_type = sparse::SparseMetricType::METRIC_BM25;
            search_params.metric_params.bm25 = {
                .k1 = cfg.bm25_k1.value(), .b = cfg.bm25_b.value(), .avgdl = std::max(cfg.bm25_avgdl.value(), 1.0f)};
        } else if (IsMetricType(cfg.metric_type.value(), metric::IP)) {
            search_params.metric_type = sparse::SparseMetricType::METRIC_IP;
        } else {
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::invalid_metric_type,
                                                                      "Unsupported metric type");
        }

        if (auto status = index_->valid_search_check(search_params); status != Status::success) {
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(status, "Invalid AnnIterator params");
        }

        auto vec = std::vector<std::shared_ptr<IndexNode::iterator>>(nq, nullptr);
        try {
            for (int i = 0; i < nq; ++i) {
                // Heavy computations with `compute_dist_func` will be deferred until the first call to
                // 'Iterator->Next()'.
                auto compute_dist_func = [=]() -> std::vector<DistId> {
                    auto queries = static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor());
                    std::vector<float> distances = index_->get_all_distances(queries[i], bitset, search_params);
                    std::vector<DistId> distances_ids;
                    // 30% is a ratio guesstimate of non-zero distances: probability of 2 random sparse splade
                    // vectors(100 non zero dims out of 30000 total dims) sharing at least 1 common non-zero
                    // dimension.
                    distances_ids.reserve(distances.size() * 0.3);
                    for (size_t i = 0; i < distances.size(); i++) {
                        if (distances[i] != 0) {
                            distances_ids.emplace_back((int64_t)i, distances[i]);
                        }
                    }
                    return distances_ids;
                };

                auto it =
                    std::make_shared<PrecomputedDistanceIterator>(compute_dist_func, true, use_knowhere_search_pool);
                vec[i] = it;
            }
        } catch (const std::exception& e) {
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::sparse_inner_error, e.what());
        }

        return vec;
    }

    [[nodiscard]] expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
    }

    [[nodiscard]] bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    [[nodiscard]] expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not supported for current index type");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not serialize empty " << Type();
            return Status::empty_index;
        }

        MemoryIOWriter writer;
        RETURN_IF_ERROR(index_->convert_to_raw_data(writer));
        std::shared_ptr<uint8_t[]> data(writer.data());
        binset.Append(Type(), data, writer.tellg());

        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        auto cfg = static_cast<const knowhere::SparseInvertedIndexConfig&>(*config);

        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid BinarySet";
            return Status::invalid_binary_set;
        }

        MemoryIOReader reader(binary->data.get(), binary->size);
        auto index_or = IsMetricType(cfg.metric_type.value(), metric::IP) ? CreateIndex<T, fp16>(cfg, false)
                                                                          : CreateIndex<T, uint16_t>(cfg, false);
        if (!index_or.has_value()) {
            return index_or.error();
        }

        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, old index will be deleted";
        }
        index_ = std::move(index_or.value());

        return index_->build_from_raw_data(reader, false, "");
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, old index will be deleted";
        }

        auto cfg = static_cast<const knowhere::SparseInvertedIndexConfig&>(*config);

        auto index_or = IsMetricType(cfg.metric_type.value(), metric::IP) ? CreateIndex<T, fp16>(cfg, false)
                                                                          : CreateIndex<T, uint16_t>(cfg, false);
        if (!index_or.has_value()) {
            return index_or.error();
        }

        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, old index will be deleted";
        }
        index_ = std::move(index_or.value());

        auto reader = knowhere::FileReader(filename);
        size_t map_size = reader.size();
        int map_flags = MAP_SHARED;
#ifdef MAP_POPULATE
        if (cfg.enable_mmap_pop.has_value() && cfg.enable_mmap_pop.value()) {
            map_flags |= MAP_POPULATE;
        }
#endif
        void* mapped_memory = mmap(nullptr, map_size, PROT_READ, map_flags, reader.descriptor(), 0);
        if (mapped_memory == MAP_FAILED) {
            LOG_KNOWHERE_ERROR_ << "Failed to mmap file " << filename << ": " << strerror(errno);
            return Status::disk_file_error;
        }

        auto cleanup_mmap = [map_size, filename](void* map_addr) {
            if (munmap(map_addr, map_size) != 0) {
                LOG_KNOWHERE_ERROR_ << "Failed to munmap file " << filename << ": " << strerror(errno);
            }
        };
        std::unique_ptr<void, decltype(cleanup_mmap)> mmap_guard(mapped_memory, cleanup_mmap);

        MemoryIOReader map_reader(reinterpret_cast<uint8_t*>(mapped_memory), map_size);
        auto supplement_target_filename = filename + ".knowhere_sparse_index_supplement";
        return index_->build_from_raw_data(map_reader, true, supplement_target_filename);
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<SparseInvertedIndexConfig>();
    }

    [[nodiscard]] std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    // note that the Dim of a sparse vector index may change as new vectors are added
    [[nodiscard]] int64_t
    Dim() const override {
        return index_ ? index_->nr_cols() : 0;
    }

    [[nodiscard]] int64_t
    Size() const override {
        return index_ ? index_->size() : 0;
    }

    [[nodiscard]] int64_t
    Count() const override {
        return index_ ? index_->nr_rows() : 0;
    }

    [[nodiscard]] std::string
    Type() const override {
        return use_wand ? knowhere::IndexEnum::INDEX_SPARSE_WAND : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
    }

 private:
    template <typename DType, typename QType>
    expected<std::unique_ptr<sparse::InvertedIndex<T>>>
    CreateIndex(const SparseInvertedIndexConfig& cfg, bool mutable_index) const {
        std::unique_ptr<sparse::InvertedIndex<T>> index;

        if (mutable_index) {
            index = std::make_unique<sparse::GrowableInvertedIndex<DType, QType>>();
        } else {
            index = std::make_unique<sparse::FlattenInvertedIndex<DType, QType>>();
        }

        // set metadata flags of the index, which will be used in the data add phase
        sparse::InvertedIndexMetaData::MetaDataFlags flags = sparse::InvertedIndexMetaData::FLAG_NONE;

        if (IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            index->set_metric(
                sparse::SparseMetricType::METRIC_BM25,
                sparse::SparseMetricParams{.bm25 = {.k1 = cfg.bm25_k1.value(),
                                                    .b = cfg.bm25_b.value(),
                                                    // avgdl is used as a denominator in BM25 score computation,
                                                    // so it should be at least 1.0 to avoid division by zero.
                                                    .avgdl = std::max(cfg.bm25_avgdl.value(), 1.0f)}});
            flags |= sparse::InvertedIndexMetaData::FLAG_HAS_ROW_SUMS;
        } else if (IsMetricType(cfg.metric_type.value(), metric::IP)) {
            index->set_metric(sparse::SparseMetricType::METRIC_IP, sparse::SparseMetricParams{});
        } else {
            return expected<std::unique_ptr<sparse::InvertedIndex<T>>>::Err(Status::invalid_metric_type,
                                                                            "Unsupported metric type");
        }

        if (cfg.inverted_index_algo.has_value() &&
            (cfg.inverted_index_algo.value() == "DAAT_MAXSCORE" || cfg.inverted_index_algo.value() == "DAAT_WAND")) {
            flags |= sparse::InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM;
        }

        index->set_build_algo(cfg.inverted_index_algo.value());

        index->set_metadata_flags(flags);

        return index;
    }

    std::unique_ptr<sparse::InvertedIndex<T>> index_;
    std::shared_ptr<ThreadPool> search_pool_;
    std::shared_ptr<ThreadPool> build_pool_;
};  // class SparseInvertedIndexNode

// Concurrent version of SparseInvertedIndexNode
//
// Thread safety: only the overridden methods are allowed to be called concurrently.
template <typename T, bool use_wand>
class SparseInvertedIndexNodeCC : public SparseInvertedIndexNode<T, use_wand> {
 public:
    explicit SparseInvertedIndexNodeCC(const int32_t& version, const Object& object)
        : SparseInvertedIndexNode<T, use_wand>(version, object) {
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> config) override {
        std::unique_lock<std::mutex> lock(mutex_);
        uint64_t task_id = next_task_id_++;
        add_tasks_.push(task_id);

        // add task is allowed to run only after all search tasks that come before it have finished.
        cv_.wait(lock, [this, task_id]() { return current_task_id_ == task_id && active_readers_ == 0; });

        auto res = SparseInvertedIndexNode<T, use_wand>::Add(dataset, config);

        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);
        if (IsMetricType(cfg.metric_type.value(), metric::IP)) {
            // insert dataset to raw data if metric type is IP
            auto data = static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor());
            auto rows = dataset->GetRows();
            raw_data_.insert(raw_data_.end(), data, data + rows);
        }

        add_tasks_.pop();
        current_task_id_++;
        lock.unlock();
        cv_.notify_all();
        return res;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Search(dataset, std::move(cfg), bitset);
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::AnnIterator(dataset, std::move(cfg), bitset,
                                                                 use_knowhere_search_pool);
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::RangeSearch(dataset, std::move(cfg), bitset);
    }

    int64_t
    Dim() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Dim();
    }

    int64_t
    Size() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Size();
    }

    int64_t
    Count() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Count();
    }

    std::string
    Type() const override {
        return use_wand ? knowhere::IndexEnum::INDEX_SPARSE_WAND_CC
                        : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC;
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        ReadPermission permission(*this);

        if (raw_data_.empty()) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "GetVectorByIds failed: raw data is empty");
        }

        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();
        auto data = std::make_unique<sparse::SparseRow<T>[]>(rows);
        int64_t dim = 0;

        try {
            for (int64_t i = 0; i < rows; ++i) {
                data[i] = raw_data_[ids[i]];
                dim = std::max(dim, data[i].dim());
            }
        } catch (std::exception& e) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "GetVectorByIds failed: " + std::string(e.what()));
        }

        auto res = GenResultDataSet(rows, dim, data.release());
        res->SetIsSparse(true);

        return res;
    }

    [[nodiscard]] bool
    HasRawData(const std::string& metric_type) const override {
        return IsMetricType(metric_type, metric::IP);
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        return Status::not_implemented;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        return Status::not_implemented;
    }

 private:
    struct ReadPermission {
        ReadPermission(const SparseInvertedIndexNodeCC& node) : node_(node) {
            std::unique_lock<std::mutex> lock(node_.mutex_);
            uint64_t task_id = node_.next_task_id_++;
            // read task may execute only after all add tasks that come before it have finished.
            if (!node_.add_tasks_.empty() && task_id > node_.add_tasks_.front()) {
                node_.cv_.wait(
                    lock, [this, task_id]() { return node_.add_tasks_.empty() || task_id < node_.add_tasks_.front(); });
            }
            // read task is allowed to run, block all add tasks
            node_.active_readers_++;
        }

        ~ReadPermission() {
            std::unique_lock<std::mutex> lock(node_.mutex_);
            node_.active_readers_--;
            node_.current_task_id_++;
            node_.cv_.notify_all();
        }
        const SparseInvertedIndexNodeCC& node_;
    };

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    mutable int64_t active_readers_ = 0;
    mutable std::queue<uint64_t> add_tasks_;
    mutable uint64_t next_task_id_ = 0;
    mutable uint64_t current_task_id_ = 0;
    mutable std::vector<sparse::SparseRow<T>> raw_data_ = {};
};  // class SparseInvertedIndexNodeCC

KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_INVERTED_INDEX, SparseInvertedIndexNode, knowhere::feature::MMAP,
                                             /*use_wand=*/false)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_WAND, SparseInvertedIndexNode, knowhere::feature::MMAP,
                                             /*use_wand=*/true)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_INVERTED_INDEX_CC, SparseInvertedIndexNodeCC,
                                             knowhere::feature::MMAP,
                                             /*use_wand=*/false)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_WAND_CC, SparseInvertedIndexNodeCC, knowhere::feature::MMAP,
                                             /*use_wand=*/true)
}  // namespace knowhere
