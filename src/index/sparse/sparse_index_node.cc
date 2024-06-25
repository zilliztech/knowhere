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

#include "index/hnsw/hnsw_config.h"
#include "index/sparse/sparse_inverted_index.h"
#include "index/sparse/sparse_inverted_index_config.h"
#include "io/file_io.h"
#include "io/memory_io.h"
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

// Inverted Index impl for sparse vectors. May optionally use WAND algorithm to speed up search.
//
// Not overriding RangeSerach, will use the default implementation in IndexNode.
template <typename T, bool use_wand>
class SparseInvertedIndexNode : public IndexNode {
    static_assert(std::is_same_v<T, fp32>, "SparseInvertedIndexNode only support float");

 public:
    explicit SparseInvertedIndexNode(const int32_t& /*version*/, const Object& /*object*/)
        : search_pool_(ThreadPool::GetGlobalSearchThreadPool()) {
    }

    ~SparseInvertedIndexNode() override {
        delete_index();
    }

    Status
    Train(const DataSetPtr dataset, const Config& config) override {
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(config);
        if (!IsMetricType(cfg.metric_type.value(), metric::IP)) {
            LOG_KNOWHERE_ERROR_ << Type() << " only support metric_type: IP";
            return Status::invalid_metric_type;
        }
        auto drop_ratio_build = cfg.drop_ratio_build.value_or(0.0f);
        auto index = new sparse::InvertedIndex<T>();
        index->SetUseWand(use_wand);
        index->Train(static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor()), dataset->GetRows(),
                     drop_ratio_build);
        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type() << " deleting old index during train";
            delete_index();
        }
        index_ = index;
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, const Config& config) override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not add data to empty " << Type();
            return Status::empty_index;
        }
        return index_->Add(static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor()), dataset->GetRows(),
                           dataset->GetDim());
    }

    [[nodiscard]] expected<DataSetPtr>
    Search(const DataSetPtr dataset, const Config& config, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not search empty " << Type();
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(config);
        auto nq = dataset->GetRows();
        auto queries = static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor());
        auto k = cfg.k.value();
        auto refine_factor = cfg.refine_factor.value_or(10);
        auto drop_ratio_search = cfg.drop_ratio_search.value_or(0.0f);

        auto p_id = std::make_unique<sparse::label_t[]>(nq * k);
        auto p_dist = std::make_unique<float[]>(nq * k);

        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (int64_t idx = 0; idx < nq; ++idx) {
            futs.emplace_back(search_pool_->push([&, idx = idx, p_id = p_id.get(), p_dist = p_dist.get()]() {
                index_->Search(queries[idx], k, drop_ratio_search, p_dist + idx * k, p_id + idx * k, refine_factor,
                               bitset);
            }));
        }
        WaitAllSuccess(futs);
        return GenResultDataSet(nq, k, p_id.release(), p_dist.release());
    }

    // TODO: for now inverted index and wand use the same impl for AnnIterator.
    [[nodiscard]] expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, const Config& config, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "creating iterator on empty index";
            return expected<std::vector<std::shared_ptr<IndexNode::iterator>>>::Err(Status::empty_index,
                                                                                    "index not loaded");
        }
        auto nq = dataset->GetRows();
        auto queries = static_cast<const sparse::SparseRow<T>*>(dataset->GetTensor());

        auto cfg = static_cast<const SparseInvertedIndexConfig&>(config);
        auto drop_ratio_search = cfg.drop_ratio_search.value_or(0.0f);

        auto vec = std::vector<std::shared_ptr<IndexNode::iterator>>(nq, nullptr);
        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (int i = 0; i < nq; ++i) {
            futs.emplace_back(search_pool_->push([&, i]() {
                vec[i].reset(new PrecomputedDistanceIterator(
                    index_->GetAllDistances(queries[i], drop_ratio_search, bitset), true));
            }));
        }
        WaitAllSuccess(futs);

        return vec;
    }

    [[nodiscard]] expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        if (!index_) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();

        auto data = std::make_unique<sparse::SparseRow<T>[]>(rows);
        int64_t dim = 0;
        try {
            for (int64_t i = 0; i < rows; ++i) {
                auto& target = data[i];
                index_->GetVectorById(ids[i], target);
                dim = std::max(dim, target.dim());
            }
        } catch (std::exception& e) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "GetVectorByIds failed");
        }
        auto res = GenResultDataSet(rows, dim, data.release());
        res->SetIsSparse(true);
        return res;
    }

    [[nodiscard]] bool
    HasRawData(const std::string& metric_type) const override {
        return true;
    }

    [[nodiscard]] expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        throw std::runtime_error("GetIndexMeta not supported for current index type");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not serialize empty " << Type();
            return Status::empty_index;
        }
        MemoryIOWriter writer;
        RETURN_IF_ERROR(index_->Save(writer));
        std::shared_ptr<uint8_t[]> data(writer.data());
        binset.Append(Type(), data, writer.tellg());
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        if (index_) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, deleting old";
            delete_index();
        }
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid BinarySet.";
            return Status::invalid_binary_set;
        }
        MemoryIOReader reader(binary->data.get(), binary->size);
        index_ = new sparse::InvertedIndex<T>();
        // no need to set use_wand_ of index_, since it will be set in Load()
        return index_->Load(reader, false);
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        if (index_) {
            LOG_KNOWHERE_WARNING_ << Type() << " has already been created, deleting old";
            delete_index();
        }
        auto cfg = static_cast<const knowhere::BaseConfig&>(config);
        auto reader = knowhere::FileReader(filename);
        map_size_ = reader.size();
        int map_flags = MAP_SHARED;
        if (cfg.enable_mmap_pop.has_value() && cfg.enable_mmap_pop.value()) {
#ifdef MAP_POPULATE
            map_flags |= MAP_POPULATE;
#endif
        }
        map_ = static_cast<char*>(mmap(nullptr, map_size_, PROT_READ, map_flags, reader.descriptor(), 0));
        if (map_ == MAP_FAILED) {
            LOG_KNOWHERE_ERROR_ << "Failed to mmap file: " << strerror(errno);
            return Status::disk_file_error;
        }
        if (madvise(map_, map_size_, MADV_RANDOM) != 0) {
            LOG_KNOWHERE_WARNING_ << "Failed to madvise file: " << strerror(errno);
        }
        index_ = new sparse::InvertedIndex<T>();
        MemoryIOReader map_reader((uint8_t*)map_, map_size_);
        return index_->Load(map_reader, true);
    }

    [[nodiscard]] std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<SparseInvertedIndexConfig>();
    }

    // note that the Dim of a sparse vector index may change as new vectors are added
    [[nodiscard]] int64_t
    Dim() const override {
        return index_ ? index_->n_cols() : 0;
    }

    [[nodiscard]] int64_t
    Size() const override {
        return index_ ? index_->size() : 0;
    }

    [[nodiscard]] int64_t
    Count() const override {
        return index_ ? index_->n_rows() : 0;
    }

    [[nodiscard]] std::string
    Type() const override {
        return use_wand ? knowhere::IndexEnum::INDEX_SPARSE_WAND : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
    }

 private:
    void
    delete_index() {
        if (index_ != nullptr) {
            delete index_;
            index_ = nullptr;
        }
        if (map_ != nullptr) {
            auto res = munmap(map_, map_size_);
            if (res != 0) {
                LOG_KNOWHERE_ERROR_ << "Failed to munmap when trying to delete index: " << strerror(errno);
            }
            map_ = nullptr;
            map_size_ = 0;
        }
    }

    sparse::InvertedIndex<T>* index_{};
    std::shared_ptr<ThreadPool> search_pool_;

    // if map_ is not nullptr, it means the index is mmapped from disk.
    char* map_ = nullptr;
    size_t map_size_ = 0;
};  // class SparseInvertedIndexNode

KNOWHERE_SIMPLE_REGISTER_GLOBAL(SPARSE_INVERTED_INDEX, SparseInvertedIndexNode, fp32, /*use_wand=*/false);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(SPARSE_WAND, SparseInvertedIndexNode, fp32, /*use_wand=*/true);

}  // namespace knowhere
