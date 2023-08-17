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

#include "common/metric.h"
#include "common/range_util.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexFlat.h"
#include "faiss/index_io.h"
#include "index/flat/flat_config.h"
#include "io/FaissIO.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

template <typename T>
class FlatIndexNode : public IndexNode {
 public:
    FlatIndexNode(const Object&) : index_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexFlat>::value || std::is_same<T, faiss::IndexBinaryFlat>::value,
                      "not support");
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
    }

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);

        // do normalize for COSINE metric type
        if (IsMetricType(f_cfg.metric_type.value(), knowhere::metric::COSINE)) {
            Normalize(dataset);
        }

        auto metric = Str2FaissMetricType(f_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_WARNING_ << "please check metric type: " << f_cfg.metric_type.value();
            return metric.error();
        }
        index_ = std::make_unique<T>(dataset.GetDim(), metric.value());
        return Status::success;
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        auto x = dataset.GetTensor();
        auto n = dataset.GetRows();
        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            index_->add(n, (const float*)x);
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            index_->add(n, (const uint8_t*)x);
        }
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            return Status::empty_index;
        }

        DataSetPtr results = std::make_shared<DataSet>();
        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        bool is_cosine = IsMetricType(f_cfg.metric_type.value(), knowhere::metric::COSINE);

        auto k = f_cfg.k.value();
        auto nq = dataset.GetRows();
        auto x = dataset.GetTensor();
        auto dim = dataset.GetDim();

        auto len = k * nq;
        int64_t* ids = nullptr;
        float* distances = nullptr;
        try {
            ids = new (std::nothrow) int64_t[len];
            distances = new (std::nothrow) float[len];
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(nq);
            for (int i = 0; i < nq; ++i) {
                futs.emplace_back(search_pool_->push([&, index = i] {
                    ThreadPool::ScopedOmpSetter setter(1);
                    auto cur_ids = ids + k * index;
                    auto cur_dis = distances + k * index;
                    if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
                        auto cur_query = (const float*)x + dim * index;
                        std::unique_ptr<float[]> copied_query = nullptr;
                        if (is_cosine) {
                            copied_query = CopyAndNormalizeFloatVec(cur_query, dim);
                            cur_query = copied_query.get();
                        }
                        index_->search(1, cur_query, k, cur_dis, cur_ids, bitset);
                    }
                    if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
                        auto cur_i_dis = reinterpret_cast<int32_t*>(cur_dis);
                        index_->search(1, (const uint8_t*)x + index * dim / 8, k, cur_i_dis, cur_ids, bitset);
                        if (index_->metric_type == faiss::METRIC_Hamming) {
                            for (int64_t j = 0; j < k; j++) {
                                cur_dis[j] = static_cast<float>(cur_i_dis[j]);
                            }
                        }
                    }
                }));
            }
            for (auto& fut : futs) {
                fut.wait();
            }
        } catch (const std::exception& e) {
            std::unique_ptr<int64_t[]> auto_delete_ids(ids);
            std::unique_ptr<float[]> auto_delete_dis(distances);
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }

        return GenResultDataSet(nq, k, ids, distances);
    }

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "range search on empty index";
            return Status::empty_index;
        }

        const FlatConfig& f_cfg = static_cast<const FlatConfig&>(cfg);
        bool is_cosine = IsMetricType(f_cfg.metric_type.value(), knowhere::metric::COSINE);

        auto nq = dataset.GetRows();
        auto xq = dataset.GetTensor();
        auto dim = dataset.GetDim();

        float radius = f_cfg.radius.value();
        float range_filter = f_cfg.range_filter.value();
        bool is_ip = (index_->metric_type == faiss::METRIC_INNER_PRODUCT);

        int64_t* ids = nullptr;
        float* distances = nullptr;
        size_t* lims = nullptr;

        std::vector<std::vector<int64_t>> result_id_array(nq);
        std::vector<std::vector<float>> result_dist_array(nq);
        std::vector<size_t> result_size(nq);
        std::vector<size_t> result_lims(nq + 1);

        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(nq);
            for (int i = 0; i < nq; ++i) {
                futs.emplace_back(search_pool_->push([&, index = i] {
                    ThreadPool::ScopedOmpSetter setter(1);
                    faiss::RangeSearchResult res(1);
                    if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
                        auto cur_query = (const float*)xq + dim * index;
                        std::unique_ptr<float[]> copied_query = nullptr;
                        if (is_cosine) {
                            copied_query = CopyAndNormalizeFloatVec(cur_query, dim);
                            cur_query = copied_query.get();
                        }
                        index_->range_search(1, cur_query, radius, &res, bitset);
                    }
                    if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
                        index_->range_search(1, (const uint8_t*)xq + index * dim / 8, radius, &res, bitset);
                    }
                    auto elem_cnt = res.lims[1];
                    result_dist_array[index].resize(elem_cnt);
                    result_id_array[index].resize(elem_cnt);
                    result_size[index] = elem_cnt;
                    for (size_t j = 0; j < elem_cnt; j++) {
                        result_dist_array[index][j] = res.distances[j];
                        result_id_array[index][j] = res.labels[j];
                    }
                    if (f_cfg.range_filter.value() != defaultRangeFilter) {
                        FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, radius,
                                                        range_filter);
                    }
                }));
            }
            for (auto& fut : futs) {
                fut.wait();
            }
            GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius, range_filter, distances, ids,
                                 lims);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }

        return GenResultDataSet(nq, ids, distances, lims);
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override {
        auto dim = Dim();
        auto rows = dataset.GetRows();
        auto ids = dataset.GetIds();
        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            float* data = nullptr;
            try {
                data = new float[rows * dim];
                for (int64_t i = 0; i < rows; i++) {
                    index_->reconstruct(ids[i], data + i * dim);
                }
                return GenResultDataSet(rows, dim, data);
            } catch (const std::exception& e) {
                std::unique_ptr<float[]> auto_del(data);
                LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
                return Status::faiss_inner_error;
            }
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            uint8_t* data = nullptr;
            try {
                data = new uint8_t[rows * dim / 8];
                for (int64_t i = 0; i < rows; i++) {
                    index_->reconstruct(ids[i], data + i * dim / 8);
                }
                return GenResultDataSet(rows, dim, data);
            } catch (const std::exception& e) {
                std::unique_ptr<uint8_t[]> auto_del(data);
                LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
                return Status::faiss_inner_error;
            }
        }
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            return !IsMetricType(metric_type, metric::COSINE);
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            return true;
        }
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        return Status::not_implemented;
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty index.";
            return Status::empty_index;
        }
        try {
            MemoryIOWriter writer;
            if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
                faiss::write_index(index_.get(), &writer);
            }
            if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
                faiss::write_index_binary(index_.get(), &writer);
            }
            std::shared_ptr<uint8_t[]> data(writer.data_);
            binset.Append(Type(), data, writer.rp);
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        std::vector<std::string> names = {"IVF",        // compatible with knowhere-1.x
                                          "BinaryIVF",  // compatible with knowhere-1.x
                                          Type()};
        auto binary = binset.GetByNames(names);
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }

        MemoryIOReader reader;
        reader.total = binary->size;
        reader.data_ = binary->data.get();
        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            faiss::Index* index = faiss::read_index(&reader);
            index_.reset(static_cast<T*>(index));
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            faiss::IndexBinary* index = faiss::read_index_binary(&reader);
            index_.reset(static_cast<T*>(index));
        }
        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        auto cfg = static_cast<const knowhere::BaseConfig&>(config);

        int io_flags = 0;
        if (cfg.enable_mmap.value()) {
            io_flags |= faiss::IO_FLAG_MMAP;
        }

        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            faiss::Index* index = faiss::read_index(filename.data(), io_flags);
            index_.reset(static_cast<T*>(index));
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            faiss::IndexBinary* index = faiss::read_index_binary(filename.data(), io_flags);
            index_.reset(static_cast<T*>(index));
        }
        return Status::success;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<FlatConfig>();
    }

    int64_t
    Dim() const override {
        return index_->d;
    }

    int64_t
    Size() const override {
        return index_->ntotal * index_->d * sizeof(float);
    }

    int64_t
    Count() const override {
        return index_->ntotal;
    }

    std::string
    Type() const override {
        if constexpr (std::is_same<T, faiss::IndexFlat>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IDMAP;
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryFlat>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP;
        }
    }

 private:
    std::unique_ptr<T> index_;
    std::shared_ptr<ThreadPool> search_pool_;
};

KNOWHERE_REGISTER_GLOBAL(FLAT,
                         [](const Object& object) { return Index<FlatIndexNode<faiss::IndexFlat>>::Create(object); });
KNOWHERE_REGISTER_GLOBAL(BINFLAT, [](const Object& object) {
    return Index<FlatIndexNode<faiss::IndexBinaryFlat>>::Create(object);
});
KNOWHERE_REGISTER_GLOBAL(BIN_FLAT, [](const Object& object) {
    return Index<FlatIndexNode<faiss::IndexBinaryFlat>>::Create(object);
});

}  // namespace knowhere
