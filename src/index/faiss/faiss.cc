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
#include "faiss/index_io.h"
#include "faiss/index_factory.h"
#include "index/faiss/faiss_config.h"
#include "io/memory_io.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

class FaissIndexNode : public IndexNode {
 public:
    FaissIndexNode(const int32_t version, const Object& object);

    Status
    Train(const DataSet& dataset, const Config& cfg) override;

    Status
    Add(const DataSet& dataset, const Config& cfg) override;

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;
    
    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override;

    bool
    HasRawData(const std::string& metric_type) const override;

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override;

    Status
    Serialize(BinarySet& binset) const override;

    Status
    Deserialize(const BinarySet& binset, const Config& config) override;

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override;

    std::unique_ptr<BaseConfig>
    CreateConfig() const override;

    int64_t
    Dim() const override;

    int64_t
    Size() const override;

    int64_t
    Count() const override;

    std::string
    Type() const override;

 private:
    std::unique_ptr<faiss::Index> index_;
    std::shared_ptr<ThreadPool> search_pool_;    
};

//
FaissIndexNode::FaissIndexNode(const int32_t version, const Object& object) : index_(nullptr) {
    search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
}

//
Status
FaissIndexNode::Train(const DataSet& dataset, const Config& cfg) {
    const FaissConfig& f_cfg = static_cast<const FaissConfig&>(cfg);

    auto metric = Str2FaissMetricType(f_cfg.metric_type.value());
    if (!metric.has_value()) {
        LOG_KNOWHERE_WARNING_ << "please check metric type: " << f_cfg.metric_type.value();
        return metric.error();
    }
    if (!f_cfg.factory_string.has_value()) {
        LOG_KNOWHERE_WARNING_ << "factory string for faiss index is undefined";
        return Status::invalid_args;
    }

    index_.reset(faiss::index_factory(dataset.GetDim(), f_cfg.factory_string.value().c_str(), metric.value()));

    auto rows = dataset.GetRows();
    auto dim = dataset.GetDim();
    auto data = dataset.GetTensor();

    index_->train(rows, (const float*)data);
    return Status::success;
}

//
Status
FaissIndexNode::Add(const DataSet& dataset, const Config& cfg) {
    if (!this->index_) {
        LOG_KNOWHERE_ERROR_ << "Can not add data to empty index.";
        return Status::empty_index;
    }
    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    const BaseConfig& base_cfg = static_cast<const FaissConfig&>(cfg);
    std::unique_ptr<ThreadPool::ScopedOmpSetter> setter;
    if (base_cfg.num_build_thread.has_value()) {
        setter = std::make_unique<ThreadPool::ScopedOmpSetter>(base_cfg.num_build_thread.value());
    }
    try {
        index_->add(rows, (const float*)data);
    } catch (std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

//
expected<DataSetPtr>
FaissIndexNode::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_) {
        LOG_KNOWHERE_WARNING_ << "search on empty index";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
    }
    if (!this->index_->is_trained) {
        LOG_KNOWHERE_WARNING_ << "index not trained";
        return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
    }

    auto dim = dataset.GetDim();
    auto rows = dataset.GetRows();
    auto data = dataset.GetTensor();

    const BaseConfig& base_cfg = static_cast<const BaseConfig&>(cfg);
    bool is_cosine = IsMetricType(base_cfg.metric_type.value(), knowhere::metric::COSINE);

    auto k = base_cfg.k.value();
    //auto nprobe = ivf_cfg.nprobe.value();

    faiss::Index::idx_t* ids(new (std::nothrow) faiss::Index::idx_t[rows * k]);
    float* distances(new (std::nothrow) float[rows * k]);
    try {
        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(rows);
        for (int i = 0; i < rows; ++i) {
            futs.emplace_back(search_pool_->push([&, index = i] {
                ThreadPool::ScopedOmpSetter setter(1);
                auto cur_ids = ids + k * index;
                auto cur_dis = distances + k * index;

                std::unique_ptr<float[]> copied_query = nullptr;
                auto cur_query = (const float*)data + index * dim;
                if (is_cosine) {
                    copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                    cur_query = copied_query.get();
                }
                index_->search(
                    1, 
                    cur_query, 
                    k, 
                    cur_dis, 
                    cur_ids, 
                    bitset);
            }));
        }
        for (auto& fut : futs) {
            fut.wait();
        }
    } catch (const std::exception& e) {
        delete[] ids;
        delete[] distances;
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
    }

    auto res = GenResultDataSet(rows, k, ids, distances);
    return res;
}

//
expected<DataSetPtr>
FaissIndexNode::RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    if (!this->index_) {
        LOG_KNOWHERE_WARNING_ << "range search on empty index";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
    }
    if (!this->index_->is_trained) {
        LOG_KNOWHERE_WARNING_ << "index not trained";
        return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
    }

    auto nq = dataset.GetRows();
    auto xq = dataset.GetTensor();
    auto dim = dataset.GetDim();

    const BaseConfig& base_cfg = static_cast<const BaseConfig&>(cfg);
    bool is_cosine = IsMetricType(base_cfg.metric_type.value(), knowhere::metric::COSINE);

    float radius = base_cfg.radius.value();
    float range_filter = base_cfg.range_filter.value();
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
                std::unique_ptr<float[]> copied_query = nullptr;

                auto cur_query = (const float*)xq + index * dim;
                if (is_cosine) {
                    copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                    cur_query = copied_query.get();
                }

                //
                index_->range_search(1, cur_query, radius, &res, bitset);

                auto elem_cnt = res.lims[1];
                result_dist_array[index].resize(elem_cnt);
                result_id_array[index].resize(elem_cnt);
                result_size[index] = elem_cnt;
                for (size_t j = 0; j < elem_cnt; j++) {
                    result_dist_array[index][j] = res.distances[j];
                    result_id_array[index][j] = res.labels[j];
                }
                if (range_filter != defaultRangeFilter) {
                    FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, radius,
                                                    range_filter);
                }
            }));
        }
        for (auto& fut : futs) {
            fut.wait();
        }
        GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius, range_filter, distances, ids, lims);
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
    }

    return GenResultDataSet(nq, ids, distances, lims);
}

//
expected<DataSetPtr>
FaissIndexNode::GetVectorByIds(const DataSet& dataset) const {
    return expected<DataSetPtr>::Err(Status::not_implemented, "not implemented for FaissIndex");
}

//
bool
FaissIndexNode::HasRawData(const std::string& metric_type) const {
    // todo aguzhva: not implemented
    return false;
}

//
expected<DataSetPtr>
FaissIndexNode::GetIndexMeta(const Config& cfg) const {
    return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
}

//
Status
FaissIndexNode::Serialize(BinarySet& binset) const {
    try {
        MemoryIOWriter writer;
        faiss::write_index(index_.get(), &writer);
        
        std::shared_ptr<uint8_t[]> data(writer.data());
        binset.Append(Type(), data, writer.tellg());
        return Status::success;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
}

//
Status
FaissIndexNode::Deserialize(const BinarySet& binset, const Config& config) {
    std::vector<std::string> names = {Type()};
    auto binary = binset.GetByNames(names);
    if (binary == nullptr) {
        LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
        return Status::invalid_binary_set;
    }

    MemoryIOReader reader(binary->data.get(), binary->size);
    try {
        index_.reset(faiss::read_index(&reader));
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

//
Status
FaissIndexNode::DeserializeFromFile(const std::string& filename, const Config& config) {
    auto cfg = static_cast<const knowhere::BaseConfig&>(config);

    int io_flags = 0;
    if (cfg.enable_mmap.value()) {
        io_flags |= faiss::IO_FLAG_MMAP;
    }
    try {
        index_.reset(faiss::read_index(filename.data(), io_flags));
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

//
std::unique_ptr<BaseConfig>
FaissIndexNode::CreateConfig() const {
    return std::make_unique<FaissConfig>();
};

//
int64_t
FaissIndexNode::Dim() const {
    if (!index_) {
        return -1;
    }
    return index_->d;
};

int64_t
FaissIndexNode::Size() const {
    if (!index_) {
        return 0;
    }

    // slow and wrong, but universal way of estimating the size
    faiss::VectorIOWriter writer;
    faiss::write_index(index_.get(), &writer);

    return writer.data.size();
};

int64_t
FaissIndexNode::Count() const {
    if (!index_) {
        return 0;
    }
    return index_->ntotal;
};

std::string
FaissIndexNode::Type() const {
    return knowhere::IndexEnum::INDEX_FAISS;
};

KNOWHERE_REGISTER_GLOBAL(FAISS, [](const int32_t& version, const Object& object) {
    return Index<FaissIndexNode>::Create(version, object);
});

}  // namespace knowhere

