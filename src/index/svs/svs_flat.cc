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

#ifdef KNOWHERE_WITH_SVS

#include "common/metric.h"
#include "faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h"
#include "faiss/impl/mapped_io.h"
#include "faiss/index_io.h"
#include "faiss/svs/IndexSVSFlat.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/task.h"
#include "knowhere/context.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

template <typename DataType>
class SvsFlatIndexNode : public IndexNode {
 public:
    SvsFlatIndexNode(const int32_t version, const Object& object) : IndexNode(version), index_(nullptr) {
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        const BaseConfig& f_cfg = static_cast<const BaseConfig&>(*cfg);

        auto metric = Str2FaissMetricType(f_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "unsupported metric type: " << f_cfg.metric_type.value();
            return metric.error();
        }

        index_ = std::make_unique<faiss::IndexSVSFlat>(dataset->GetDim(), metric.value());
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to empty index.";
            return Status::empty_index;
        }

        const BaseConfig& f_cfg = static_cast<const BaseConfig&>(*cfg);
        bool is_cosine = IsMetricType(f_cfg.metric_type.value(), knowhere::metric::COSINE);
        if (is_cosine) {
            NormalizeDataset<DataType>(dataset);
        }

        auto x = dataset->GetTensor();
        auto n = dataset->GetRows();
        index_->add(n, (const float*)x);
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        if (!bitset.empty()) {
            return expected<DataSetPtr>::Err(Status::not_implemented, "SVS Flat does not support bitset filtering");
        }

        const BaseConfig& f_cfg = static_cast<const BaseConfig&>(*cfg);
        auto k = f_cfg.k.value();
        auto nq = dataset->GetRows();
        auto x = dataset->GetTensor();
        auto dim = dataset->GetDim();
        bool is_cosine = IsMetricType(f_cfg.metric_type.value(), knowhere::metric::COSINE);

        auto ids = std::make_unique<int64_t[]>(k * nq);
        auto distances = std::make_unique<float[]>(k * nq);

        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(nq);
            for (int i = 0; i < nq; ++i) {
                futs.emplace_back(search_pool_->push([&, index = i] {
                    knowhere::checkCancellation(op_context);
                    ThreadPool::ScopedSearchOmpSetter setter(1);

                    auto cur_ids = ids.get() + k * index;
                    auto cur_dis = distances.get() + k * index;
                    auto cur_query = (const float*)x + dim * index;
                    std::unique_ptr<float[]> copied_query;
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    index_->search(1, cur_query, k, cur_dis, cur_ids);
                }));
            }
            WaitAllSuccess(futs);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        return GenResultDataSet(nq, k, std::move(ids), std::move(distances));
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* op_context) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "SVS Flat does not support range search");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "SVS Flat does not support vector retrieval");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config>) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty index.";
            return Status::empty_index;
        }

        try {
            MemoryIOWriter writer;
            faiss::write_index(index_.get(), &writer);
            std::shared_ptr<uint8_t[]> data(writer.data());
            binset.Append(Type(), data, writer.tellg());
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config>) override {
        auto binary = binset.GetByNames({Type()});
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }

        MemoryIOReader reader(binary->data.get(), binary->size);
        try {
            auto index = std::unique_ptr<faiss::Index>(faiss::read_index(&reader));
            if (!dynamic_cast<faiss::IndexSVSFlat*>(index.get())) {
                return Status::invalid_binary_set;
            }
            index_.reset(static_cast<faiss::IndexSVSFlat*>(index.release()));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> cfg) override {
        auto base_cfg = static_cast<const knowhere::BaseConfig&>(*cfg);
        int io_flags = 0;
        if (base_cfg.enable_mmap.value()) {
            io_flags |= faiss::IO_FLAG_MMAP_IFC;
        }

        try {
            std::unique_ptr<faiss::Index> index;

            if ((io_flags & faiss::IO_FLAG_MMAP_IFC) == faiss::IO_FLAG_MMAP_IFC) {
                // enable mmap-supporting IOReader
                auto owner = std::make_shared<faiss::MmappedFileMappingOwner>(filename.data());
                faiss::MappedFileIOReader reader(owner);

                index = std::unique_ptr<faiss::Index>(faiss::read_index(&reader, io_flags));
            } else {
                index = std::unique_ptr<faiss::Index>(faiss::read_index(filename.data(), io_flags));
            }

            if (!dynamic_cast<faiss::IndexSVSFlat*>(index.get())) {
                return Status::invalid_binary_set;
            }
            index_.reset(static_cast<faiss::IndexSVSFlat*>(index.release()));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<BaseConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    int64_t
    Dim() const override {
        if (!index_) {
            return -1;
        }

        return index_->d;
    }

    int64_t
    Size() const override {
        if (!index_) {
            return 0;
        }

        faiss::cppcontrib::knowhere::CountSizeIOWriter writer;
        faiss::write_index(index_.get(), &writer);
        return writer.total_size;
    }

    int64_t
    Count() const override {
        if (!index_) {
            return -1;
        }

        return index_->ntotal;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_SVS_FLAT;
    }

 private:
    std::unique_ptr<faiss::IndexSVSFlat> index_;
    std::shared_ptr<ThreadPool> search_pool_;
};

// temporarily removed `MMAP` until it's fully honored by SVS

KNOWHERE_MOCK_REGISTER_DENSE_FLOAT_ALL_GLOBAL(SVS_FLAT, SvsFlatIndexNode,
                                              knowhere::feature::NO_TRAIN | knowhere::feature::KNN)

}  // namespace knowhere

#endif  // KNOWHERE_WITH_SVS
