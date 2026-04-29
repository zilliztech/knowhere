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

#include <optional>

#include "common/metric.h"
#include "faiss/cppcontrib/knowhere/MetricType.h"
#include "faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h"
#include "faiss/impl/mapped_io.h"
#include "faiss/index_io.h"
#include "faiss/svs/IndexSVSVamana.h"
#include "faiss/svs/IndexSVSVamanaLVQ.h"
#include "faiss/svs/IndexSVSVamanaLeanVec.h"
#include "index/svs/svs_config.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/task.h"
#include "knowhere/context.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/utils.h"

namespace knowhere {

namespace {

std::optional<faiss::SVSStorageKind>
str_to_svs_storage_kind(const std::string& s) {
    if (s == "fp32")
        return faiss::SVS_FP32;
    if (s == "fp16")
        return faiss::SVS_FP16;
    if (s == "sqi8")
        return faiss::SVS_SQI8;
    if (s == "lvq4x0")
        return faiss::SVS_LVQ4x0;
    if (s == "lvq4x4")
        return faiss::SVS_LVQ4x4;
    if (s == "lvq4x8")
        return faiss::SVS_LVQ4x8;
    if (s == "leanvec4x4")
        return faiss::SVS_LeanVec4x4;
    if (s == "leanvec4x8")
        return faiss::SVS_LeanVec4x8;
    if (s == "leanvec8x8")
        return faiss::SVS_LeanVec8x8;
    return std::nullopt;
}

}  // namespace

template <typename DataType>
class SvsVamanaIndexNode : public IndexNode {
 public:
    SvsVamanaIndexNode(const int32_t version, const Object& object) : IndexNode(version), index_(nullptr) {
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        const SvsVamanaConfig& v_cfg = static_cast<const SvsVamanaConfig&>(*cfg);

        auto metric = Str2FaissMetricType(v_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "unsupported metric type: " << v_cfg.metric_type.value();
            return metric.error();
        }

        auto storage = str_to_svs_storage_kind(v_cfg.svs_storage_kind.value());
        if (!storage.has_value()) {
            LOG_KNOWHERE_ERROR_ << "unknown SVS storage kind: " << v_cfg.svs_storage_kind.value();
            return Status::invalid_args;
        }

        // Normalize training data for cosine metric (cosine = IP on normalized vectors)
        bool is_cosine = IsMetricType(v_cfg.metric_type.value(), knowhere::metric::COSINE);
        if (is_cosine) {
            NormalizeDataset<DataType>(dataset);
        }

        try {
            auto idx = CreateFaissIndex(dataset->GetDim(), v_cfg.svs_graph_max_degree.value(), metric.value(),
                                        storage.value(), v_cfg);
            idx->construction_window_size = v_cfg.svs_construction_window_size.value();
            if (v_cfg.svs_alpha.has_value()) {
                idx->alpha = v_cfg.svs_alpha.value();
            }
            idx->search_window_size = v_cfg.svs_search_window_size.value();
            idx->search_buffer_capacity = v_cfg.svs_search_buffer_capacity.value();
            index_ = std::move(idx);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "SVS Vamana train error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to empty index.";
            return Status::empty_index;
        }

        const SvsVamanaConfig& v_cfg = static_cast<const SvsVamanaConfig&>(*cfg);
        bool is_cosine = IsMetricType(v_cfg.metric_type.value(), knowhere::metric::COSINE);
        if (is_cosine) {
            NormalizeDataset<DataType>(dataset);
        }

        auto x = dataset->GetTensor();
        auto n = dataset->GetRows();
        try {
            index_->add(n, (const float*)x);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "SVS Vamana add error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        const SvsVamanaConfig& v_cfg = static_cast<const SvsVamanaConfig&>(*cfg);
        auto k = v_cfg.k.value();
        auto nq = dataset->GetRows();
        auto x = dataset->GetTensor();
        auto dim = dataset->GetDim();
        bool is_cosine = IsMetricType(v_cfg.metric_type.value(), knowhere::metric::COSINE);

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

                    faiss::SearchParametersSVSVamana sp;
                    sp.search_window_size = v_cfg.svs_search_window_size.value();
                    sp.search_buffer_capacity = v_cfg.svs_search_buffer_capacity.value();
                    std::unique_ptr<BitsetViewIDSelector> bw_idselector;
                    if (!bitset.empty()) {
                        bw_idselector = std::make_unique<BitsetViewIDSelector>(bitset);
                        sp.sel = bw_idselector.get();
                    }

                    index_->search(1, cur_query, k, cur_dis, cur_ids, &sp);
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
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "range search on empty index";
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        const SvsVamanaConfig& v_cfg = static_cast<const SvsVamanaConfig&>(*cfg);
        auto nq = dataset->GetRows();
        auto x = dataset->GetTensor();
        auto dim = dataset->GetDim();
        auto radius = v_cfg.radius.value();
        auto range_filter = v_cfg.range_filter.value();
        bool is_cosine = IsMetricType(v_cfg.metric_type.value(), knowhere::metric::COSINE);
        bool is_similarity = faiss::cppcontrib::knowhere::is_similarity_metric(index_->metric_type);

        std::vector<std::vector<float>> result_dist_array(nq);
        std::vector<std::vector<int64_t>> result_id_array(nq);

        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(nq);
            for (int i = 0; i < nq; ++i) {
                futs.emplace_back(search_pool_->push([&, index = i] {
                    knowhere::checkCancellation(op_context);
                    ThreadPool::ScopedSearchOmpSetter setter(1);

                    auto cur_query = (const float*)x + dim * index;
                    std::unique_ptr<float[]> copied_query;
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    faiss::SearchParametersSVSVamana sp;
                    sp.search_window_size = v_cfg.svs_search_window_size.value();
                    sp.search_buffer_capacity = v_cfg.svs_search_buffer_capacity.value();
                    std::unique_ptr<BitsetViewIDSelector> bw_idselector;
                    if (!bitset.empty()) {
                        bw_idselector = std::make_unique<BitsetViewIDSelector>(bitset);
                        sp.sel = bw_idselector.get();
                    }

                    faiss::RangeSearchResult res(1);
                    index_->range_search(1, cur_query, radius, &res, &sp);

                    auto elem_cnt = res.lims[1];
                    result_dist_array[index].resize(elem_cnt);
                    result_id_array[index].resize(elem_cnt);
                    for (size_t j = 0; j < elem_cnt; j++) {
                        result_dist_array[index][j] = res.distances[j];
                        result_id_array[index][j] = res.labels[j];
                    }
                    if (range_filter != defaultRangeFilter) {
                        FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_similarity,
                                                        radius, range_filter);
                    }
                }));
            }
            WaitAllSuccess(futs);
            auto range_search_result =
                GetRangeSearchResult(result_dist_array, result_id_array, is_similarity, nq, radius, range_filter);
            return GenResultDataSet(nq, std::move(range_search_result));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "SVS Vamana does not support vector retrieval");
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
            if (!dynamic_cast<faiss::IndexSVSVamana*>(index.get())) {
                return Status::invalid_binary_set;
            }
            index_.reset(static_cast<faiss::IndexSVSVamana*>(index.release()));
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

            if (!dynamic_cast<faiss::IndexSVSVamana*>(index.get())) {
                return Status::invalid_binary_set;
            }
            index_.reset(static_cast<faiss::IndexSVSVamana*>(index.release()));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner faiss: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<SvsVamanaConfig>();
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
        return knowhere::IndexEnum::INDEX_SVS_VAMANA;
    }

 protected:
    virtual std::unique_ptr<faiss::IndexSVSVamana>
    CreateFaissIndex(int64_t dim, int64_t degree, faiss::MetricType metric, faiss::SVSStorageKind storage,
                     const SvsVamanaConfig& cfg) {
        return std::make_unique<faiss::IndexSVSVamana>(dim, degree, metric, storage);
    }

    std::unique_ptr<faiss::IndexSVSVamana> index_;
    std::shared_ptr<ThreadPool> search_pool_;
};

template <typename DataType>
class SvsVamanaLvqIndexNode : public SvsVamanaIndexNode<DataType> {
 public:
    using SvsVamanaIndexNode<DataType>::SvsVamanaIndexNode;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
            LOG_KNOWHERE_ERROR_ << "LVQ/LeanVec is not available in this SVS runtime build";
            return Status::invalid_args;
        }

        return SvsVamanaIndexNode<DataType>::Train(dataset, cfg, use_knowhere_build_pool);
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<SvsVamanaLvqConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_SVS_VAMANA_LVQ;
    }

 protected:
    std::unique_ptr<faiss::IndexSVSVamana>
    CreateFaissIndex(int64_t dim, int64_t degree, faiss::MetricType metric, faiss::SVSStorageKind storage,
                     const SvsVamanaConfig& cfg) override {
        return std::make_unique<faiss::IndexSVSVamanaLVQ>(dim, degree, metric, storage);
    }
};

template <typename DataType>
class SvsVamanaLeanVecIndexNode : public SvsVamanaIndexNode<DataType> {
 public:
    using SvsVamanaIndexNode<DataType>::SvsVamanaIndexNode;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
            LOG_KNOWHERE_ERROR_ << "LVQ/LeanVec is not available in this SVS runtime build";
            return Status::invalid_args;
        }

        const SvsVamanaLeanVecConfig& lv_cfg = static_cast<const SvsVamanaLeanVecConfig&>(*cfg);
        auto metric = Str2FaissMetricType(lv_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "unsupported metric type: " << lv_cfg.metric_type.value();
            return metric.error();
        }

        auto storage = str_to_svs_storage_kind(lv_cfg.svs_storage_kind.value());
        if (!storage.has_value()) {
            LOG_KNOWHERE_ERROR_ << "unknown SVS storage kind: " << lv_cfg.svs_storage_kind.value();
            return Status::invalid_args;
        }

        // Normalize training data for cosine metric (cosine = IP on normalized vectors)
        bool is_cosine = IsMetricType(lv_cfg.metric_type.value(), knowhere::metric::COSINE);
        if (is_cosine) {
            NormalizeDataset<DataType>(dataset);
        }

        try {
            size_t leanvec_dim = lv_cfg.svs_leanvec_dim.value();
            auto idx = std::make_unique<faiss::IndexSVSVamanaLeanVec>(
                dataset->GetDim(), lv_cfg.svs_graph_max_degree.value(), metric.value(), leanvec_dim, storage.value());
            idx->construction_window_size = lv_cfg.svs_construction_window_size.value();
            if (lv_cfg.svs_alpha.has_value()) {
                idx->alpha = lv_cfg.svs_alpha.value();
            }
            idx->search_window_size = lv_cfg.svs_search_window_size.value();
            idx->search_buffer_capacity = lv_cfg.svs_search_buffer_capacity.value();

            // LeanVec requires training before adding vectors
            idx->train(dataset->GetRows(), (const float*)dataset->GetTensor());

            this->index_ = std::move(idx);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "SVS Vamana LeanVec train error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<SvsVamanaLeanVecConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC;
    }
};

// temporarily removed `MMAP` until it's fully honored by SVS

KNOWHERE_MOCK_REGISTER_DENSE_FLOAT_ALL_GLOBAL(SVS_VAMANA, SvsVamanaIndexNode, knowhere::feature::NONE)

KNOWHERE_MOCK_REGISTER_DENSE_FLOAT_ALL_GLOBAL(SVS_VAMANA_LVQ, SvsVamanaLvqIndexNode, knowhere::feature::NONE)

KNOWHERE_MOCK_REGISTER_DENSE_FLOAT_ALL_GLOBAL(SVS_VAMANA_LEANVEC, SvsVamanaLeanVecIndexNode, knowhere::feature::NONE)

}  // namespace knowhere

#endif  // KNOWHERE_WITH_SVS
