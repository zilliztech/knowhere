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

#include <faiss/utils/Heap.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "common/metric.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexCosine.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexRefine.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "faiss/index_io.h"
#include "index/hnsw/faiss_hnsw_config.h"
#include "index/hnsw/impl/IndexBruteForceWrapper.h"
#include "index/hnsw/impl/IndexHNSWWrapper.h"
#include "index/hnsw/impl/IndexWrapperCosine.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/config.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/utils.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

//
class BaseFaissIndexNode : public IndexNode {
 public:
    BaseFaissIndexNode(const int32_t& /*version*/, const Object& object) {
        build_pool = ThreadPool::GetGlobalBuildThreadPool();
        search_pool = ThreadPool::GetGlobalSearchThreadPool();
    }

    //
    Status
    Train(const DataSetPtr dataset, const Config& cfg) override {
        // config
        const BaseConfig& base_cfg = static_cast<const FaissHnswConfig&>(cfg);

        // use build_pool_ to make sure the OMP threads spawned by index_->train etc
        // can inherit the low nice value of threads in build_pool_.
        auto tryObj = build_pool
                          ->push([&] {
                              std::unique_ptr<ThreadPool::ScopedOmpSetter> setter;
                              if (base_cfg.num_build_thread.has_value()) {
                                  setter =
                                      std::make_unique<ThreadPool::ScopedOmpSetter>(base_cfg.num_build_thread.value());
                              } else {
                                  setter = std::make_unique<ThreadPool::ScopedOmpSetter>();
                              }

                              return TrainInternal(dataset, cfg);
                          })
                          .getTry();

        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "faiss internal error: " << tryObj.exception().what();
            return Status::faiss_inner_error;
        }

        return tryObj.value();
    }

    Status
    Add(const DataSetPtr dataset, const Config& cfg) override {
        const BaseConfig& base_cfg = static_cast<const FaissHnswConfig&>(cfg);

        // use build_pool_ to make sure the OMP threads spawned by index_->train etc
        // can inherit the low nice value of threads in build_pool_.
        auto tryObj = build_pool
                          ->push([&] {
                              std::unique_ptr<ThreadPool::ScopedOmpSetter> setter;
                              if (base_cfg.num_build_thread.has_value()) {
                                  setter =
                                      std::make_unique<ThreadPool::ScopedOmpSetter>(base_cfg.num_build_thread.value());
                              } else {
                                  setter = std::make_unique<ThreadPool::ScopedOmpSetter>();
                              }

                              return AddInternal(dataset, cfg);
                          })
                          .getTry();

        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "faiss internal error: " << tryObj.exception().what();
            return Status::faiss_inner_error;
        }

        return tryObj.value();
    }

    int64_t
    Size() const override {
        // todo
        return 0;
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        // todo
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

 protected:
    std::shared_ptr<ThreadPool> build_pool;
    std::shared_ptr<ThreadPool> search_pool;

    // train impl
    virtual Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) = 0;

    // add impl
    virtual Status
    AddInternal(const DataSetPtr dataset, const Config& cfg) = 0;
};

//
class BaseFaissRegularIndexNode : public BaseFaissIndexNode {
 public:
    BaseFaissRegularIndexNode(const int32_t& version, const Object& object)
        : BaseFaissIndexNode(version, object), index{nullptr} {
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        if (this->index == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        if (!this->index->is_trained) {
            return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
        }

        auto dim = Dim();
        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();

        try {
            auto data = std::make_unique<float[]>(dim * rows);

            for (int64_t i = 0; i < rows; i++) {
                const int64_t id = ids[i];
                assert(id >= 0 && id < index->ntotal);
                index->reconstruct(id, data.get() + i * dim);
            }

            return GenResultDataSet(rows, dim, std::move(data));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (index == nullptr) {
            return Status::empty_index;
        }

        try {
            MemoryIOWriter writer;
            faiss::write_index(index.get(), &writer);

            std::shared_ptr<uint8_t[]> data(writer.data());
            binset.Append(Type(), data, writer.tellg());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }

        MemoryIOReader reader(binary->data.get(), binary->size);
        try {
            auto read_index = std::unique_ptr<faiss::Index>(faiss::read_index(&reader));
            index.reset(read_index.release());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
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

        try {
            auto read_index = std::unique_ptr<faiss::Index>(faiss::read_index(filename.data(), io_flags));
            index.reset(read_index.release());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    //
    int64_t
    Dim() const override {
        if (index == nullptr) {
            return -1;
        }

        return index->d;
    }

    int64_t
    Count() const override {
        if (index == nullptr) {
            return -1;
        }

        // total number of indexed vectors
        return index->ntotal;
    }

 protected:
    std::unique_ptr<faiss::Index> index;

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (this->index == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto data = dataset->GetTensor();
        auto rows = dataset->GetRows();
        try {
            this->index->add(rows, reinterpret_cast<const float*>(data));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }
};

//
class BaseFaissRegularIndexHNSWNode : public BaseFaissRegularIndexNode {
 public:
    BaseFaissRegularIndexHNSWNode(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexNode(version, object) {
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if (index == nullptr) {
            return false;
        }

        // check whether we use a refined index
        const faiss::IndexRefine* const index_refine = dynamic_cast<const faiss::IndexRefine*>(index.get());
        if (index_refine == nullptr) {
            return false;
        }

        // check whether the refine index is IndexFlat
        // todo: SQfp16 is good for fp16 data type
        // todo: SQbf16 is good for bf16 data type
        const faiss::IndexFlat* const index_refine_flat =
            dynamic_cast<const faiss::IndexFlat*>(index_refine->refine_index);
        if (index_refine_flat == nullptr) {
            // we might be using a different refine index
            return false;
        }

        // yes, we're using IndexRefine with a Flat index
        return true;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (this->index == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        if (!this->index->is_trained) {
            return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
        }

        const auto dim = dataset->GetDim();
        const auto rows = dataset->GetRows();
        const auto* data = dataset->GetTensor();

        const auto hnsw_cfg = static_cast<const FaissHnswConfig&>(cfg);
        const auto k = hnsw_cfg.k.value();
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), knowhere::metric::COSINE);

        const bool whether_bf_search = WhetherPerformBruteForceSearch(hnsw_cfg, bitset);

        feder::hnsw::FederResultUniq feder_result;
        if (hnsw_cfg.trace_visit.value()) {
            if (rows != 1) {
                return expected<DataSetPtr>::Err(Status::invalid_args, "a single query vector is required");
            }
            feder_result = std::make_unique<feder::hnsw::FederResult>();
        }

        auto ids = std::make_unique<faiss::idx_t[]>(rows * k);
        auto distances = std::make_unique<float[]>(rows * k);
        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(rows);
            for (int64_t i = 0; i < rows; ++i) {
                futs.emplace_back(search_pool->push([&, idx = i] {
                    // 1 thread per element
                    ThreadPool::ScopedOmpSetter setter(1);

                    // set up a query
                    const float* cur_query = (const float*)data + idx * dim;

                    // set up local results
                    faiss::idx_t* const __restrict local_ids = ids.get() + k * idx;
                    float* const __restrict local_distances = distances.get() + k * idx;

                    // set up faiss search parameters
                    knowhere::SearchParametersHNSWWrapper hnsw_search_params;
                    if (hnsw_cfg.ef.has_value()) {
                        hnsw_search_params.efSearch = hnsw_cfg.ef.value();
                    }
                    // do not collect HNSW stats
                    hnsw_search_params.hnsw_stats = nullptr;
                    // set up feder
                    hnsw_search_params.feder = feder_result.get();
                    // set up kAlpha
                    hnsw_search_params.kAlpha = bitset.filter_ratio() * 0.7f;

                    // set up a selector
                    BitsetViewIDSelector bw_idselector(bitset);
                    faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;

                    hnsw_search_params.sel = id_selector;

                    // use knowhere-based search by default
                    const bool override_faiss_search = hnsw_cfg.override_faiss_search.value_or(true);

                    // check if we have a refine available.
                    faiss::IndexRefine* const index_refine = dynamic_cast<faiss::IndexRefine*>(index.get());

                    if (index_refine != nullptr) {
                        // yes, it is possible to refine results.

                        // cast a base index to IndexHNSW-based index
                        faiss::IndexHNSW* const index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index_refine->base_index);

                        if (index_hnsw == nullptr) {
                            // this is unexpected
                            throw std::runtime_error("Expecting faiss::IndexHNSW");
                        }

                        // pick a wrapper for hnsw which does not own indices
                        knowhere::IndexHNSWWrapper wrapper_hnsw_search(index_hnsw);
                        knowhere::IndexBruteForceWrapper wrapper_bf(index_hnsw);

                        faiss::Index* base_wrapper = nullptr;
                        if (!override_faiss_search) {
                            // use the original index, no wrappers
                            base_wrapper = index_hnsw;
                        } else if (whether_bf_search) {
                            // use brute-force wrapper
                            base_wrapper = &wrapper_bf;
                        } else {
                            // use hnsw-search wrapper
                            base_wrapper = &wrapper_hnsw_search;
                        }

                        // check if used wants a refined result
                        if (hnsw_cfg.refine_k.has_value()) {
                            // yes, a user wants to perform a refine

                            // set up search parameters
                            faiss::IndexRefineSearchParameters refine_params;
                            refine_params.k_factor = hnsw_cfg.refine_k.value();
                            // a refine procedure itself does not need to care about filtering
                            refine_params.sel = nullptr;
                            refine_params.base_index_params = &hnsw_search_params;

                            // is it a cosine index?
                            if (index_hnsw->storage->is_cosine && is_cosine) {
                                // yes, wrap both base and refine index
                                knowhere::IndexWrapperCosine cosine_wrapper(
                                    index_refine->refine_index,
                                    dynamic_cast<faiss::HasInverseL2Norms*>(index_hnsw)->get_inverse_l2_norms());

                                // create a temporary refine index which does not own
                                faiss::IndexRefine tmp_refine(base_wrapper, &cosine_wrapper);

                                // perform a search
                                tmp_refine.search(1, cur_query, k, local_distances, local_ids, &refine_params);
                            } else {
                                // no, wrap base index only.

                                // create a temporary refine index which does not own
                                faiss::IndexRefine tmp_refine(base_wrapper, index_refine->refine_index);

                                // perform a search
                                tmp_refine.search(1, cur_query, k, local_distances, local_ids, &refine_params);
                            }
                        } else {
                            // no, a user wants to skip a refine

                            // perform a search
                            base_wrapper->search(1, cur_query, k, local_distances, local_ids, &hnsw_search_params);
                        }
                    } else {
                        // there's no refining available

                        // check if refine is required
                        if (hnsw_cfg.refine_k.has_value()) {
                            // this is not possible, throw an error
                            throw std::runtime_error("Refine is not provided by the index.");
                        }

                        // cast to IndexHNSW-based index
                        faiss::IndexHNSW* const index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index.get());

                        if (index_hnsw == nullptr) {
                            // this is unexpected
                            throw std::runtime_error("Expecting faiss::IndexHNSW");
                        }

                        // pick a wrapper for hnsw which does not own indices
                        knowhere::IndexHNSWWrapper wrapper_hnsw_search(index_hnsw);
                        knowhere::IndexBruteForceWrapper wrapper_bf(index_hnsw);

                        faiss::Index* wrapper = nullptr;
                        if (!override_faiss_search) {
                            // use the original index, no wrappers
                            wrapper = index_hnsw;
                        } else if (whether_bf_search) {
                            // use brute-force wrapper
                            wrapper = &wrapper_bf;
                        } else {
                            // use hnsw-search wrapper
                            wrapper = &wrapper_hnsw_search;
                        }

                        // perform a search
                        wrapper->search(1, cur_query, k, local_distances, local_ids, &hnsw_search_params);
                    }
                }));
            }

            // wait for the completion
            WaitAllSuccess(futs);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        auto res = GenResultDataSet(rows, k, std::move(ids), std::move(distances));

        // set visit_info json string into result dataset
        if (feder_result != nullptr) {
            Json json_visit_info, json_id_set;
            nlohmann::to_json(json_visit_info, feder_result->visit_info_);
            nlohmann::to_json(json_id_set, feder_result->id_set_);
            res->SetJsonInfo(json_visit_info.dump());
            res->SetJsonIdSet(json_id_set.dump());
        }

        return res;
    }

 protected:
    // Decides whether a brute force should be used instead of a regular HNSW search.
    // This may be applicable in case of very large topk values or
    //   extremely high filtering levels.
    bool
    WhetherPerformBruteForceSearch(const BaseConfig& cfg, const BitsetView& bitset) const {
        constexpr float kHnswSearchKnnBFFilterThreshold = 0.93f;
        constexpr float kHnswSearchRangeBFFilterThreshold = 0.97f;
        constexpr float kHnswSearchBFTopkThreshold = 0.5f;

        auto k = cfg.k.value();

        if (k >= (index->ntotal * kHnswSearchBFTopkThreshold)) {
            return true;
        }

        if (!bitset.empty()) {
            const size_t filtered_out_num = bitset.count();
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            double ratio = ((double)filtered_out_num) / bitset.size();
            knowhere::knowhere_hnsw_bitset_ratio.Observe(ratio);
#endif
            if (filtered_out_num >= (index->ntotal * kHnswSearchKnnBFFilterThreshold) ||
                k >= (index->ntotal - filtered_out_num) * kHnswSearchBFTopkThreshold) {
                return true;
            }
        }

        // the default value
        return false;
    }

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (this->index == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto data = dataset->GetTensor();
        auto rows = dataset->GetRows();
        try {
            this->index->add(rows, reinterpret_cast<const float*>(data));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }
};

//
template <typename DataType>
class BaseFaissRegularIndexHNSWFlatNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWFlatNode(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWNode(version, object) {
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if (index == nullptr) {
            return false;
        }

        // yes, a flat index has it
        return true;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<FaissHnswFlatConfig>();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FAISS_HNSW_FLAT;
    }

 protected:
    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();
        // data
        auto data = dataset->GetTensor();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswFlatConfig&>(cfg);

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);

        std::unique_ptr<faiss::IndexHNSW> hnsw_index;
        if (is_cosine) {
            hnsw_index = std::make_unique<faiss::IndexHNSWFlatCosine>(dim, hnsw_cfg.M.value());
        } else {
            hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(dim, hnsw_cfg.M.value(), metric.value());
        }

        hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();

        // train
        hnsw_index->train(rows, (const float*)data);

        // done
        index = std::move(hnsw_index);
        return Status::success;
    }
};

KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_FLAT, BaseFaissRegularIndexHNSWFlatNode, fp32);

}  // namespace knowhere
