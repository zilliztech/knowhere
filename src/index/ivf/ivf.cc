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
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexFlatElkan.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexIVFPQFastScan.h"
#include "faiss/IndexScaNN.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/index_io.h"
#include "index/ivf/ivf_config.h"
#include "io/memory_io.h"
#include "io/trailer.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/factory.h"
#include "knowhere/feder/IVFFlat.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

template <typename T>
class IvfIndexNode : public IndexNode {
 public:
    IvfIndexNode(const int32_t version, const Object& object) : IndexNode(version), index_(nullptr) {
        static_assert(std::is_same<T, faiss::IndexIVFFlat>::value || std::is_same<T, faiss::IndexIVFFlatCC>::value ||
                          std::is_same<T, faiss::IndexIVFPQ>::value ||
                          std::is_same<T, faiss::IndexIVFScalarQuantizer>::value ||
                          std::is_same<T, faiss::IndexBinaryIVF>::value || std::is_same<T, faiss::IndexScaNN>::value,
                      "not support");
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
    }
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
    HasRawData(const std::string& metric_type) const override {
        if (!index_) {
            return false;
        }
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            return true;
        }
        if constexpr (std::is_same<faiss::IndexIVFFlatCC, T>::value) {
            return true;
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            return false;
        }
        if constexpr (std::is_same<faiss::IndexScaNN, T>::value) {
            return index_->with_raw_data();
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            return false;
        }
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            return true;
        }
    }
    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }
    Status
    Serialize(BinarySet& binset) const override;
    Status
    Deserialize(const BinarySet& binset, const Config& config) override;
    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override;
    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            return std::make_unique<IvfFlatConfig>();
        }
        if constexpr (std::is_same<faiss::IndexIVFFlatCC, T>::value) {
            return std::make_unique<IvfFlatCcConfig>();
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            return std::make_unique<IvfPqConfig>();
        }
        if constexpr (std::is_same<faiss::IndexScaNN, T>::value) {
            return std::make_unique<ScannConfig>();
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            return std::make_unique<IvfSqConfig>();
        }
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            return std::make_unique<IvfBinConfig>();
        }
    };
    int64_t
    Dim() const override {
        if (!index_) {
            return -1;
        }
        return index_->d;
    };
    int64_t
    Size() const override {
        if (!index_) {
            return 0;
        }
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto nlist = index_->nlist;
            auto code_size = index_->code_size;
            return ((nb + nlist) * (code_size + sizeof(int64_t)));
        }
        if constexpr (std::is_same<T, faiss::IndexIVFFlatCC>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto nlist = index_->nlist;
            auto code_size = index_->code_size;
            return (nb * code_size + nb * sizeof(int64_t) + nlist * code_size);
        }
        if constexpr (std::is_same<T, faiss::IndexIVFPQ>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto code_size = index_->code_size;
            auto pq = index_->pq;
            auto nlist = index_->nlist;
            auto d = index_->d;

            auto capacity = nb * code_size + nb * sizeof(int64_t) + nlist * d * sizeof(float);
            auto centroid_table = pq.M * pq.ksub * pq.dsub * sizeof(float);
            auto precomputed_table = nlist * pq.M * pq.ksub * sizeof(float);
            return (capacity + centroid_table + precomputed_table);
        }
        if constexpr (std::is_same<T, faiss::IndexScaNN>::value) {
            return index_->size();
        }
        if constexpr (std::is_same<T, faiss::IndexIVFScalarQuantizer>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto code_size = index_->code_size;
            auto nlist = index_->nlist;
            return (nb * code_size + nb * sizeof(int64_t) + 2 * code_size + nlist * code_size);
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            auto nb = index_->invlists->compute_ntotal();
            auto nlist = index_->nlist;
            auto code_size = index_->code_size;
            return (nb * code_size + nb * sizeof(int64_t) + nlist * code_size);
        }
    };
    int64_t
    Count() const override {
        if (!index_) {
            return 0;
        }
        return index_->ntotal;
    };
    std::string
    Type() const override {
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
        }
        if constexpr (std::is_same<T, faiss::IndexIVFFlatCC>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC;
        }
        if constexpr (std::is_same<T, faiss::IndexIVFPQ>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IVFPQ;
        }
        if constexpr (std::is_same<T, faiss::IndexScaNN>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_SCANN;
        }
        if constexpr (std::is_same<T, faiss::IndexIVFScalarQuantizer>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;
        }
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            return knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT;
        }
    };

 private:
    std::unique_ptr<T> index_;
    std::shared_ptr<ThreadPool> search_pool_;
};

}  // namespace knowhere

namespace knowhere {

inline int64_t
MatchNlist(int64_t size, int64_t nlist) {
    const int64_t MIN_POINTS_PER_CENTROID = 39;

    if (nlist * MIN_POINTS_PER_CENTROID > size) {
        // nlist is too large, adjust to a proper value
        LOG_KNOWHERE_WARNING_ << "nlist(" << nlist << ") is too large, adjust to a proper value";
        nlist = std::max(static_cast<int64_t>(1), size / MIN_POINTS_PER_CENTROID);
        LOG_KNOWHERE_WARNING_ << "Row num " << size << " match nlist " << nlist;
    }
    return nlist;
}

int64_t
MatchNbits(int64_t size, int64_t nbits) {
    if (size < (1 << nbits)) {
        // nbits is too large, adjust to a proper value
        LOG_KNOWHERE_WARNING_ << "nbits(" << nbits << ") is too large, adjust to a proper value";
        if (size >= (1 << 8)) {
            nbits = 8;
        } else if (size >= (1 << 4)) {
            nbits = 4;
        } else if (size >= (1 << 2)) {
            nbits = 2;
        } else {
            nbits = 1;
        }
        LOG_KNOWHERE_WARNING_ << "Row num " << size << " match nbits " << nbits;
    }
    return nbits;
}

namespace {

// turn IndexFlatElkan into IndexFlat
std::unique_ptr<faiss::IndexFlat>
to_index_flat(std::unique_ptr<faiss::IndexFlat>&& index) {
    // C++ slicing here
    return std::make_unique<faiss::IndexFlat>(std::move(*index));
}

}  // namespace

template <typename T>
Status
IvfIndexNode<T>::Train(const DataSet& dataset, const Config& cfg) {
    const BaseConfig& base_cfg = static_cast<const IvfConfig&>(cfg);
    std::unique_ptr<ThreadPool::ScopedOmpSetter> setter;
    if (base_cfg.num_build_thread.has_value()) {
        setter = std::make_unique<ThreadPool::ScopedOmpSetter>(base_cfg.num_build_thread.value());
    } else {
        setter = std::make_unique<ThreadPool::ScopedOmpSetter>();
    }

    bool is_cosine = IsMetricType(base_cfg.metric_type.value(), knowhere::metric::COSINE);

    // do normalize for COSINE metric type
    if constexpr (std::is_same_v<faiss::IndexIVFPQ, T> || std::is_same_v<faiss::IndexIVFScalarQuantizer, T>) {
        if (is_cosine) {
            Normalize(dataset);
        }
    }

    auto metric = Str2FaissMetricType(base_cfg.metric_type.value());
    if (!metric.has_value()) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << base_cfg.metric_type.value();
        return Status::invalid_metric_type;
    }

    auto rows = dataset.GetRows();
    auto dim = dataset.GetDim();
    auto data = dataset.GetTensor();

    // faiss scann needs at least 16 rows since nbits=4
    constexpr int64_t SCANN_MIN_ROWS = 16;
    if constexpr (std::is_same<faiss::IndexScaNN, T>::value) {
        if (rows < SCANN_MIN_ROWS) {
            LOG_KNOWHERE_ERROR_ << rows << " rows is not enough, scann needs at least 16 rows to build index";
            return Status::faiss_inner_error;
        }
    }

    std::unique_ptr<T> index;
    // if cfg.use_elkan is used, then we'll use a temporary instance of
    //  IndexFlatElkan for the training.
    try {
        if constexpr (std::is_same<faiss::IndexIVFFlat, T>::value) {
            const IvfFlatConfig& ivf_flat_cfg = static_cast<const IvfFlatConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_flat_cfg.nlist.value());

            const bool use_elkan = ivf_flat_cfg.use_elkan.value_or(true);

            // create quantizer for the training
            std::unique_ptr<faiss::IndexFlat> qzr =
                std::make_unique<faiss::IndexFlatElkan>(dim, metric.value(), false, use_elkan);
            // create index. Index does not own qzr
            index = std::make_unique<faiss::IndexIVFFlat>(qzr.get(), dim, nlist, metric.value(), is_cosine);
            // train
            index->train(rows, (const float*)data);
            // replace quantizer with a regular IndexFlat
            qzr = to_index_flat(std::move(qzr));
            index->quantizer = qzr.get();
            // transfer ownership of qzr to index
            qzr.release();
            index->own_fields = true;
        }
        if constexpr (std::is_same<faiss::IndexIVFFlatCC, T>::value) {
            const IvfFlatCcConfig& ivf_flat_cc_cfg = static_cast<const IvfFlatCcConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_flat_cc_cfg.nlist.value());

            const bool use_elkan = ivf_flat_cc_cfg.use_elkan.value_or(true);

            // create quantizer for the training
            std::unique_ptr<faiss::IndexFlat> qzr =
                std::make_unique<faiss::IndexFlatElkan>(dim, metric.value(), false, use_elkan);
            // create index. Index does not own qzr
            index = std::make_unique<faiss::IndexIVFFlatCC>(qzr.get(), dim, nlist, ivf_flat_cc_cfg.ssize.value(),
                                                            metric.value(), is_cosine);
            // train
            index->train(rows, (const float*)data);
            // replace quantizer with a regular IndexFlat
            qzr = to_index_flat(std::move(qzr));
            index->quantizer = qzr.get();
            // transfer ownership of qzr to index
            qzr.release();
            index->own_fields = true;
            // ivfflat_cc has no serialize stage, make map at build stage
            index->make_direct_map(true, faiss::DirectMap::ConcurrentArray);
        }
        if constexpr (std::is_same<faiss::IndexIVFPQ, T>::value) {
            const IvfPqConfig& ivf_pq_cfg = static_cast<const IvfPqConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_pq_cfg.nlist.value());
            auto nbits = MatchNbits(rows, ivf_pq_cfg.nbits.value());

            const bool use_elkan = ivf_pq_cfg.use_elkan.value_or(true);

            // create quantizer for the training
            std::unique_ptr<faiss::IndexFlat> qzr =
                std::make_unique<faiss::IndexFlatElkan>(dim, metric.value(), false, use_elkan);
            // create index. Index does not own qzr
            index =
                std::make_unique<faiss::IndexIVFPQ>(qzr.get(), dim, nlist, ivf_pq_cfg.m.value(), nbits, metric.value());
            // train
            index->train(rows, (const float*)data);
            // replace quantizer with a regular IndexFlat
            qzr = to_index_flat(std::move(qzr));
            index->quantizer = qzr.get();
            // transfer ownership of qzr to index
            qzr.release();
            index->own_fields = true;
        }
        if constexpr (std::is_same<faiss::IndexScaNN, T>::value) {
            const ScannConfig& scann_cfg = static_cast<const ScannConfig&>(cfg);
            auto nlist = MatchNlist(rows, scann_cfg.nlist.value());
            bool is_cosine = base_cfg.metric_type.value() == metric::COSINE;

            const bool use_elkan = scann_cfg.use_elkan.value_or(true);

            // create quantizer for the training
            std::unique_ptr<faiss::IndexFlat> qzr =
                std::make_unique<faiss::IndexFlatElkan>(dim, metric.value(), false, use_elkan);
            // create base index. it does not own qzr
            auto base_index = std::make_unique<faiss::IndexIVFPQFastScan>(qzr.get(), dim, nlist, (dim + 1) / 2, 4,
                                                                          is_cosine, metric.value());
            // create scann index, which does not base_index by default,
            //    but owns the refine index by default omg
            if (scann_cfg.with_raw_data.value()) {
                index = std::make_unique<faiss::IndexScaNN>(base_index.get(), (const float*)data);
            } else {
                index = std::make_unique<faiss::IndexScaNN>(base_index.get(), nullptr);
            }
            // train
            index->train(rows, (const float*)data);
            // at this moment, we still own qzr.
            // replace quantizer with a regular IndexFlat
            qzr = to_index_flat(std::move(qzr));
            base_index->quantizer = qzr.get();
            // release qzr
            qzr.release();
            base_index->own_fields = true;
            // transfer ownership of the base index
            base_index.release();
            index->own_fields = true;
        }
        if constexpr (std::is_same<faiss::IndexIVFScalarQuantizer, T>::value) {
            const IvfSqConfig& ivf_sq_cfg = static_cast<const IvfSqConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_sq_cfg.nlist.value());

            const bool use_elkan = ivf_sq_cfg.use_elkan.value_or(true);

            // create quantizer for the training
            std::unique_ptr<faiss::IndexFlat> qzr =
                std::make_unique<faiss::IndexFlatElkan>(dim, metric.value(), false, use_elkan);
            // create index. Index does not own qzr
            index = std::make_unique<faiss::IndexIVFScalarQuantizer>(
                qzr.get(), dim, nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, metric.value());
            // train
            index->train(rows, (const float*)data);
            // replace quantizer with a regular IndexFlat
            qzr = to_index_flat(std::move(qzr));
            index->quantizer = qzr.get();
            // transfer ownership of qzr to index
            qzr.release();
            index->own_fields = true;
        }
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            const IvfBinConfig& ivf_bin_cfg = static_cast<const IvfBinConfig&>(cfg);
            auto nlist = MatchNlist(rows, ivf_bin_cfg.nlist.value());

            // create quantizer
            auto qzr = std::make_unique<faiss::IndexBinaryFlat>(dim, metric.value());
            // create index. Index does not own qzr
            index = std::make_unique<faiss::IndexBinaryIVF>(qzr.get(), dim, nlist, metric.value());
            // train
            index->train(rows, (const uint8_t*)data);
            // transfer ownership of qzr to index
            qzr.release();
            index->own_fields = true;
        }
    } catch (std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    index_ = std::move(index);

    return Status::success;
}

template <typename T>
Status
IvfIndexNode<T>::Add(const DataSet& dataset, const Config& cfg) {
    if (!this->index_) {
        LOG_KNOWHERE_ERROR_ << "Can not add data to empty IVF index.";
        return Status::empty_index;
    }
    auto data = dataset.GetTensor();
    auto rows = dataset.GetRows();
    const BaseConfig& base_cfg = static_cast<const IvfConfig&>(cfg);
    std::unique_ptr<ThreadPool::ScopedOmpSetter> setter;
    if (base_cfg.num_build_thread.has_value()) {
        setter = std::make_unique<ThreadPool::ScopedOmpSetter>(base_cfg.num_build_thread.value());
    } else {
        setter = std::make_unique<ThreadPool::ScopedOmpSetter>();
    }
    try {
        if constexpr (std::is_same<faiss::IndexBinaryIVF, T>::value) {
            index_->add(rows, (const uint8_t*)data);
        } else {
            index_->add(rows, (const float*)data);
        }
    } catch (std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

template <typename T>
expected<DataSetPtr>
IvfIndexNode<T>::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
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

    const IvfConfig& ivf_cfg = static_cast<const IvfConfig&>(cfg);
    bool is_cosine = IsMetricType(ivf_cfg.metric_type.value(), knowhere::metric::COSINE);

    auto k = ivf_cfg.k.value();
    auto nprobe = ivf_cfg.nprobe.value();

    int64_t* ids(new (std::nothrow) int64_t[rows * k]);
    float* distances(new (std::nothrow) float[rows * k]);
    int32_t* i_distances = reinterpret_cast<int32_t*>(distances);
    try {
        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(rows);
        for (int i = 0; i < rows; ++i) {
            futs.emplace_back(search_pool_->push([&, index = i] {
                ThreadPool::ScopedOmpSetter setter(1);
                auto offset = k * index;
                std::unique_ptr<float[]> copied_query = nullptr;

                BitsetViewIDSelector bw_idselector(bitset);
                faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;

                if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
                    auto cur_data = (const uint8_t*)data + index * dim / 8;

                    faiss::IVFSearchParameters ivf_search_params;
                    ivf_search_params.nprobe = nprobe;
                    ivf_search_params.sel = id_selector;
                    index_->search(1, cur_data, k, i_distances + offset, ids + offset, &ivf_search_params);

                    if (index_->metric_type == faiss::METRIC_Hamming) {
                        for (int64_t i = 0; i < k; i++) {
                            distances[i + offset] = static_cast<float>(i_distances[i + offset]);
                        }
                    }
                } else if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
                    auto cur_query = (const float*)data + index * dim;
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    faiss::IVFSearchParameters ivf_search_params;
                    ivf_search_params.nprobe = nprobe;
                    ivf_search_params.max_codes = 0;
                    ivf_search_params.sel = id_selector;

                    index_->search(1, cur_query, k, distances + offset, ids + offset, &ivf_search_params);
                } else if constexpr (std::is_same<T, faiss::IndexScaNN>::value) {
                    auto cur_query = (const float*)data + index * dim;
                    const ScannConfig& scann_cfg = static_cast<const ScannConfig&>(cfg);
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    // todo aguzhva: this is somewhat alogical. Refactor?
                    faiss::IVFSearchParameters base_search_params;
                    base_search_params.sel = id_selector;
                    base_search_params.nprobe = nprobe;

                    faiss::IndexScaNNSearchParameters scann_search_params;
                    scann_search_params.base_index_params = &base_search_params;
                    scann_search_params.reorder_k = scann_cfg.reorder_k.value();

                    index_->search(1, cur_query, k, distances + offset, ids + offset, &scann_search_params);
                } else {
                    auto cur_query = (const float*)data + index * dim;
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    faiss::IVFSearchParameters ivf_search_params;
                    ivf_search_params.nprobe = nprobe;
                    ivf_search_params.max_codes = 0;
                    ivf_search_params.sel = id_selector;

                    index_->search(1, cur_query, k, distances + offset, ids + offset, &ivf_search_params);
                }
            }));
        }
        // wait for the completion
        for (auto& fut : futs) {
            fut.wait();
        }
        // check for exceptions. value() is {}, so either
        //   a call does nothing, or it throws an inner exception.
        for (auto& fut : futs) {
            fut.result().value();
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

template <typename T>
expected<DataSetPtr>
IvfIndexNode<T>::RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
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

    const IvfConfig& ivf_cfg = static_cast<const IvfConfig&>(cfg);
    bool is_cosine = IsMetricType(ivf_cfg.metric_type.value(), knowhere::metric::COSINE);

    float radius = ivf_cfg.radius.value();
    float range_filter = ivf_cfg.range_filter.value();
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

                BitsetViewIDSelector bw_idselector(bitset);
                faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;

                if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
                    auto cur_data = (const uint8_t*)xq + index * dim / 8;

                    faiss::IVFSearchParameters ivf_search_params;
                    ivf_search_params.nprobe = index_->nlist;
                    ivf_search_params.sel = id_selector;

                    index_->range_search(1, cur_data, radius, &res, &ivf_search_params);
                } else if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
                    auto cur_query = (const float*)xq + index * dim;
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    faiss::IVFSearchParameters ivf_search_params;
                    ivf_search_params.nprobe = index_->nlist;
                    ivf_search_params.max_codes = 0;
                    ivf_search_params.sel = id_selector;

                    index_->range_search(1, cur_query, radius, &res, &ivf_search_params);
                } else if constexpr (std::is_same<T, faiss::IndexScaNN>::value) {
                    auto cur_query = (const float*)xq + index * dim;
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    // todo aguzhva: this is somewhat alogical. Refactor?
                    faiss::IVFSearchParameters base_search_params;
                    base_search_params.sel = id_selector;

                    faiss::IndexScaNNSearchParameters scann_search_params;
                    scann_search_params.base_index_params = &base_search_params;

                    index_->range_search(1, cur_query, radius, &res, &scann_search_params);
                } else {
                    auto cur_query = (const float*)xq + index * dim;
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    faiss::IVFSearchParameters ivf_search_params;
                    ivf_search_params.nprobe = index_->nlist;
                    ivf_search_params.max_codes = 0;
                    ivf_search_params.sel = id_selector;

                    index_->range_search(1, cur_query, radius, &res, &ivf_search_params);
                }
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
        // wait for the completion
        for (auto& fut : futs) {
            fut.wait();
        }
        // check for exceptions. value() is {}, so either
        //   a call does nothing, or it throws an inner exception.
        for (auto& fut : futs) {
            fut.result().value();
        }
        GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius, range_filter, distances, ids, lims);
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
    }

    return GenResultDataSet(nq, ids, distances, lims);
}

template <typename T>
expected<DataSetPtr>
IvfIndexNode<T>::GetVectorByIds(const DataSet& dataset) const {
    if (!this->index_) {
        return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
    }
    if (!this->index_->is_trained) {
        return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
    }
    if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
        auto dim = Dim();
        auto rows = dataset.GetRows();
        auto ids = dataset.GetIds();

        uint8_t* data = nullptr;
        try {
            data = new uint8_t[dim * rows / 8];
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < index_->ntotal);
                index_->reconstruct(id, data + i * dim / 8);
            }
            return GenResultDataSet(rows, dim, data);
        } catch (const std::exception& e) {
            std::unique_ptr<uint8_t[]> auto_del(data);
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
    } else if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value || std::is_same<T, faiss::IndexIVFFlatCC>::value) {
        auto dim = Dim();
        auto rows = dataset.GetRows();
        auto ids = dataset.GetIds();

        float* data = nullptr;
        try {
            data = new float[dim * rows];
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < index_->ntotal);
                index_->reconstruct(id, data + i * dim);
            }
            return GenResultDataSet(rows, dim, data);
        } catch (const std::exception& e) {
            std::unique_ptr<float[]> auto_del(data);
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
    } else if constexpr (std::is_same<T, faiss::IndexScaNN>::value) {
        // we should never go here since we should call HasRawData() first
        if (!index_->with_raw_data()) {
            return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
        }
        auto dim = Dim();
        auto rows = dataset.GetRows();
        auto ids = dataset.GetIds();

        float* data = nullptr;
        try {
            data = new float[dim * rows];
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < index_->ntotal);
                index_->reconstruct(id, data + i * dim);
            }
            return GenResultDataSet(rows, dim, data);
        } catch (const std::exception& e) {
            std::unique_ptr<float[]> auto_del(data);
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
    } else {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
    }
}

template <>
expected<DataSetPtr>
IvfIndexNode<faiss::IndexIVFFlat>::GetIndexMeta(const Config& config) const {
    if (!index_) {
        LOG_KNOWHERE_WARNING_ << "get index meta on empty index";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
    }

    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    auto ivf_quantizer = dynamic_cast<faiss::IndexFlat*>(ivf_index->quantizer);

    int64_t dim = ivf_index->d;
    int64_t nlist = ivf_index->nlist;
    int64_t ntotal = ivf_index->ntotal;

    feder::ivfflat::IVFFlatMeta meta(nlist, dim, ntotal);
    std::unordered_set<int64_t> id_set;

    for (int32_t i = 0; i < nlist; i++) {
        // copy from IndexIVF::search_preassigned
        std::unique_ptr<faiss::InvertedLists::ScopedIds> sids =
            std::make_unique<faiss::InvertedLists::ScopedIds>(index_->invlists, i);

        // node ids
        auto node_num = index_->invlists->list_size(i);
        auto node_id_codes = sids->get();

        // centroid vector
        auto centroid_vec = ivf_quantizer->get_xb() + i * dim;

        meta.AddCluster(i, node_id_codes, node_num, centroid_vec, dim);
    }

    Json json_meta, json_id_set;
    nlohmann::to_json(json_meta, meta);
    nlohmann::to_json(json_id_set, id_set);
    return GenResultDataSet(json_meta.dump(), json_id_set.dump());
}

template <typename T>
Status
IvfIndexNode<T>::Serialize(BinarySet& binset) const {
    try {
        MemoryIOWriter writer;
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            faiss::write_index_binary(index_.get(), &writer);
        } else {
            faiss::write_index(index_.get(), &writer);
        }
        auto trailer_status = AddTrailerForMemoryIO(writer, Type(), this->version_);
        if (trailer_status != Status::success) {
            LOG_KNOWHERE_ERROR_ << "fail to append trailer.";
            return trailer_status;
        }
        std::shared_ptr<uint8_t[]> data(writer.data());
        binset.Append(Type(), data, writer.tellg());
        return Status::success;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
}

template <>
Status
IvfIndexNode<faiss::IndexIVFFlat>::Serialize(BinarySet& binset) const {
    try {
        MemoryIOWriter writer;
        LOG_KNOWHERE_INFO_ << "request version " << version_.VersionNumber();
        if (version_ <= Version::GetMinimalVersion()) {
            faiss::write_index_nm(index_.get(), &writer);
            LOG_KNOWHERE_INFO_ << "write IVF_FLAT_NM, file size " << writer.tellg();
        } else {
            faiss::write_index(index_.get(), &writer);
            LOG_KNOWHERE_INFO_ << "write IVF_FLAT, file size " << writer.tellg();
        }
        auto trailer_status = AddTrailerForMemoryIO(writer, Type(), this->version_);
        if (trailer_status != Status::success) {
            LOG_KNOWHERE_ERROR_ << "fail to append trailer.";
            return trailer_status;
        }
        std::shared_ptr<uint8_t[]> index_data_ptr(writer.data());
        binset.Append(Type(), index_data_ptr, writer.tellg());

        // append raw data for backward compatible
        if (version_ <= Version::GetMinimalVersion()) {
            size_t dim = index_->d;
            size_t rows = index_->ntotal;
            size_t raw_data_size = dim * rows * sizeof(float);
            uint8_t* raw_data = new uint8_t[raw_data_size];
            std::shared_ptr<uint8_t[]> raw_data_ptr(raw_data);
            for (size_t i = 0; i < index_->nlist; i++) {
                size_t list_size = index_->invlists->list_size(i);
                const faiss::idx_t* ids = index_->invlists->get_ids(i);
                const uint8_t* codes = index_->invlists->get_codes(i);
                for (size_t j = 0; j < list_size; j++) {
                    faiss::idx_t id = ids[j];
                    const uint8_t* src = codes + j * dim * sizeof(float);
                    uint8_t* dst = raw_data + id * dim * sizeof(float);
                    memcpy(dst, src, dim * sizeof(float));
                }
            }
            binset.Append("RAW_DATA", raw_data_ptr, raw_data_size);
            LOG_KNOWHERE_INFO_ << "append raw data for IVF_FLAT_NM, size " << raw_data_size;
        }
        return Status::success;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
}

template <typename T>
Status
IvfIndexNode<T>::Deserialize(const BinarySet& binset, const Config& config) {
    std::vector<std::string> names = {"IVF",        // compatible with knowhere-1.x
                                      "BinaryIVF",  // compatible with knowhere-1.x
                                      Type()};
    auto binary = binset.GetByNames(names);
    if (binary == nullptr) {
        LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
        return Status::invalid_binary_set;
    }

    MemoryIOReader reader(binary->data.get(), binary->size);
    try {
        if constexpr (std::is_same<T, faiss::IndexIVFFlat>::value) {
            if (version_ <= Version::GetMinimalVersion()) {
                auto raw_binary = binset.GetByName("RAW_DATA");
                const BaseConfig& base_cfg = static_cast<const BaseConfig&>(config);
                ConvertIVFFlat(binset, base_cfg.metric_type.value(), raw_binary->data.get(), raw_binary->size);
                // after conversion, binary size and data will be updated
                reader.data_ = binary->data.get();
                reader.total_ = binary->size;
            }
            index_.reset(static_cast<faiss::IndexIVFFlat*>(faiss::read_index(&reader)));
        } else if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            index_.reset(static_cast<T*>(faiss::read_index_binary(&reader)));
        } else {
            index_.reset(static_cast<T*>(faiss::read_index(&reader)));
        }
        if constexpr (!std::is_same_v<T, faiss::IndexScaNN>) {
            const BaseConfig& base_cfg = static_cast<const BaseConfig&>(config);
            if (HasRawData(base_cfg.metric_type.value())) {
                index_->make_direct_map(true);
            }
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

template <typename T>
Status
IvfIndexNode<T>::DeserializeFromFile(const std::string& filename, const Config& config) {
    auto cfg = static_cast<const knowhere::BaseConfig&>(config);

    int io_flags = 0;
    if (cfg.enable_mmap.value()) {
        io_flags |= faiss::IO_FLAG_MMAP;
    }
    try {
        if constexpr (std::is_same<T, faiss::IndexBinaryIVF>::value) {
            index_.reset(static_cast<T*>(faiss::read_index_binary(filename.data(), io_flags)));
        } else {
            index_.reset(static_cast<T*>(faiss::read_index(filename.data(), io_flags)));
        }
        if constexpr (!std::is_same_v<T, faiss::IndexScaNN>) {
            const BaseConfig& base_cfg = static_cast<const BaseConfig&>(config);
            if (HasRawData(base_cfg.metric_type.value())) {
                index_->make_direct_map(true);
            }
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return Status::faiss_inner_error;
    }
    return Status::success;
}

KNOWHERE_REGISTER_GLOBAL(IVFBIN, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexBinaryIVF>>::Create(version, object);
});

KNOWHERE_REGISTER_GLOBAL(BIN_IVF_FLAT, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexBinaryIVF>>::Create(version, object);
});

KNOWHERE_REGISTER_GLOBAL(IVFFLAT, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFFlat>>::Create(version, object);
});
KNOWHERE_REGISTER_GLOBAL(IVF_FLAT, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFFlat>>::Create(version, object);
});
KNOWHERE_REGISTER_GLOBAL(IVFFLATCC, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFFlatCC>>::Create(version, object);
});
KNOWHERE_REGISTER_GLOBAL(IVF_FLAT_CC, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFFlatCC>>::Create(version, object);
});
KNOWHERE_REGISTER_GLOBAL(SCANN, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexScaNN>>::Create(version, object);
});
KNOWHERE_REGISTER_GLOBAL(IVFPQ, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFPQ>>::Create(version, object);
});
KNOWHERE_REGISTER_GLOBAL(IVF_PQ, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFPQ>>::Create(version, object);
});

KNOWHERE_REGISTER_GLOBAL(IVFSQ, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFScalarQuantizer>>::Create(version, object);
});
KNOWHERE_REGISTER_GLOBAL(IVF_SQ8, [](const int32_t& version, const Object& object) {
    return Index<IvfIndexNode<faiss::IndexIVFScalarQuantizer>>::Create(version, object);
});

}  // namespace knowhere
