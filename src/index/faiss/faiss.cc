// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions
// and limitations under the License.

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/IndexFlat.h>
#include <faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/impl/mapped_io.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#include <cstring>

#include "common/metric.h"
#include "index/faiss/faiss_config.h"
#include "index/faiss/faiss_dispatch.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/operands.h"
#include "knowhere/range_util.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"

namespace knowhere {

namespace {
// Backing faiss base type per DataType.
template <typename DataType>
using FaissBase = std::conditional_t<std::is_same_v<DataType, fp32>, faiss::Index, faiss::IndexBinary>;
}  // namespace

template <typename DataType>
class FaissIndexNode : public IndexNode {
 public:
    static_assert(std::is_same_v<DataType, fp32> || std::is_same_v<DataType, bin1>,
                  "FaissIndexNode supports only fp32 and bin1");

    FaissIndexNode(const int32_t version, const Object& /*object*/) : IndexNode(version) {
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool /*use_knowhere_build_pool*/) override {
        const auto* fc = static_cast<const FaissConfig*>(cfg.get());
        const auto metric = Str2FaissMetricType(fc->metric_type.value());
        if (!metric.has_value()) {
            return Status::invalid_metric_type;
        }
        is_cosine_ = IsMetricType(fc->metric_type.value(), knowhere::metric::COSINE);

        try {
            if constexpr (std::is_same_v<DataType, fp32>) {
                index_.reset(::faiss::index_factory(static_cast<int>(dataset->GetDim()),
                                                    fc->faiss_index_name.value().c_str(), metric.value()));
            } else {
                index_.reset(::faiss::index_binary_factory(static_cast<int>(dataset->GetDim()),
                                                           fc->faiss_index_name.value().c_str()));
            }
        } catch (const ::faiss::FaissException& e) {
            LOG_KNOWHERE_ERROR_ << "faiss::index_factory failed: " << e.what();
            return Status::invalid_args;
        }

        std::string err;
        auto st = faiss_vanilla::apply_build_params(index_.get(), fc->raw_params, &err);
        if (st != Status::success) {
            LOG_KNOWHERE_ERROR_ << err;
            return st;
        }

        try {
            const auto* raw = dataset->GetTensor();
            const auto n = dataset->GetRows();
            if constexpr (std::is_same_v<DataType, fp32>) {
                auto data = static_cast<const float*>(raw);
                std::unique_ptr<float[]> copy;
                if (is_cosine_) {
                    copy = CopyAndNormalizeVecs(data, n, dataset->GetDim());
                    data = copy.get();
                }
                index_->train(n, data);
            } else {
                index_->train(n, static_cast<const uint8_t*>(raw));
            }
        } catch (const ::faiss::FaissException& e) {
            LOG_KNOWHERE_ERROR_ << "faiss train failed: " << e.what();
            return Status::faiss_inner_error;
        }
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> /*cfg*/, bool /*use_knowhere_build_pool*/) override {
        if (!index_) {
            return Status::empty_index;
        }
        try {
            const auto* raw = dataset->GetTensor();
            const auto n = dataset->GetRows();
            if constexpr (std::is_same_v<DataType, fp32>) {
                auto data = static_cast<const float*>(raw);
                std::unique_ptr<float[]> copy;
                if (is_cosine_) {
                    copy = CopyAndNormalizeVecs(data, n, dataset->GetDim());
                    data = copy.get();
                }
                index_->add(n, data);
            } else {
                index_->add(n, static_cast<const uint8_t*>(raw));
            }
        } catch (const ::faiss::FaissException& e) {
            LOG_KNOWHERE_ERROR_ << "faiss add failed: " << e.what();
            return Status::faiss_inner_error;
        }
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* /*op_context*/) const override {
        if (!index_) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        const auto* fc = static_cast<const FaissConfig*>(cfg.get());
        const auto k = static_cast<int64_t>(fc->k.value());
        const auto nq = dataset->GetRows();

        BitsetViewIDSelector bw_sel(bitset);
        ::faiss::IDSelector* sel = bitset.empty() ? nullptr : &bw_sel;

        std::unique_ptr<::faiss::SearchParameters> search_params;
        std::string err_msg;

        auto ids = std::make_unique<int64_t[]>(nq * k);
        auto distances = std::make_unique<float[]>(nq * k);

        try {
            if constexpr (std::is_same_v<DataType, fp32>) {
                Status st = faiss_vanilla::build_search_params(static_cast<const ::faiss::Index*>(index_.get()),
                                                               fc->raw_params, sel, &search_params, &err_msg);
                if (st != Status::success) {
                    LOG_KNOWHERE_ERROR_ << err_msg;
                    return expected<DataSetPtr>::Err(st, err_msg);
                }

                const auto* raw = static_cast<const float*>(dataset->GetTensor());
                std::unique_ptr<float[]> norm_copy;
                if (is_cosine_) {
                    norm_copy = CopyAndNormalizeVecs(raw, nq, dataset->GetDim());
                    raw = norm_copy.get();
                }
                ThreadPool::ScopedSearchOmpSetter setter(1);
                index_->search(nq, raw, k, distances.get(), ids.get(), search_params.get());
            } else {
                Status st = faiss_vanilla::build_search_params(static_cast<const ::faiss::IndexBinary*>(index_.get()),
                                                               fc->raw_params, sel, &search_params, &err_msg);
                if (st != Status::success) {
                    LOG_KNOWHERE_ERROR_ << err_msg;
                    return expected<DataSetPtr>::Err(st, err_msg);
                }

                const auto* raw = static_cast<const uint8_t*>(dataset->GetTensor());
                // faiss binary search returns int32 distances; reinterpret buffer then cast to float
                auto int_distances = std::make_unique<int32_t[]>(nq * k);
                ThreadPool::ScopedSearchOmpSetter setter(1);
                index_->search(nq, raw, k, int_distances.get(), ids.get(), search_params.get());
                for (int64_t i = 0; i < nq * k; ++i) {
                    distances[i] = static_cast<float>(int_distances[i]);
                }
            }
        } catch (const ::faiss::FaissException& e) {
            LOG_KNOWHERE_ERROR_ << "faiss search failed: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
        return GenResultDataSet(nq, k, std::move(ids), std::move(distances));
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* /*op_context*/) const override {
        if (!index_) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        if constexpr (std::is_same_v<DataType, bin1>) {
            return expected<DataSetPtr>::Err(Status::not_implemented,
                                             "RangeSearch unsupported for binary faiss indexes");
        } else {
            const auto* fc = static_cast<const FaissConfig*>(cfg.get());
            const float radius = fc->radius.value();
            const float range_filter = fc->range_filter.value();
            const auto nq = dataset->GetRows();

            BitsetViewIDSelector bw_sel(bitset);
            ::faiss::IDSelector* sel = bitset.empty() ? nullptr : &bw_sel;

            std::unique_ptr<::faiss::SearchParameters> search_params;
            std::string err_msg;
            Status st = faiss_vanilla::build_search_params(static_cast<const ::faiss::Index*>(index_.get()),
                                                           fc->raw_params, sel, &search_params, &err_msg);
            if (st != Status::success) {
                LOG_KNOWHERE_ERROR_ << err_msg;
                return expected<DataSetPtr>::Err(st, err_msg);
            }

            const auto* raw = static_cast<const float*>(dataset->GetTensor());
            std::unique_ptr<float[]> norm_copy;
            if (is_cosine_) {
                norm_copy = CopyAndNormalizeVecs(raw, nq, dataset->GetDim());
                raw = norm_copy.get();
            }

            ::faiss::RangeSearchResult r(nq);
            try {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                index_->range_search(nq, raw, radius, &r, search_params.get());
            } catch (const ::faiss::FaissException& e) {
                const std::string msg = std::string("faiss range_search failed: ") + e.what() +
                                        ". Please check if the corresponding faiss index has "
                                        "implemented interface";
                LOG_KNOWHERE_ERROR_ << msg;
                return expected<DataSetPtr>::Err(Status::faiss_inner_error, msg);
            }

            const bool is_ip = is_cosine_ || IsMetricType(fc->metric_type.value(), knowhere::metric::IP);

            // Convert flat lims/labels/distances to nested vectors for GetRangeSearchResult
            std::vector<std::vector<float>> result_distances(nq);
            std::vector<std::vector<int64_t>> result_labels(nq);
            for (size_t i = 0; i < static_cast<size_t>(nq); ++i) {
                const size_t start = r.lims[i];
                const size_t end = r.lims[i + 1];
                result_distances[i].assign(r.distances + start, r.distances + end);
                result_labels[i].assign(r.labels + start, r.labels + end);
            }

            auto rr = GetRangeSearchResult(result_distances, result_labels, is_ip, nq, radius, range_filter);
            return GenResultDataSet(nq, std::move(rr));
        }
    }

    // Note: faiss's reconstruct() returns lossy approximations for quantized indexes
    // (PQ, SQ, etc.). Only IndexFlat and similar non-compressed indexes return exact data.
    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* /*op_context*/) const override {
        if (!index_) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        const auto nq = dataset->GetRows();
        const auto* ids = dataset->GetIds();
        const auto dim = index_->d;

        try {
            if constexpr (std::is_same_v<DataType, fp32>) {
                auto out = std::make_unique<float[]>(nq * dim);
                for (int64_t i = 0; i < nq; ++i) {
                    index_->reconstruct(ids[i], out.get() + i * dim);
                }
                return GenResultDataSet(nq, dim, std::move(out));
            } else {
                // dim is in bits for binary indexes; bytes = dim / 8
                const auto bytes_per_vec = dim / 8;
                auto out = std::make_unique<uint8_t[]>(nq * bytes_per_vec);
                for (int64_t i = 0; i < nq; ++i) {
                    index_->reconstruct(ids[i], out.get() + i * bytes_per_vec);
                }
                return GenResultDataSet(nq, dim, static_cast<void*>(out.release()));
            }
        } catch (const ::faiss::FaissException& e) {
            return expected<DataSetPtr>::Err(Status::not_implemented, e.what());
        }
    }

    bool
    HasRawData(const std::string& /*metric_type*/) const override {
        // Vanilla faiss adapter does not guarantee raw data from reconstruct();
        // quantized indexes (PQ, SQ) return lossy approximations. Callers
        // needing exact raw data should use a dedicated index wrapper.
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> /*cfg*/) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not supported");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_)
            return Status::empty_index;
        try {
            ::faiss::VectorIOWriter writer;
            if constexpr (std::is_same_v<DataType, fp32>) {
                ::faiss::write_index(index_.get(), &writer);
            } else {
                ::faiss::write_index_binary(index_.get(), &writer);
            }
            auto sz = writer.data.size();
            std::shared_ptr<uint8_t[]> buf(new uint8_t[sz]);
            std::memcpy(buf.get(), writer.data.data(), sz);
            binset.Append(Type(), buf, static_cast<int64_t>(sz));
            return Status::success;
        } catch (const ::faiss::FaissException& e) {
            LOG_KNOWHERE_ERROR_ << "Serialize failed: " << e.what();
            return Status::faiss_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> /*config*/) override {
        auto bin = binset.GetByName(Type());
        if (bin == nullptr)
            return Status::invalid_binary_set;
        try {
            ::faiss::VectorIOReader reader;
            reader.data.assign(bin->data.get(), bin->data.get() + bin->size);
            if constexpr (std::is_same_v<DataType, fp32>) {
                index_.reset(::faiss::read_index(&reader));
            } else {
                index_.reset(::faiss::read_index_binary(&reader));
            }
            return Status::success;
        } catch (const ::faiss::FaissException& e) {
            LOG_KNOWHERE_ERROR_ << "Deserialize failed: " << e.what();
            return Status::faiss_inner_error;
        }
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        const auto* fc = static_cast<const FaissConfig*>(config.get());
        const bool use_mmap = fc->enable_mmap.value_or(false);
        try {
            if constexpr (std::is_same_v<DataType, fp32>) {
                if (use_mmap) {
                    auto owner = std::make_shared<faiss::MmappedFileMappingOwner>(filename.data());
                    faiss::MappedFileIOReader reader(owner);
                    index_.reset(faiss::read_index(&reader));
                } else {
                    faiss::FileIOReader reader(filename.data());
                    index_.reset(faiss::read_index(&reader));
                }
            } else {
                if (use_mmap) {
                    auto owner = std::make_shared<faiss::MmappedFileMappingOwner>(filename.data());
                    faiss::MappedFileIOReader reader(owner);
                    index_.reset(faiss::read_index_binary(&reader));
                } else {
                    faiss::FileIOReader reader(filename.data());
                    index_.reset(faiss::read_index_binary(&reader));
                }
            }
            return Status::success;
        } catch (const ::faiss::FaissException& e) {
            LOG_KNOWHERE_ERROR_ << "DeserializeFromFile failed: " << e.what();
            return Status::faiss_inner_error;
        }
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    int64_t
    Dim() const override {
        return index_ ? index_->d : 0;
    }

    int64_t
    Size() const override {
        if (!index_)
            return 0;
        faiss::cppcontrib::knowhere::CountSizeIOWriter writer;
        if constexpr (std::is_same_v<DataType, fp32>) {
            faiss::write_index(index_.get(), &writer);
        } else {
            faiss::write_index_binary(index_.get(), &writer);
        }
        return static_cast<int64_t>(writer.total_size);
    }

    int64_t
    Count() const override {
        return index_ ? index_->ntotal : 0;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FAISS;
    }

 protected:
    std::unique_ptr<FaissBase<DataType>> index_;
    std::shared_ptr<ThreadPool> search_pool_;
    bool is_cosine_{false};
};

KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS, FaissIndexNode, fp32, knowhere::feature::MMAP | knowhere::feature::FLOAT32);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS, FaissIndexNode, bin1, knowhere::feature::MMAP | knowhere::feature::BINARY);

}  // namespace knowhere
