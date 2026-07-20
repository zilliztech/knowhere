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

#include "index/emb_list/emb_list_raw_storage.h"

#include <utility>
#include <vector>

#include "faiss/cppcontrib/knowhere/IndexBinaryFlat.h"
#include "faiss/cppcontrib/knowhere/IndexFlat.h"
#include "faiss/cppcontrib/knowhere/index_io.h"
#include "io/memory_io.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/context.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {
namespace {

inline bool
IsBinarySubMetric(const std::string& metric_type) {
    return metric_type == metric::HAMMING || metric_type == metric::JACCARD;
}

inline faiss::MetricType
ToFaissBinaryMetric(const std::string& metric_type) {
    return metric_type == metric::JACCARD ? faiss::METRIC_Jaccard : faiss::METRIC_Hamming;
}

inline int
PopcountByte(uint8_t value) {
    return __builtin_popcount(static_cast<unsigned int>(value));
}

class EmbListFloatRawStorage final : public EmbListRawStorage {
 public:
    EmbListFloatRawStorage(int64_t dim, faiss::MetricType metric_type)
        : index_(std::make_shared<faiss::cppcontrib::knowhere::IndexFlat>(dim, metric_type)) {
    }

    explicit EmbListFloatRawStorage(std::shared_ptr<::faiss::IndexFlat> index) : index_(std::move(index)) {
    }

    [[nodiscard]] int64_t
    Dim() const override {
        return index_->d;
    }

    [[nodiscard]] int64_t
    Count() const override {
        return index_->ntotal;
    }

    [[nodiscard]] size_t
    CodeSize() const override {
        return static_cast<size_t>(index_->d) * sizeof(float);
    }

    Status
    Add(const DataSetPtr dataset) override {
        index_->add(dataset->GetRows(), static_cast<const float*>(dataset->GetTensor()));
        return Status::success;
    }

    Status
    ReconstructN(int64_t start, int64_t n, uint8_t* out) const override {
        index_->reconstruct_n(start, n, reinterpret_cast<float*>(out));
        return Status::success;
    }

    Status
    Serialize(BinarySet& binset) const override {
        MemoryIOWriter writer;
        faiss::cppcontrib::knowhere::write_index(index_.get(), &writer);
        std::shared_ptr<uint8_t[]> raw_bin(writer.data());
        binset.Append(meta::EMB_LIST_RAW_INDEX, raw_bin, writer.tellg());
        return Status::success;
    }

    expected<DataSetPtr>
    CalcDistance(const DataSetPtr dataset, const int64_t* labels, size_t labels_len, const std::string& /*metric_type*/,
                 bool is_cosine, std::shared_ptr<ThreadPool> pool, milvus::OpContext* op_context) const override {
        auto num_queries = dataset->GetRows();
        auto dim = dataset->GetDim();
        auto query_data = dataset->GetTensor();
        auto distances = std::make_unique<float[]>(num_queries * labels_len);

        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(num_queries);
            for (int64_t i = 0; i < num_queries; ++i) {
                futs.emplace_back(pool->push([&, idx = i]() {
                    knowhere::checkCancellation(op_context);
                    std::unique_ptr<faiss::DistanceComputer> dist_computer(index_->get_distance_computer());

                    const float* cur_query = static_cast<const float*>(query_data) + idx * dim;
                    std::unique_ptr<float[]> copied_query = nullptr;
                    if (is_cosine) {
                        copied_query = CopyAndNormalizeVecs(cur_query, 1, dim);
                        cur_query = copied_query.get();
                    }

                    dist_computer->set_query(cur_query);
                    auto cur_distances = distances.get() + idx * labels_len;
                    for (size_t j = 0; j < labels_len; ++j) {
                        cur_distances[j] = (*dist_computer)(labels[j]);
                    }
                }));
            }
            WaitAllSuccess(futs);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "CalcDistance by float raw storage error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        return GenResultDataSet(num_queries, labels_len, std::unique_ptr<int64_t[]>{}, std::move(distances));
    }

 private:
    std::shared_ptr<::faiss::IndexFlat> index_;
};

class EmbListBinaryRawStorage final : public EmbListRawStorage {
 public:
    EmbListBinaryRawStorage(int64_t dim, faiss::MetricType metric_type)
        : index_(std::make_shared<faiss::cppcontrib::knowhere::IndexBinaryFlat>(dim, metric_type)) {
    }

    explicit EmbListBinaryRawStorage(std::shared_ptr<faiss::cppcontrib::knowhere::IndexBinary> index)
        : index_(std::move(index)) {
    }

    [[nodiscard]] int64_t
    Dim() const override {
        return index_->d;
    }

    [[nodiscard]] int64_t
    Count() const override {
        return index_->ntotal;
    }

    [[nodiscard]] size_t
    CodeSize() const override {
        return static_cast<size_t>(index_->code_size);
    }

    Status
    Add(const DataSetPtr dataset) override {
        index_->add(dataset->GetRows(), static_cast<const uint8_t*>(dataset->GetTensor()));
        return Status::success;
    }

    Status
    ReconstructN(int64_t start, int64_t n, uint8_t* out) const override {
        index_->reconstruct_n(start, n, out);
        return Status::success;
    }

    Status
    Serialize(BinarySet& binset) const override {
        MemoryIOWriter writer;
        faiss::cppcontrib::knowhere::write_index_binary(index_.get(), &writer);
        std::shared_ptr<uint8_t[]> raw_bin(writer.data());
        binset.Append(meta::EMB_LIST_RAW_INDEX, raw_bin, writer.tellg());
        return Status::success;
    }

    expected<DataSetPtr>
    CalcDistance(const DataSetPtr dataset, const int64_t* labels, size_t labels_len, const std::string& metric_type,
                 bool /*is_cosine*/, std::shared_ptr<ThreadPool> pool, milvus::OpContext* op_context) const override {
        auto num_queries = dataset->GetRows();
        auto dim = dataset->GetDim();
        if (dim != index_->d) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "binary raw index dim mismatch");
        }
        if (dim % 8 != 0) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "binary raw distance requires dim multiple of 8");
        }

        const size_t code_size = CodeSize();
        const auto* query_data = static_cast<const uint8_t*>(dataset->GetTensor());
        auto distances = std::make_unique<float[]>(num_queries * labels_len);

        try {
            std::vector<uint8_t> label_codes(labels_len * code_size);
            for (size_t j = 0; j < labels_len; ++j) {
                index_->reconstruct(labels[j], label_codes.data() + j * code_size);
            }

            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(num_queries);
            for (int64_t i = 0; i < num_queries; ++i) {
                futs.emplace_back(pool->push([&, idx = i]() {
                    knowhere::checkCancellation(op_context);
                    const uint8_t* cur_query = query_data + idx * code_size;
                    auto cur_distances = distances.get() + idx * labels_len;

                    for (size_t j = 0; j < labels_len; ++j) {
                        const uint8_t* cur_doc = label_codes.data() + j * code_size;
                        int intersection = 0;
                        int union_count = 0;
                        int hamming = 0;
                        for (size_t b = 0; b < code_size; ++b) {
                            const uint8_t q = cur_query[b];
                            const uint8_t d = cur_doc[b];
                            if (metric_type == metric::JACCARD) {
                                intersection += PopcountByte(static_cast<uint8_t>(q & d));
                                union_count += PopcountByte(static_cast<uint8_t>(q | d));
                            } else {
                                hamming += PopcountByte(static_cast<uint8_t>(q ^ d));
                            }
                        }

                        if (metric_type == metric::JACCARD) {
                            cur_distances[j] =
                                union_count == 0 ? 0.0f : 1.0f - static_cast<float>(intersection) / union_count;
                        } else {
                            cur_distances[j] = static_cast<float>(hamming);
                        }
                    }
                }));
            }
            WaitAllSuccess(futs);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "CalcDistance by binary raw storage error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        return GenResultDataSet(num_queries, labels_len, std::unique_ptr<int64_t[]>{}, std::move(distances));
    }

 private:
    std::shared_ptr<faiss::cppcontrib::knowhere::IndexBinary> index_;
};

}  // namespace

expected<std::shared_ptr<EmbListRawStorage>>
CreateEmbListRawStorageForBuild(const DataSetPtr dataset, const std::string& metric_type) {
    auto dim = dataset->GetDim();

    std::shared_ptr<EmbListRawStorage> storage;
    if (IsBinarySubMetric(metric_type)) {
        if (dim % 8 != 0) {
            LOG_KNOWHERE_WARNING_ << "Binary emb_list raw storage requires dim to be a multiple of 8, got " << dim;
            return expected<std::shared_ptr<EmbListRawStorage>>::Err(
                Status::invalid_args, "binary emb_list raw storage dim must be a multiple of 8");
        }
        storage = std::make_shared<EmbListBinaryRawStorage>(dim, ToFaissBinaryMetric(metric_type));
    } else {
        faiss::MetricType faiss_metric = faiss::METRIC_INNER_PRODUCT;
        if (metric_type == metric::L2) {
            faiss_metric = faiss::METRIC_L2;
        }
        storage = std::make_shared<EmbListFloatRawStorage>(dim, faiss_metric);
    }

    const auto status = storage->Add(dataset);
    if (status != Status::success) {
        return expected<std::shared_ptr<EmbListRawStorage>>::Err(status,
                                                                 "failed to add vectors to emb_list raw storage");
    }
    return storage;
}

expected<std::shared_ptr<EmbListRawStorage>>
ReadEmbListRawStorageFromBinary(const BinaryPtr& raw_index_bin, const std::string& metric_type) {
    MemoryIOReader reader(raw_index_bin->data.get(), raw_index_bin->size);
    if (IsBinarySubMetric(metric_type)) {
        auto* index = faiss::cppcontrib::knowhere::read_index_binary(&reader);
        auto* flat_index = dynamic_cast<faiss::cppcontrib::knowhere::IndexBinaryFlat*>(index);
        if (flat_index == nullptr) {
            delete index;
            LOG_KNOWHERE_WARNING_ << "EMB_LIST_RAW_INDEX is not an IndexBinaryFlat";
            return expected<std::shared_ptr<EmbListRawStorage>>::Err(Status::emb_list_inner_error,
                                                                     "EMB_LIST_RAW_INDEX is not an IndexBinaryFlat");
        }
        auto storage = std::make_shared<EmbListBinaryRawStorage>(
            std::shared_ptr<faiss::cppcontrib::knowhere::IndexBinary>(flat_index));
        LOG_KNOWHERE_INFO_ << "Loaded binary raw vector storage: " << storage->Count() << " vectors";
        return storage;
    }

    auto* index = faiss::cppcontrib::knowhere::read_index(&reader);
    auto* flat_index = dynamic_cast<::faiss::IndexFlat*>(index);
    if (flat_index == nullptr) {
        delete index;
        LOG_KNOWHERE_WARNING_ << "EMB_LIST_RAW_INDEX is not an IndexFlat";
        return expected<std::shared_ptr<EmbListRawStorage>>::Err(Status::emb_list_inner_error,
                                                                 "EMB_LIST_RAW_INDEX is not an IndexFlat");
    }
    auto storage = std::make_shared<EmbListFloatRawStorage>(std::shared_ptr<::faiss::IndexFlat>(flat_index));
    LOG_KNOWHERE_INFO_ << "Loaded raw vector storage: " << storage->Count() << " vectors";
    return storage;
}

expected<std::shared_ptr<EmbListRawStorage>>
ReadEmbListRawStorageFromFile(const std::string& filename, int io_flags, const std::string& metric_type) {
    if (IsBinarySubMetric(metric_type)) {
        auto* index = faiss::cppcontrib::knowhere::read_index_binary(filename.data(), io_flags);
        auto* flat_index = dynamic_cast<faiss::cppcontrib::knowhere::IndexBinaryFlat*>(index);
        if (flat_index == nullptr) {
            delete index;
            LOG_KNOWHERE_WARNING_ << "EMB_LIST_RAW_INDEX file is not an IndexBinaryFlat";
            return expected<std::shared_ptr<EmbListRawStorage>>::Err(
                Status::emb_list_inner_error, "EMB_LIST_RAW_INDEX file is not an IndexBinaryFlat");
        }
        auto storage = std::make_shared<EmbListBinaryRawStorage>(
            std::shared_ptr<faiss::cppcontrib::knowhere::IndexBinary>(flat_index));
        LOG_KNOWHERE_INFO_ << "Loaded binary raw vector storage from file: " << storage->Count() << " vectors";
        return storage;
    }

    auto* index = faiss::cppcontrib::knowhere::read_index(filename.data(), io_flags);
    auto* flat_index = dynamic_cast<::faiss::IndexFlat*>(index);
    if (flat_index == nullptr) {
        delete index;
        LOG_KNOWHERE_WARNING_ << "EMB_LIST_RAW_INDEX file is not an IndexFlat";
        return expected<std::shared_ptr<EmbListRawStorage>>::Err(Status::emb_list_inner_error,
                                                                 "EMB_LIST_RAW_INDEX file is not an IndexFlat");
    }
    auto storage = std::make_shared<EmbListFloatRawStorage>(std::shared_ptr<::faiss::IndexFlat>(flat_index));
    LOG_KNOWHERE_INFO_ << "Loaded raw vector storage from file: " << storage->Count() << " vectors";
    return storage;
}

}  // namespace knowhere
