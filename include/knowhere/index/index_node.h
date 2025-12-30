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

#ifndef INDEX_NODE_H
#define INDEX_NODE_H

#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/emb_list_utils.h"
#include "knowhere/expected.h"
#include "knowhere/object.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"
#include "ncs/ncs.h"

#if defined(NOT_COMPILE_FOR_SWIG)
#include "common/OpContext.h"
#else
namespace milvus {
struct OpContext;
}  // namespace milvus
#endif

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/comp/task.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

class Interrupt;

class IndexNode : public Object {
 public:
    IndexNode(const int32_t ver) : version_(ver) {
    }

    IndexNode() : version_(Version::GetDefaultVersion()) {
    }

    IndexNode(const IndexNode& other) : version_(other.version_) {
    }

    IndexNode(const IndexNode&& other) : version_(other.version_) {
    }

    /**
     * @brief Builds the index using the provided dataset and configuration.
     *
     * Mostly, this method combines the `Train` and `Add` steps to create the index structure, but it can be overridden
     * if the index doesn't support Train-Add pattern, such as immutable indexes like DiskANN.
     *
     * @param dataset Dataset to build the index from.
     * @param cfg
     * @return Status.
     *
     * @note Indexes need to be ready to search after `Build` is called. TODO:@liliu-z DiskANN is an exception and need
     * to be revisited.
     * @note Providing support for the async interface is possible.Since the config object needs to be held in a future
     * or lambda function, a smart pointer is required to delay its release.
     * @note Since the build interface aggregates calls to both the train and add interfaces, `unique_ptr` cannot be
     * shared, so `shared_ptr` is used instead.
     */
    virtual Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool = true) {
        RETURN_IF_ERROR(Train(dataset, cfg, use_knowhere_build_pool));
        return Add(dataset, std::move(cfg), use_knowhere_build_pool);
    }

/*
 *@ @brief Builds the index using the provided dataset,configuration and handle.
 */
#ifdef KNOWHERE_WITH_CARDINAL
    virtual Status
    BuildAsync(const DataSetPtr dataset, std::shared_ptr<Config> cfg, const Interrupt* = nullptr) {
        return Build(dataset, std::move(cfg), true);
    }
#endif

    /**
     * @brief Trains the index model using the provided dataset and configuration.
     *
     * @param dataset Dataset used to train the index.
     * @param cfg
     * @return Status.
     *
     * @note This interface is only available for growable indexes. For immutable indexes like DiskANN, this method
     * should return an error.
     * @note Providing support for the async interface is possible.Since the config object needs to be held in a future
     * or lambda function, a smart pointer is required to delay its release.
     * @note Since the build interface aggregates calls to both the train and add interfaces, `unique_ptr` cannot be
     * shared, so `shared_ptr` is used instead.
     */
    virtual Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool = true) = 0;

    /**
     * @brief Adds data to the trained index.
     *
     * @param dataset Dataset to add to the index.
     * @param cfg
     * @return Status
     *
     * @note
     * 1. This interface is only available for growable indexes. For immutable indexes like DiskANN, this method
     * should return an error.
     * 2. This method need to be thread safe when called with search methods like @see Search, @see RangeSearch and @see
     * AnnIterator.
     * @note Providing support for the async interface is possible.Since the config object needs to be held in a future
     * or lambda function, a smart pointer is required to delay its release.
     * @note Since the build interface aggregates calls to both the train and add interfaces, `unique_ptr` cannot be
     * shared, so `shared_ptr` is used instead.
     */
    virtual Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool = true) = 0;

    /**
     * @brief Performs a search operation on the index.
     *
     * @param dataset Query vectors.
     * @param cfg
     * @param bitset A BitsetView object for filtering results.
     * @return An expected<> object containing the search results or an error.
     * @note Since the config object needs to be held in a future or lambda function, a smart pointer is required to
     * delay its release.
     */
    virtual expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context = nullptr) const = 0;

    /**
     * @brief Performs a brute-force search operation on the index for given labels.
     *
     * @param dataset Query vectors.
     * @param labels
     * @param labels_len
     * @param is_cosine
     * @return An expected<> object containing the search results or an error.
     *
     * @note emb-list index search will use this method to perform brute-force distance calculation.
     * @note Any index_node that supports emb list search must implement this method.
     */
    virtual expected<DataSetPtr>
    CalcDistByIDs(const DataSetPtr dataset, const BitsetView& bitset, const int64_t* labels, const size_t labels_len,
                  const bool is_cosine, milvus::OpContext* op_context = nullptr) const {
        return expected<DataSetPtr>::Err(Status::not_implemented,
                                         "BruteForceByIDs not supported for current index type");
    };

    // not thread safe.
    class iterator {
     public:
        virtual std::pair<int64_t, float>
        Next() = 0;
        [[nodiscard]] virtual bool
        HasNext() = 0;
        virtual ~iterator() {
        }
    };
    using IteratorPtr = std::shared_ptr<iterator>;

    virtual expected<std::vector<IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool = true, milvus::OpContext* op_context = nullptr) const {
        return expected<std::vector<std::shared_ptr<iterator>>>::Err(
            Status::not_implemented, "annIterator not supported for current index type");
    }

    /**
     * @brief Performs a range search operation on the index.
     *
     * This method provides a default implementation of range search based on the `AnnIterator`, assuming the iterator
     * will buffer an expanded range and return the closest elements on each Next() call. It can be overridden by
     * derived classes for more efficient implementations.
     *
     * @param dataset Query vectors.
     * @param cfg
     * @param bitset A BitsetView object for filtering results.
     * @return An expected<> object containing the range search results or an error.
     * @note Since the config object needs to be held in a future or lambda function, a smart pointer is required to
     * delay its release.
     */
    virtual expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* op_context = nullptr) const;

    /**
     * @brief Retrieves raw vectors by their IDs from the index.
     *
     * @param dataset Dataset containing the IDs of the vectors to retrieve.
     * @return An expected<> object containing the retrieved vectors or an error.
     *
     * @note
     * 1. This method may return an error if the index does not contain raw data. The returned raw data must be exactly
     * the same as the input data when we do @see Add or @see Build. For example, if the datatype is BF16, then we need
     * to return a dataset with BF16 vectors.
     * 2. It doesn't guarantee the index contains raw data, so it's better to check with @see HasRawData() before
     */
    virtual expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context = nullptr) const = 0;

    /**
     * @brief Checks if the index contains raw vector data.
     *
     * @param metric_type The metric type used in the index.
     * @return true if the index contains raw data, false otherwise.
     */
    virtual bool
    HasRawData(const std::string& metric_type) const = 0;

    virtual bool
    IsAdditionalScalarSupported(bool is_mv_only) const {
        return false;
    }

    /**
     * @unused Milvus is not using this method, so it cannot guarantee all indexes implement this method.
     *
     * This is for Feder, and we can ignore it for now.
     */
    virtual expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const = 0;

    /**
     * @brief Serializes the index to a binary set.
     *
     * @param binset The BinarySet to store the serialized index.
     * @return Status indicating success or failure of the serialization.
     */
    virtual Status
    Serialize(BinarySet& binset) const = 0;

    /**
     * @brief Deserializes the index from a binary set.
     *
     * @param binset The BinarySet containing the serialized index.
     * @param config
     * @return Status indicating success or failure of the deserialization.
     *
     * @note
     * 1. The index should be ready to search after deserialization.
     * 2. For immutable indexes, the path for now if Build->Serialize->Deserialize->Search.
     * 3. Since the config object needs to be held in a future or lambda function, a smart pointer is required to delay
     * its release.
     */
    virtual Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) = 0;

    /**
     * @brief Deserializes the index from a file.
     *
     * This method is mostly used for mmap deserialization. However, it has some conflicts with the FileManager that we
     * used to deserialize DiskANN. TODO: @liliu-z some redesign is needed here.
     *
     * @param filename Path to the file containing the serialized index.
     * @param config
     * @return Status indicating success or failure of the deserialization.
     */
    virtual Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) = 0;

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const = 0;

    /**
     * @brief Gets the dimensionality of the vectors in the index.
     *
     * @return The number of dimensions as an int64_t.
     */
    virtual int64_t
    Dim() const = 0;

    /**
     * @unused Milvus is not using this method, so it cannot guarantee all indexes implement this method.
     *
     * @brief Gets the memory usage of the index in bytes.
     *
     * @return The size of the index as an int64_t.
     * @note This method doesn't have to be very accurate.
     */
    virtual int64_t
    Size() const = 0;

    /**
     * @brief Gets the number of vectors in the index.
     *
     * @return The count of vectors as an int64_t.
     */
    virtual int64_t
    Count() const = 0;

    virtual std::string
    Type() const = 0;

    /**
     * @brief Gets the mapping from internal IDs to external IDs.
     *
     * @return A reference to the mapping vector.
     * @note If not implemented, the default implementation is to return a mapping, from 0 to Count()-1.
     */
    virtual std::shared_ptr<std::vector<uint32_t>>
    GetInternalIdToExternalIdMap() const {
        auto n_rows = Count();
        auto internal_id_to_external_id_map = std::make_shared<std::vector<uint32_t>>(n_rows);
        std::iota(internal_id_to_external_id_map->begin(), internal_id_to_external_id_map->end(), 0);
        return internal_id_to_external_id_map;
    }

    /**
     * @brief Sets the mapping from internal IDs to "most external" IDs for 1-hop bitset check!
     * Only used for hierarchical indexnode, such as emb_list + hnsw, each index node has its own relayout mapping.
     *
     * @param map The mapping vector to set.
     * @return Status indicating success or failure of the mapping.
     */
    virtual Status
    SetInternalIdToMostExternalIdMap(std::vector<uint32_t>&& map) {
        return Status::not_implemented;
    }

    /**
     * @brief Establishes the mapping from internal base-index IDs to emb_list IDs.
     *
     * This mapping is essential for base indexes to correctly apply bitset filtering using only a 1-hop mapping during
     * search. In some cases, such as with mv-only *relayout*, a base-index may have its own
     * (base)internal-to-external ID mapping.
     * However, the emb_list search bitset operates on emb_list IDs, which we refer to as the "most external" IDs.
     * Therefore, we need to create a mapping from the base_internal_id (used by the base-index) to the most external
     * emb_list_id, ensuring that bitset checks and search results are consistent at the emb_list level.
     */
    Status
    SetBaseIndexIDMap() {
        auto internal_id_to_external_id_map = GetInternalIdToExternalIdMap();
        size_t id_map_size = internal_id_to_external_id_map->size();
        assert(id_map_size == static_cast<size_t>(Count()));
        std::vector<uint32_t> internal_id_to_most_external_id_map(id_map_size);
        for (size_t i = 0; i < id_map_size; i++) {
            internal_id_to_most_external_id_map[i] = emb_list_offset_->get_el_id(internal_id_to_external_id_map->at(i));
        }
        return SetInternalIdToMostExternalIdMap(std::move(internal_id_to_most_external_id_map));
    }

    virtual Status
    BuildEmbList(const DataSetPtr dataset, std::shared_ptr<Config> cfg, const size_t* lims, size_t num_rows,
                 bool use_knowhere_build_pool = true) {
        // 1. split metric_type to el_metric_type and sub_metric_type
        auto& config = static_cast<BaseConfig&>(*cfg);
        auto original_metric_type = config.metric_type.value();
        auto el_metric_type_or = get_el_metric_type(original_metric_type);
        if (!el_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid metric type for emb_list: " << original_metric_type;
            return Status::emb_list_inner_error;
        }
        auto el_metric_type = el_metric_type_or.value();
        auto sub_metric_type_or = get_sub_metric_type(original_metric_type);
        if (!sub_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid sub metric type for emb_list: " << original_metric_type;
            return Status::emb_list_inner_error;
        }
        // set sub metric type as the metric type for build
        auto sub_metric_type = sub_metric_type_or.value();
        config.metric_type = sub_metric_type;

        // 2. build index
        LOG_KNOWHERE_INFO_ << "Build EmbList-Index with metric type: " << original_metric_type
                           << ", el metric type: " << el_metric_type << ", sub metric type: " << sub_metric_type;
        RETURN_IF_ERROR(Build(dataset, cfg, use_knowhere_build_pool));

        // 3. create emb_list_offset
        emb_list_offset_ = std::make_unique<EmbListOffset>(lims, num_rows);

        // 4. Set the mapping from base index internal vector IDs to emb_list IDs.
        // When using emb_list, all filtering bitset checks are performed at the emb_list level,
        // not at the individual vector level. This means that whenever the index needs to check whether
        // a vector is masked (filtered out), it must first map the vector's idx to its corresponding emb_list idx.
        return SetBaseIndexIDMap();
    }

    virtual Status
    BuildEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool = true) {
        auto& config = static_cast<BaseConfig&>(*cfg);
        auto el_metric_type_or = get_el_metric_type(config.metric_type.value());
        if (!el_metric_type_or.has_value()) {
            // if not emb_list, use the default build method
            return Build(dataset, std::move(cfg), use_knowhere_build_pool);
        }
        if (dataset == nullptr) {
            LOG_KNOWHERE_WARNING_
                << "Dataset is nullptr, but metric type is emb_list, need emb_list_offset from dataset";
            return Status::emb_list_inner_error;
        }
        const size_t* lims = dataset->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        if (lims == nullptr) {
            LOG_KNOWHERE_WARNING_ << "Could not find emb list offset from dataset, but metric type is emb_list";
            return Status::emb_list_inner_error;
        }

        return BuildEmbList(dataset, std::move(cfg), lims, dataset->GetRows(), use_knowhere_build_pool);
    }

    virtual Status
    AddEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool = true) {
        auto& config = static_cast<BaseConfig&>(*cfg);
        auto el_metric_type_or = get_el_metric_type(config.metric_type.value());
        if (!el_metric_type_or.has_value()) {
            // if not emb_list, use the default build method
            return Add(dataset, std::move(cfg), use_knowhere_build_pool);
        }
        if (dataset == nullptr) {
            LOG_KNOWHERE_WARNING_
                << "Dataset is nullptr, but metric type is emb_list, need emb_list_offset from dataset";
            return Status::emb_list_inner_error;
        }
        const size_t* lims = dataset->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        if (lims == nullptr) {
            LOG_KNOWHERE_WARNING_ << "Could not find emb list offset from dataset, but metric type is emb_list";
            return Status::emb_list_inner_error;
        }
        return AddEmbList(dataset, std::move(cfg), lims, dataset->GetRows(), use_knowhere_build_pool);
    }

    virtual Status
    AddEmbList(const DataSetPtr dataset, std::shared_ptr<Config> cfg, const size_t* lims, size_t num_rows,
               bool use_knowhere_build_pool = true) {
        LOG_KNOWHERE_WARNING_ << "AddEmbList not implemented for current index type";
        return Status::not_implemented;
    }

    virtual Status
    BulidAsyncEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                            const Interrupt* interrupt = nullptr) {
        return BuildEmbListIfNeed(dataset, std::move(cfg), true);
    }

    /**
     * @brief Serializes the index to a binary set, including the emb_list meta if is emb_list.
     *
     * @param binset The BinarySet to store the serialized index.
     * @return Status indicating success or failure of the serialization.
     */
    virtual Status
    SerializeEmbListIfNeed(BinarySet& binset) const {
        if (emb_list_offset_ == nullptr || emb_list_offset_->offset.size() == 0) {
            // if not emb_list, use the default serialize method
            return Serialize(binset);
        }

        // if is emb_list,
        //   1. serialize emb_list offset
        //   2. serialize base index
        LOG_KNOWHERE_INFO_ << "Serialize emb_list offset";
        try {
            // serialize emb_list_offset_
            // 1 * size_t + offset.size() * size_t
            int64_t total_bytes = (emb_list_offset_->offset.size() + 1) * sizeof(size_t);
            auto data = std::shared_ptr<uint8_t[]>(new uint8_t[total_bytes]);
            auto size = emb_list_offset_->offset.size();
            std::memcpy(data.get(), &size, sizeof(size_t));
            std::memcpy(data.get() + sizeof(size_t), emb_list_offset_->offset.data(), size * sizeof(size_t));
            binset.Append(knowhere::meta::EMB_LIST_META, data, total_bytes);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "serialize emb_list offset error: " << e.what();
            return Status::emb_list_inner_error;
        }
        return Serialize(binset);
    }

    /**
     * @brief Deserializes the index from a binary set, including the emb_list meta if is emb_list.
     *
     * @param binset The BinarySet containing the serialized index.
     * @param config
     * @return Status indicating success or failure of the deserialization.
     */
    virtual Status
    DeserializeEmbListIfNeed(const BinarySet& binset, std::shared_ptr<Config> config) {
        auto cfg = static_cast<const knowhere::BaseConfig&>(*config);
        auto el_metric_type_or = get_el_metric_type(cfg.metric_type.value());
        if (!el_metric_type_or.has_value()) {
            // if not emb_list, use the default deserialize method
            return Deserialize(binset, config);
        }

        // if is emb_list,
        //   1. split metric_type into el_metric_type and sub_metric_type
        //   2. deserialize base index
        //   2. deserialize emb_list offset
        //   3. set base index id map

        el_metric_type_ = el_metric_type_or.value();
        auto sub_metric_type_or = get_sub_metric_type(cfg.metric_type.value());
        if (!sub_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid sub metric type: " << cfg.metric_type.value();
            return Status::emb_list_inner_error;
        }
        cfg.metric_type = sub_metric_type_or.value();
        RETURN_IF_ERROR(Deserialize(binset, config));

        try {
            auto binary_ptr = binset.GetByName(knowhere::meta::EMB_LIST_META);
            if (binary_ptr == nullptr) {
                LOG_KNOWHERE_INFO_ << "No emb_list offset found, but metric type is emb_list";
                return Status::emb_list_inner_error;
            }
            LOG_KNOWHERE_INFO_ << "Deserialize emb_list offset";
            size_t size = 0;
            std::memcpy(&size, binary_ptr->data.get(), sizeof(size_t));
            const auto total_bytes = binary_ptr->size;
            const auto comp_size = total_bytes / sizeof(size_t) - 1;
            if (comp_size != size) {
                LOG_KNOWHERE_WARNING_ << "the computed size of emb_list offset is not equal to size from binary set";
                return Status::emb_list_inner_error;
            }
            std::vector<size_t> offset(size);
            std::memcpy(offset.data(), binary_ptr->data.get() + sizeof(size_t), size * sizeof(size_t));
            emb_list_offset_ = std::make_unique<EmbListOffset>(std::move(offset));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "deserialize emb_list offset error: " << e.what();
            return Status::emb_list_inner_error;
        }

        return SetBaseIndexIDMap();
    }

    virtual bool
    LoadIndexWithStream() {
        return false;
    }

    /**
     * @brief Lists files required for NCS upload. The strings should be considered as patterns. Any file matching "*<pattern>*" is required.
     *
     * @return A vector of file name patterns.
     * @note The default implementation returns an empty vector, indicating no files are required for NCS upload.
     */
    virtual std::vector<std::string>
    ListFilesForNcsUpload() const{
        return {};
    }

    virtual milvus::NcsStatus
    NcsUpload(std::shared_ptr<Config> cfg) {
        return milvus::NcsStatus::ERROR;
    }


    virtual ~IndexNode() {
    }

    virtual Status
    DeserializeFromFileIfNeed(const std::string& filename, std::shared_ptr<Config> config) {
        auto cfg = static_cast<const knowhere::BaseConfig&>(*config);
        auto el_metric_type_or = get_el_metric_type(cfg.metric_type.value());
        if (!el_metric_type_or.has_value()) {
            // if not emb_list, use the default deserialize method
            return DeserializeFromFile(filename, config);
        }

        // if is emb_list,
        //   1. split metric_type into el_metric_type and sub_metric_type
        //   2. deserialize base index
        //   3. deserialize emb_list offset
        //   4. set base index id map

        el_metric_type_ = el_metric_type_or.value();
        auto sub_metric_type_or = get_sub_metric_type(cfg.metric_type.value());
        if (!sub_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid sub metric type: " << cfg.metric_type.value();
            return Status::emb_list_inner_error;
        }
        cfg.metric_type = sub_metric_type_or.value();
        RETURN_IF_ERROR(DeserializeFromFile(filename, config));

        try {
            auto emb_list_meta_file_path = cfg.emb_list_meta_file_path.value();
            if (emb_list_meta_file_path.empty()) {
                LOG_KNOWHERE_WARNING_ << "emb_list_meta_file is empty, but metric type is emb_list";
                return Status::emb_list_inner_error;
            }

            std::ifstream emb_list_meta_file(emb_list_meta_file_path, std::ios::binary);
            if (!emb_list_meta_file.is_open()) {
                LOG_KNOWHERE_WARNING_ << "emb_list_meta_file does not exist: " << emb_list_meta_file_path;
                return Status::emb_list_inner_error;
            }
            size_t size = 0;
            char size_buffer[sizeof(size_t)];
            emb_list_meta_file.read(size_buffer, sizeof(size_t));
            std::memcpy(&size, size_buffer, sizeof(size_t));
            std::vector<size_t> offset(size);
            emb_list_meta_file.read(reinterpret_cast<char*>(offset.data()), size * sizeof(size_t));
            emb_list_offset_ = std::make_unique<EmbListOffset>(std::move(offset));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "deserialize emb_list offset error: " << e.what();
            return Status::emb_list_inner_error;
        }

        return SetBaseIndexIDMap();
    }

    virtual expected<DataSetPtr>
    SearchEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset,
                        milvus::OpContext* op_context = nullptr) const;

    /**
     * @brief Returns the code size (in bytes) of a single query vector, which varies depending on the data type (e.g.,
     * fp32, bf16, etc).
     * @param dataset The query dataset.
     * @return The code size of each query vector.
     *
     * @note For emb list search, the query dataset must be split according to emb list offsets.
     *       Any index_node that supports emb list search must implement this method.
     */
    virtual std::optional<size_t>
    GetQueryCodeSize(const DataSetPtr dataset) const {
        throw std::runtime_error("GetQueryCodeSize not supported for current index type");
    }

    /**
     * @brief Search interface supporting two-stage emb_list search.
     * @param dataset Query dataset
     * @param cfg     Search configuration
     * @param bitset  Mask for filtering vectors
     *
     * Search process:
     * 1. Check emb_list offset information and build the query group structure.
     * 2. Stage 1: Call (vector-based) Search method to retrieve candidate vector IDs for each query emb_list.
     * 3. Stage 2: For each query emb_list, collect candidate emb_list IDs and its vectors, and perform brute-force
     *    distance calculation to aggregate scores at the emb_list level.
     * 4. Return top-k emb_list results.
     *
     * Note: The emb_list index node does not need to split tasks by nq and dispatch them to the search thread pool for
     * parallel processing, because the (vector-based) Search method already handles this.
     */
    virtual expected<DataSetPtr>
    SearchEmbList(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                  milvus::OpContext* op_context = nullptr) const;

    virtual expected<DataSetPtr>
    RangeSearchEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                             milvus::OpContext* op_context = nullptr) const;

    virtual expected<std::vector<IteratorPtr>>
    AnnIteratorEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                             bool use_knowhere_search_pool = true, milvus::OpContext* op_context = nullptr) const;

 protected:
    Version version_;
    std::unique_ptr<EmbListOffset> emb_list_offset_;  // emb_list group offset structure
    std::string el_metric_type_;
};

// Common superclass for iterators that expand search range as needed. Subclasses need
//   to override `next_batch` which will add expanded vectors to the results. For indexes
//   with quantization, override `raw_distance`.
// Internally, this structure uses the same priority queue class, but may multiply all
//   incoming distances to (-1) value in order to turn max priority queue into a min one.
// If use_knowhere_search_pool is True (the default), the iterator->Next() will be scheduled by the
//   knowhere_search_thread_pool.
//   If False, will Not involve thread scheduling internally, so please take caution.
class IndexIterator : public IndexNode::iterator {
 public:
    IndexIterator(bool larger_is_closer, bool use_knowhere_search_pool = true, float refine_ratio = 0.0f,
                  bool retain_iterator_order = false)
        : refine_ratio_(refine_ratio),
          refine_(refine_ratio != 0.0f),
          retain_iterator_order_(retain_iterator_order),
          sign_(larger_is_closer ? -1 : 1),
          use_knowhere_search_pool_(use_knowhere_search_pool) {
    }

    std::pair<int64_t, float>
    Next() override {
        if (!initialized_) {
            initialize();
        }
        auto& q = !refine_ ? res_ : refined_res_;
        if (q.empty()) {
            throw std::runtime_error("No more elements");
        }
        auto ret = q.top();
        q.pop();

        auto update_next_func = [&]() {
            UpdateNext();
            if (retain_iterator_order_) {
                while (HasNext()) {
                    auto& q = !refine_ ? res_ : refined_res_;
                    auto next_ret = q.top();
                    // with the help of `sign_`, both `res_` and `refine_res` are min-heap.
                    //   such as `COSINE`, `-dist` will be inserted to `res_` or `refine_res`.
                    // just make sure that the next value is greater than or equal to the current value.
                    if (next_ret.val >= ret.val) {
                        break;
                    }
                    q.pop();
                    UpdateNext();
                }
            }
        };
        if (use_knowhere_search_pool_) {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            std::vector<folly::Future<folly::Unit>> futs;
            futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&]() {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                update_next_func();
            }));
            WaitAllSuccess(futs);
#else
            update_next_func();
#endif
        } else {
            update_next_func();
        }

        return std::make_pair(ret.id, ret.val * sign_);
    }

    [[nodiscard]] bool
    HasNext() override {
        if (!initialized_) {
            initialize();
        }
        return !res_.empty() || !refined_res_.empty();
    }

    virtual void
    initialize() {
        if (initialized_) {
            throw std::runtime_error("initialize should not be called twice");
        }
        UpdateNext();
        initialized_ = true;
    }

 protected:
    inline size_t
    min_refine_size() const {
        // TODO: maybe make this configurable
        return std::max((size_t)20, (size_t)(res_.size() * refine_ratio_));
    }

    virtual void
    next_batch(std::function<void(const std::vector<DistId>&)> batch_handler) {
        throw std::runtime_error("next_batch not implemented");
    }
    // will be called only if refine_ratio_ is not 0.
    virtual float
    raw_distance(int64_t) {
        if (!refine_) {
            throw std::runtime_error("raw_distance should not be called for indexes without quantization");
        }
        throw std::runtime_error("raw_distance not implemented");
    }

    const float refine_ratio_;
    const bool refine_;
    bool initialized_ = false;
    bool retain_iterator_order_ = false;
    const int64_t sign_;

    std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> res_;
    // unused if refine_ is false
    std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> refined_res_;

 private:
    void
    UpdateNext() {
        auto batch_handler = [this](const std::vector<DistId>& batch) {
            for (const auto& dist_id : batch) {
                res_.emplace(dist_id.id, dist_id.val * sign_);
            }
            if (refine_) {
                while (!res_.empty() && (refined_res_.empty() || refined_res_.size() < min_refine_size())) {
                    auto pair = res_.top();
                    res_.pop();
                    refined_res_.emplace(pair.id, raw_distance(pair.id) * sign_);
                }
            }
        };
        next_batch(batch_handler);
    }

    bool use_knowhere_search_pool_ = true;
};

// An iterator implementation that accepts a function to get distances and ids list and returns them in order.
// We do not directly accept a distance list as input. The main reason for this is to minimize the initialization time
//   for all types of iterators in the `ANNIterator` interface, moving heavy computations to the first '->Next()' call.
//   This way, the iterator initialization does not need to perform any concurrent acceleration, and the search pool
//   only needs to handle the heavy work of `->Next()`
class PrecomputedDistanceIterator : public IndexNode::iterator {
 public:
    PrecomputedDistanceIterator(std::function<std::vector<DistId>()> compute_dist_func, bool larger_is_closer,
                                bool use_knowhere_search_pool = true)
        : compute_dist_func_(compute_dist_func),
          larger_is_closer_(larger_is_closer),
          use_knowhere_search_pool_(use_knowhere_search_pool) {
    }

    std::pair<int64_t, float>
    Next() override {
        if (!initialized_) {
            initialize();
        }
        if (use_knowhere_search_pool_) {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            std::vector<folly::Future<folly::Unit>> futs;
            futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&]() {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                sort_next();
            }));
            WaitAllSuccess(futs);
#else
            sort_next();
#endif
        } else {
            sort_next();
        }
        auto& result = results_[next_++];
        return std::make_pair(result.id, result.val);
    }

    [[nodiscard]] bool
    HasNext() override {
        if (!initialized_) {
            initialize();
        }
        return next_ < results_.size() && results_[next_].id != -1;
    }

    void
    initialize() {
        if (initialized_) {
            throw std::runtime_error("initialize should not be called twice");
        }
        if (use_knowhere_search_pool_) {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            std::vector<folly::Future<folly::Unit>> futs;
            futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&]() {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                results_ = compute_dist_func_();
            }));
            WaitAllSuccess(futs);
#else
            results_ = compute_dist_func_();
#endif
        } else {
            results_ = compute_dist_func_();
        }
        sort_size_ = get_sort_size(results_.size());
        sort_next();
        initialized_ = true;
    }

 private:
    static inline size_t
    get_sort_size(size_t rows) {
        return std::max((size_t)50000, rows / 10);
    }

    // sort the next sort_size_ elements
    inline void
    sort_next() {
        if (next_ < sorted_) {
            return;
        }
        size_t current_end = std::min(results_.size(), sorted_ + sort_size_);
        if (larger_is_closer_) {
            std::partial_sort(results_.begin() + sorted_, results_.begin() + current_end, results_.end(),
                              std::greater<DistId>());
        } else {
            std::partial_sort(results_.begin() + sorted_, results_.begin() + current_end, results_.end(),
                              std::less<DistId>());
        }

        sorted_ = current_end;
    }

    std::function<std::vector<DistId>()> compute_dist_func_;
    const bool larger_is_closer_;
    bool use_knowhere_search_pool_ = true;
    bool initialized_ = false;
    std::vector<DistId> results_;
    size_t next_ = 0;
    size_t sorted_ = 0;
    size_t sort_size_ = 0;
};

}  // namespace knowhere

#endif /* INDEX_NODE_H */
