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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/emb_list_utils.h"
#include "knowhere/expected.h"
#include "knowhere/id_map.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/object.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"

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
#else
class ThreadPool;
#endif

namespace faiss {
class IndexFlat;
}  // namespace faiss

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
    // Next()/HasNext() report errors through the returned expected<> instead of throwing,
    // consistent with the error-code based contract of the facade APIs. Implementations
    // should convert exceptions at this boundary (see IndexIterator/GuardedCall); the
    // built-in implementations additionally declare their overrides noexcept.
    class iterator {
     public:
        virtual expected<std::pair<int64_t, float>>
        Next() = 0;
        [[nodiscard]] virtual expected<bool>
        HasNext() = 0;
        virtual ~iterator() {
        }
    };
    using IteratorPtr = std::shared_ptr<iterator>;

    struct PreparedBitset {
        // BitsetView after id-domain projection.
        BitsetView bitset;

        PreparedBitset() = default;

        explicit PreparedBitset(BitsetView bitset) : bitset(bitset) {
        }
    };

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
    NeedBitsetExactCount() const {
        // Backends with filter-ratio shortcuts need exact projected counts.
        return false;
    }

    // Projects a public-id bitset to the backend id domain.
    virtual expected<PreparedBitset>
    PrepareBitset(BitsetView bitset) const {
        PreparedBitset prepared(bitset);
        const auto& id_map = GetIdMap();
        const auto count = id_map.OutCount();
        if (count != 0 && bitset.num_bits() > static_cast<size_t>(count)) {
            const auto msg = std::string("bitset size should be <= external count, but we get bitset size: ") +
                             std::to_string(bitset.num_bits()) + ", external count: " + std::to_string(count);
            LOG_KNOWHERE_ERROR_ << msg;
            return expected<PreparedBitset>::Err(Status::invalid_args, msg);
        }
        if (bitset.num_bits() == 0) {
            return prepared;
        }
        if (bitset.data() == nullptr) {
            const auto msg = std::string("bitset data is null while bitset size is non-zero");
            LOG_KNOWHERE_ERROR_ << msg;
            return expected<PreparedBitset>::Err(Status::invalid_args, msg);
        }

        PrepareBitsetMap(prepared, id_map);
        if (NeedBitsetExactCount()) {
            CalcBitsetCount(prepared.bitset, id_map, 0, std::numeric_limits<size_t>::max());
        }
        return prepared;
    }

    virtual bool
    IsAdditionalScalarSupported(bool is_mv_only) const {
        return false;
    }

    virtual bool
    IsIndexRefineEnabled() const {
        return false;
    }

    /**
     * @unused Not every index implementation supports this optional metadata path.
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

    virtual IdMap&
    GetIdMap() {
        return id_map_;
    }

    virtual const IdMap&
    GetIdMap() const {
        return id_map_;
    }

    virtual void
    SetIdMapType(IdMap::Type type) {
        // Selects sealed/growing id-map storage for this node and wrappers.
        GetIdMap().SetType(type);
    }

    virtual int64_t
    MapOutToIn(int64_t out_id) const {
        return GetIdMap().MapOutToIn(out_id);
    }

    virtual const int64_t*
    MapOutToIn(const int64_t* out_ids, size_t count, std::vector<int64_t>& in_ids) const {
        return GetIdMap().MapOutToIn(out_ids, count, in_ids);
    }

    /**
     * Finalizes derived id maps after backend layout is available.
     *
     * Bitmap input carries public validity only. This step builds the
     * internal-to-public and public-to-internal maps used by result mapping and
     * selected-id APIs. EmbList strategies may also derive a base-vector ->
     * public-list map for vector-level filtering.
     */
    virtual Status
    FinalizeIdMap() {
        if (id_map_.InToOutIds().empty()) {
            id_map_.FinalizeVectorIds();
        }
        if (emb_list_offset_ != nullptr && emb_list_strategy_ != nullptr && emb_list_strategy_->NeedsBaseIndexIDMap()) {
            id_map_.FinalizeEmbListIds(emb_list_offset_->offset.data(), emb_list_offset_->num_el());
        }
        return Status::success;
    }

 protected:
    // Retrieve vectors by ids already in this node's base-vector storage domain.
    virtual expected<DataSetPtr>
    GetVectorByStorageIds(const DataSetPtr dataset, milvus::OpContext* op_context = nullptr) const {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByStorageIds not implemented");
    }

    Status
    AppendEmbListOffsetAndIdMap(const size_t* lims, size_t append_row_count) {
        // lims is absolute in the whole index; AppendEmbListIds consumes the
        // appended local offset window.
        const auto old_el_count = emb_list_offset_->num_el();
        const auto old_vector_count = emb_list_offset_->offset.back();
        const auto new_vector_count = old_vector_count + append_row_count;
        if (lims[0] != old_vector_count) {
            LOG_KNOWHERE_WARNING_ << "emb list offset is not continuous";
            return Status::emb_list_inner_error;
        }

        size_t append_el_count = 1;
        while (lims[append_el_count] < new_vector_count) {
            if (lims[append_el_count] < lims[append_el_count - 1]) {
                LOG_KNOWHERE_WARNING_ << "emb list offset is not increasing";
                return Status::emb_list_inner_error;
            }
            ++append_el_count;
        }
        if (lims[append_el_count] != new_vector_count) {
            LOG_KNOWHERE_WARNING_ << "emb list offset should end with the total_cnt of the whole index";
            return Status::emb_list_inner_error;
        }

        std::vector<size_t> append_lims;
        append_lims.reserve(append_el_count + 1);
        append_lims.push_back(0);
        for (size_t i = 1; i <= append_el_count; ++i) {
            append_lims.push_back(lims[i] - old_vector_count);
        }
        emb_list_offset_->offset.reserve(emb_list_offset_->offset.size() + append_el_count);

        if (emb_list_strategy_ != nullptr && emb_list_strategy_->NeedsBaseIndexIDMap()) {
            const auto& ebl_ids = id_map_.InToOutIds(IdMap::Domain::EMB_LIST);
            if (ebl_ids.size() != old_vector_count && ebl_ids.size() != new_vector_count) {
                LOG_KNOWHERE_WARNING_ << "invalid emb_list external id map size: " << ebl_ids.size()
                                      << ", expected: " << old_vector_count << " or " << new_vector_count;
                return Status::invalid_args;
            }
            try {
                if (ebl_ids.size() == old_vector_count) {
                    // Extend the vector -> public-list map for this append batch.
                    id_map_.AppendEmbListIds(static_cast<int64_t>(old_el_count), append_lims.data(),
                                             static_cast<int64_t>(append_el_count));
                }
            } catch (const std::exception& e) {
                LOG_KNOWHERE_WARNING_ << "invalid emb_list external id map: " << e.what();
                return Status::invalid_args;
            }
        }
        for (size_t i = 1; i <= append_el_count; ++i) {
            emb_list_offset_->offset.push_back(lims[i]);
        }
        return Status::success;
    }

    void
    PrepareBitsetMap(PreparedBitset& prepared, const IdMap& id_map) const {
        // Backend selectors test internal ids; bitsets use public ids.
        const auto& ebl_out_ids = id_map.InToOutIds(IdMap::Domain::EMB_LIST);
        const auto& out_ids = ebl_out_ids.empty() ? id_map.InToOutIds() : ebl_out_ids;
        if (!out_ids.empty()) {
            prepared.bitset.set_id_offset(0);
            prepared.bitset.set_out_ids(out_ids, out_ids.size());
        }
    }

    static size_t
    CountBitmapBits(const uint8_t* data, size_t bit_offset, size_t bit_count) {
        // Counts a logical bit range in a packed public-id bitmap.
        if (data == nullptr || bit_count == 0) {
            return 0;
        }

        const auto end_bit = bit_offset + bit_count;
        auto bit_pos = bit_offset;
        size_t count = 0;

        if ((bit_pos & 7) != 0) {
            const auto byte_idx = bit_pos >> 3;
            const auto bits_in_byte = std::min<size_t>(8 - (bit_pos & 7), end_bit - bit_pos);
            const auto mask = static_cast<uint8_t>(((1U << bits_in_byte) - 1) << (bit_pos & 7));
            count += __builtin_popcount(static_cast<unsigned>(data[byte_idx] & mask));
            bit_pos += bits_in_byte;
        }

        const auto full_bytes = (end_bit - bit_pos) >> 3;
        const auto byte_begin = bit_pos >> 3;
        const auto len_u64 = full_bytes >> 3;
        for (size_t i = 0; i < len_u64; ++i) {
            uint64_t bits;
            std::memcpy(&bits, data + byte_begin + i * sizeof(uint64_t), sizeof(bits));
            count += __builtin_popcountll(bits);
        }

        auto byte_pos = byte_begin + len_u64 * sizeof(uint64_t);
        const auto byte_end = byte_begin + full_bytes;
        while (byte_pos < byte_end) {
            count += __builtin_popcount(static_cast<unsigned>(data[byte_pos]));
            ++byte_pos;
        }
        bit_pos += full_bytes << 3;

        if (bit_pos < end_bit) {
            const auto byte_idx = bit_pos >> 3;
            const auto tail_bits = end_bit - bit_pos;
            const auto mask = static_cast<uint8_t>((1U << tail_bits) - 1);
            count += __builtin_popcount(static_cast<unsigned>(data[byte_idx] & mask));
        }

        return count;
    }

    static size_t
    CountBitmapBits(const BitmapArray& data, size_t bit_offset, size_t bit_count) {
        // BitmapArray may be backed by append records, where data() is not a
        // contiguous pointer.
        if (data.empty() || bit_count == 0 || bit_offset >= data.size()) {
            return 0;
        }

        const auto end_bit = bit_offset + std::min(bit_count, data.size() - bit_offset);
        auto bit_pos = bit_offset;
        size_t count = 0;

        if ((bit_pos & 7) != 0) {
            const auto byte_idx = bit_pos >> 3;
            const auto bits_in_byte = std::min<size_t>(8 - (bit_pos & 7), end_bit - bit_pos);
            const auto mask = static_cast<uint8_t>(((1U << bits_in_byte) - 1) << (bit_pos & 7));
            count += __builtin_popcount(static_cast<unsigned>(data[byte_idx] & mask));
            bit_pos += bits_in_byte;
        }

        const auto full_bytes = (end_bit - bit_pos) >> 3;
        const auto byte_begin = bit_pos >> 3;
        const auto len_u64 = full_bytes >> 3;
        for (size_t i = 0; i < len_u64; ++i) {
            uint64_t bits = 0;
            const auto byte_offset = byte_begin + i * sizeof(uint64_t);
            for (size_t byte = 0; byte < sizeof(uint64_t); ++byte) {
                bits |= static_cast<uint64_t>(data[byte_offset + byte]) << (byte * 8);
            }
            count += __builtin_popcountll(bits);
        }

        auto byte_pos = byte_begin + len_u64 * sizeof(uint64_t);
        const auto byte_end = byte_begin + full_bytes;
        while (byte_pos < byte_end) {
            count += __builtin_popcount(static_cast<unsigned>(data[byte_pos]));
            ++byte_pos;
        }
        bit_pos += full_bytes << 3;

        if (bit_pos < end_bit) {
            const auto byte_idx = bit_pos >> 3;
            const auto tail_bits = end_bit - bit_pos;
            const auto mask = static_cast<uint8_t>((1U << tail_bits) - 1);
            count += __builtin_popcount(static_cast<unsigned>(data[byte_idx] & mask));
        }

        return count;
    }

    void
    CalcBitsetCount(BitsetView& bitset, const IdMap& id_map, size_t offset = 0,
                    size_t bitset_count = std::numeric_limits<size_t>::max()) const {
        // Exact filter counts are reported in the backend vector domain.
        if (bitset.num_bits() == 0 || bitset.data() == nullptr) {
            return;
        }

        auto count_to_check = [](size_t total, size_t offset, size_t bitset_count) {
            if (offset >= total) {
                return static_cast<size_t>(0);
            }
            const auto remain = total - offset;
            return bitset_count == std::numeric_limits<size_t>::max() ? remain : std::min(bitset_count, remain);
        };

        const auto& valid_bitmap = id_map.ValidBitmap();
        if (emb_list_offset_ != nullptr) {
            auto bit_is_set = [](const uint8_t* data, size_t bit) { return (data[bit >> 3] & (1U << (bit & 7))) != 0; };

            auto is_valid_out_id = [&](int64_t out_id) { return id_map.IsValidOutId(out_id); };

            auto is_filtered_out_id = [&](int64_t out_id) {
                if (out_id < 0) {
                    return true;
                }
                const auto bit = static_cast<size_t>(out_id);
                return bit >= bitset.num_bits() || bit_is_set(bitset.data(), bit);
            };

            const auto& in_to_out_ids = id_map.InToOutIds();
            auto get_out_el_id = [&](size_t in_el_id) {
                if (in_to_out_ids.empty()) {
                    return static_cast<int64_t>(in_el_id);
                }
                return static_cast<int64_t>(in_to_out_ids[in_el_id]);
            };

            const auto& in_to_out_ebl_ids = id_map.InToOutIds(IdMap::Domain::EMB_LIST);
            const auto in_el_count = emb_list_offset_->num_el();
            if (!in_to_out_ebl_ids.empty()) {
                // Base-vector EmbList backends search vector ids while filters
                // are list-id addressed.
                const auto total_vector_count = emb_list_offset_->offset.back();
                const auto scoped_count = count_to_check(total_vector_count, offset, bitset_count);
                const auto scope_end = offset + scoped_count;
                size_t vector_count = 0;
                size_t filtered_count = 0;
                auto iter = std::upper_bound(emb_list_offset_->offset.begin(), emb_list_offset_->offset.end(), offset);
                size_t in_el_id = iter == emb_list_offset_->offset.begin()
                                      ? 0
                                      : static_cast<size_t>(std::distance(emb_list_offset_->offset.begin(), iter) - 1);
                for (; in_el_id < in_el_count && emb_list_offset_->offset[in_el_id] < scope_end; ++in_el_id) {
                    const auto vector_begin = std::max(emb_list_offset_->offset[in_el_id], offset);
                    const auto vector_end = std::min(emb_list_offset_->offset[in_el_id + 1], scope_end);
                    if (vector_begin >= vector_end) {
                        continue;
                    }
                    const auto out_el_id = static_cast<int64_t>(in_to_out_ebl_ids[vector_begin]);
                    if (!is_valid_out_id(out_el_id)) {
                        continue;
                    }
                    vector_count += vector_end - vector_begin;
                    if (is_filtered_out_id(out_el_id)) {
                        filtered_count += vector_end - vector_begin;
                    }
                }
                bitset.set_vector_count(vector_count);
                bitset.set_filter_count(filtered_count);
                return;
            }

            // Compact-list strategies count by list ranges.
            const auto total_vector_count = emb_list_offset_->offset.back();
            const auto scoped_count = count_to_check(total_vector_count, offset, bitset_count);
            const auto scope_end = offset + scoped_count;
            size_t vector_count = 0;
            size_t filtered_count = 0;
            auto iter = std::upper_bound(emb_list_offset_->offset.begin(), emb_list_offset_->offset.end(), offset);
            size_t in_el_id = iter == emb_list_offset_->offset.begin()
                                  ? 0
                                  : static_cast<size_t>(std::distance(emb_list_offset_->offset.begin(), iter) - 1);
            for (; in_el_id < in_el_count && emb_list_offset_->offset[in_el_id] < scope_end; ++in_el_id) {
                const auto vector_begin = std::max(emb_list_offset_->offset[in_el_id], offset);
                const auto vector_end = std::min(emb_list_offset_->offset[in_el_id + 1], scope_end);
                if (vector_begin >= vector_end) {
                    continue;
                }
                const auto out_el_id = get_out_el_id(in_el_id);
                if (!is_valid_out_id(out_el_id)) {
                    continue;
                }
                vector_count += vector_end - vector_begin;
                if (is_filtered_out_id(out_el_id)) {
                    filtered_count += vector_end - vector_begin;
                }
            }
            bitset.set_vector_count(vector_count);
            bitset.set_filter_count(filtered_count);
            return;
        }

        const auto count = Count();
        const auto total = valid_bitmap.empty() && count > 0 ? static_cast<size_t>(count) : bitset.num_bits();
        const auto scoped_count = count_to_check(total, offset, bitset_count);
        // Vector indexes intersect the public filter with public validity.
        if (valid_bitmap.empty()) {
            bitset.count_filtered_bits(offset, scoped_count);
        } else {
            bitset.count_filtered_bits(offset, scoped_count, valid_bitmap);
        }
        if (offset != 0 || bitset_count != std::numeric_limits<size_t>::max()) {
            return;
        }
        if (!valid_bitmap.empty()) {
            if (valid_bitmap.size() <= bitset.num_bits()) {
                return;
            }
            auto filter_count = bitset.count();
            filter_count += CountBitmapBits(valid_bitmap, bitset.num_bits(), valid_bitmap.size() - bitset.num_bits());
            bitset.set_vector_count(count > 0 ? static_cast<size_t>(count) : bitset.size());
            bitset.set_filter_count(std::min(filter_count, bitset.size()));
            return;
        }
        if (count > 0 && static_cast<size_t>(count) > bitset.num_bits()) {
            auto filter_count = bitset.count();
            bitset.set_vector_count(static_cast<size_t>(count));
            bitset.set_filter_count(
                std::min(filter_count + static_cast<size_t>(count) - bitset.num_bits(), bitset.size()));
        }
    }

    // Base ANN result id map. Base-vector EmbList stages keep compact ids until
    // final list mapping.
    const IdMap*
    SearchResultIdMap(const IdMap& id_map) const {
        if (emb_list_offset_ != nullptr && emb_list_strategy_ != nullptr && emb_list_strategy_->NeedsBaseIndexIDMap()) {
            return nullptr;
        }
        return &id_map;
    }

    void
    MapResultIdsToOutIds(const DataSetPtr& result, const IdMap* id_map) const {
        // Backend storage ids -> public result ids.
        if (id_map == nullptr || result == nullptr || result->GetIds() == nullptr) {
            return;
        }
        auto* ids = const_cast<int64_t*>(result->GetIds());
        const auto* lims = result->GetLims();
        const auto rows = result->GetRows();
        const auto count = lims != nullptr ? lims[rows] : static_cast<size_t>(rows * result->GetDim());
        id_map->MapInToOut(ids, count);
    }

    void
    MapSearchResultIdsToOutIds(const DataSetPtr& result) const {
        const auto& id_map = GetIdMap();
        MapResultIdsToOutIds(result, SearchResultIdMap(id_map));
    }

    void
    MapEmbListResultIdsToOutIds(const DataSetPtr& result) const {
        // Compact list ids -> public list ids.
        const auto& id_map = GetIdMap();
        MapResultIdsToOutIds(result, &id_map);
    }

 public:
    virtual std::string
    Type() const = 0;

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    struct LatencyMetricCache {
        mutable std::once_flag once;
        mutable prometheus::Histogram* metric{nullptr};
    };

    prometheus::Histogram&
    GetLatencyMetric(LatencyMetricCache& cache, prometheus::Family<prometheus::Histogram>& family) const {
        std::call_once(cache.once,
                       [this, &cache, &family] { cache.metric = &GetPrometheusHistogram(family, "knowhere", Type()); });
        return *cache.metric;
    }

    prometheus::Histogram&
    GetBuildLatencyMetric() const {
        return GetLatencyMetric(prometheus_metrics_.build, build_latency_family);
    }

    prometheus::Histogram&
    GetLoadLatencyMetric() const {
        return GetLatencyMetric(prometheus_metrics_.load, load_latency_family);
    }

    prometheus::Histogram&
    GetSearchLatencyMetric() const {
        return GetLatencyMetric(prometheus_metrics_.search, search_latency_family);
    }

    prometheus::Histogram&
    GetRangeSearchLatencyMetric() const {
        return GetLatencyMetric(prometheus_metrics_.range_search, range_search_latency_family);
    }
#endif

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
        if (!emb_list_strategy_) {
            // not emb_list, use the default serialize method
            return Serialize(binset);
        }

        return SerializeEmbList(binset);
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
        auto& cfg = static_cast<knowhere::BaseConfig&>(*config);
        auto el_metric_type_or = get_el_metric_type(cfg.metric_type.value());
        if (!el_metric_type_or.has_value()) {
            // not emb_list, use the default deserialize method
            return Deserialize(binset, config);
        }

        return DeserializeEmbListFromBinarySet(binset, config);
    }

    virtual bool
    LoadIndexWithStream() {
        return false;
    }

    virtual ~IndexNode() {
    }

    virtual Status
    DeserializeFromFileIfNeed(const std::string& filename, std::shared_ptr<Config> config) {
        auto& cfg = static_cast<knowhere::BaseConfig&>(*config);
        auto el_metric_type_or = get_el_metric_type(cfg.metric_type.value());
        if (!el_metric_type_or.has_value()) {
            // not emb_list, use the default deserialize method
            return DeserializeFromFile(filename, config);
        }

        return DeserializeEmbListFromFile(filename, config);
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

    virtual expected<DataSetPtr>
    RangeSearchEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                             milvus::OpContext* op_context = nullptr) const;

    virtual expected<std::vector<IteratorPtr>>
    AnnIteratorEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                             bool use_knowhere_search_pool = true, milvus::OpContext* op_context = nullptr) const;

    /**
     * @brief Retrieve raw vectors for public embedding-list ids.
     *
     * The result tensor is flattened and accompanied by EMB_LIST_OFFSET.
     */
    virtual expected<DataSetPtr>
    GetEmbListByIds(const DataSetPtr dataset, const std::string& metric_type,
                    milvus::OpContext* op_context = nullptr) const;

 protected:
    /**
     * @brief Parse EMB_LIST_META header from raw bytes.
     *
     * Detects format by magic number:
     * - New format: [int64_t magic][size_t type_len][char[type_len] type][uint8_t[] strategy_blob]
     * - Legacy format (tokenann only): entire blob is [size_t count][size_t[count] offsets]
     */
    struct EmbListMetaHeader {
        std::string strategy_type;
        const uint8_t* strategy_blob;
        int64_t strategy_blob_size;
    };

    virtual Status
    BuildEmbList(const DataSetPtr dataset, std::shared_ptr<Config> cfg, const size_t* lims, size_t num_rows,
                 bool use_knowhere_build_pool);

    /**
     * @brief Serialize emb_list: strategy meta, raw index, and base index to BinarySet.
     */
    Status
    SerializeEmbList(BinarySet& binset) const;

    /**
     * @brief Deserialize emb_list: base index, strategy, and optional raw index from BinarySet.
     */
    Status
    DeserializeEmbListFromBinarySet(const BinarySet& binset, std::shared_ptr<Config> config);

    /**
     * @brief Deserialize emb_list: base index, strategy, and optional raw index from files.
     */
    Status
    DeserializeEmbListFromFile(const std::string& filename, std::shared_ptr<Config> config);

    /**
     * @brief Search embedding-list queries through the selected strategy.
     *
     * Input filters are public list-id addressed. Strategy results are compact
     * list ids until this boundary maps them back to public ids.
     */
    virtual expected<DataSetPtr>
    SearchEmbList(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                  milvus::OpContext* op_context = nullptr) const;

    static EmbListMetaHeader
    ParseEmbListMetaHeader(const uint8_t* data, int64_t size);

 protected:
    /**
     * @brief Compute distances using emb_list_raw_index_ (raw vector storage).
     *
     * Used by CalcDistByIDs implementations when emb_list_raw_index_ is present
     * (MUVERA/LEMUR strategies). The raw index stores original vectors indexed by
     * global vector IDs, so no ID translation is needed.
     *
     * @param pool Thread pool for parallel computation
     * @return Distance results or error
     */
    expected<DataSetPtr>
    CalcDistByRawIndex(const DataSetPtr dataset, const int64_t* labels, size_t labels_len, bool is_cosine,
                       std::shared_ptr<ThreadPool> pool, milvus::OpContext* op_context = nullptr) const;

    Version version_;
    std::shared_ptr<EmbListOffset> emb_list_offset_;  // emb_list group offset structure (shared with strategy)
    IdMap id_map_;
    std::string el_metric_type_;
    EmbListStrategyPtr emb_list_strategy_;  // emb_list encoding strategy (tokenann/muvera)
    // Raw vector storage for EmbList strategies whose ANN payload is encoded.
    // Distance rerank uses original vectors by compact base-vector id.
    std::shared_ptr<::faiss::IndexFlat> emb_list_raw_index_;

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    struct PrometheusMetrics {
        LatencyMetricCache build;
        LatencyMetricCache load;
        LatencyMetricCache search;
        LatencyMetricCache range_search;
    };
    mutable PrometheusMetrics prometheus_metrics_;
#endif
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

    void
    SetResultIdMap(const IdMap* id_map) {
        // Index-owned backend id -> public id map.
        result_id_map_ = id_map;
    }

    expected<std::pair<int64_t, float>>
    Next() noexcept final {
        return GuardedCall([&]() -> expected<std::pair<int64_t, float>> { return next_impl(); });
    }

    [[nodiscard]] expected<bool>
    HasNext() noexcept final {
        return GuardedCall([&]() -> expected<bool> { return has_next_impl(); });
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
    // The *_impl methods may throw or return an error result; exceptions are converted to
    //   error codes by GuardedCall at the public Next()/HasNext() boundary.
    virtual expected<std::pair<int64_t, float>>
    next_impl() {
        if (!initialized_) {
            initialize();
        }
        auto& q = !refine_ ? res_ : refined_res_;
        if (q.empty()) {
            return expected<std::pair<int64_t, float>>::Err(Status::knowhere_inner_error, "No more elements");
        }
        auto ret = q.top();
        q.pop();

        auto update_next_func = [&]() {
            UpdateNext();
            if (retain_iterator_order_) {
                while (!res_.empty() || !refined_res_.empty()) {
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

        return std::make_pair(MapResultId(ret.id), ret.val * sign_);
    }

    virtual expected<bool>
    has_next_impl() {
        if (!initialized_) {
            initialize();
        }
        return !res_.empty() || !refined_res_.empty();
    }

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

    int64_t
    MapResultId(int64_t id) const {
        return result_id_map_ == nullptr ? id : result_id_map_->MapInToOut(id);
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
    const IdMap* result_id_map_ = nullptr;
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

    void
    SetResultIdMap(const IdMap* id_map) {
        // Index-owned backend id -> public id map.
        result_id_map_ = id_map;
    }

    expected<std::pair<int64_t, float>>
    Next() noexcept override {
        return GuardedCall([&]() -> expected<std::pair<int64_t, float>> {
            if (!initialized_) {
                initialize();
            }
            if (!has_next_unchecked()) {
                return expected<std::pair<int64_t, float>>::Err(Status::knowhere_inner_error, "No more elements");
            }
            // sort_next() is a no-op unless the cursor has caught up with the
            // sorted prefix, which happens once every sort_size_ elements
            // (>= 50000). Checking that here rather than inside the task keeps
            // the common case from paying for a thread-pool round trip and a
            // blocking wait just to return immediately.
            if (needs_sort()) {
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
            }
            auto& result = results_[next_++];
            return std::make_pair(MapResultId(result.id), result.val);
        });
    }

    [[nodiscard]] expected<bool>
    HasNext() noexcept override {
        return GuardedCall([&]() -> expected<bool> {
            if (!initialized_) {
                initialize();
            }
            return has_next_unchecked();
        });
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
    bool
    has_next_unchecked() const {
        return next_ < results_.size() && results_[next_].id != -1;
    }

    static inline size_t
    get_sort_size(size_t rows) {
        return std::max((size_t)50000, rows / 10);
    }

    // Whether the cursor has caught up with the sorted prefix and sort_next()
    // would do real work.
    inline bool
    needs_sort() const {
        return next_ >= sorted_;
    }

    // sort the next sort_size_ elements
    inline void
    sort_next() {
        if (!needs_sort()) {
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

    int64_t
    MapResultId(int64_t id) const {
        return result_id_map_ == nullptr ? id : result_id_map_->MapInToOut(id);
    }

    std::function<std::vector<DistId>()> compute_dist_func_;
    const bool larger_is_closer_;
    bool use_knowhere_search_pool_ = true;
    bool initialized_ = false;
    std::vector<DistId> results_;
    size_t next_ = 0;
    size_t sorted_ = 0;
    size_t sort_size_ = 0;
    const IdMap* result_id_map_ = nullptr;
};

}  // namespace knowhere

#endif /* INDEX_NODE_H */
