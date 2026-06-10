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

#ifndef EXTERNAL_ID_MAP_H
#define EXTERNAL_ID_MAP_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "knowhere/bitsetview.h"

namespace knowhere {

struct ExternalIdMap {
    // State.
    void
    Clear() {
        internal_to_external_ids_.clear();
        internal_id_to_emb_list_id_.clear();
        external_to_internal_ids_.clear();
        valid_bitmap_.clear();
        external_count_ = 0;
        external_id_offset_ = 0;
    }

    bool
    HasInternalToExternalIds() const {
        return !internal_to_external_ids_.empty();
    }

    bool
    HasInternalToEmbListIds() const {
        return !internal_id_to_emb_list_id_.empty();
    }

    bool
    HasResultIdMap() const {
        return HasInternalToEmbListIds() || HasInternalToExternalIds() || HasExternalIdOffset();
    }

    int64_t
    ExternalCount(int64_t default_count) const {
        return external_count_ == 0 ? default_count : static_cast<int64_t>(external_count_);
    }

    ValidBitmapView
    GetValidBitmapView() const {
        if (valid_bitmap_.empty()) {
            return {};
        }
        return {valid_bitmap_.data(), external_count_};
    }

    const std::vector<int32_t>&
    GetInternalToExternalIds() const {
        return internal_to_external_ids_;
    }

    const std::vector<int32_t>&
    GetInternalToEmbListIds() const {
        return internal_id_to_emb_list_id_;
    }

    // Mapping construction.
    void
    SetExternalIdOffset(int64_t external_id_offset) {
        external_id_offset_ = external_id_offset;
    }

    void
    SetInternalToExternalIds(const int32_t* ids, int64_t internal_count, int64_t external_count) {
        Clear();
        if (internal_count < 0 || external_count <= 0 || internal_count > external_count ||
            external_count > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("invalid external id map size");
        }
        if (ids == nullptr && internal_count != 0) {
            throw std::runtime_error("invalid external id map data");
        }
        external_count_ = static_cast<size_t>(external_count);
        internal_to_external_ids_.reserve(static_cast<size_t>(internal_count));
        external_to_internal_ids_.reserve(static_cast<size_t>(internal_count));
        for (int64_t i = 0; i < internal_count; ++i) {
            AddInternalToExternalId(ids[i], static_cast<int32_t>(i));
        }
        BuildValidBitmap();
    }

    void
    SetInternalToExternalIds(std::vector<int32_t> ids, size_t external_count) {
        Clear();
        if (external_count == 0 || ids.size() > external_count ||
            external_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("invalid external id map size");
        }
        external_count_ = external_count;
        internal_to_external_ids_ = std::move(ids);
        RebuildInternalToExternalDerivedData();
    }

    void
    SetInternalToEmbListIds(std::vector<int32_t> ids) {
        internal_id_to_emb_list_id_ = std::move(ids);
    }

    void
    AppendInternalToEmbListIds(size_t internal_id_begin, size_t internal_id_end, size_t emb_list_id) {
        if (internal_id_to_emb_list_id_.size() < internal_id_end) {
            internal_id_to_emb_list_id_.resize(internal_id_end);
        }
        std::fill(
            internal_id_to_emb_list_id_.begin() + internal_id_begin,
            internal_id_to_emb_list_id_.begin() + internal_id_end,
            ToExternalId(emb_list_id)
        );
    }

    // Incremental append for mutable indexes.
    void
    AddInternalToExternalIds(
        const int32_t* ids, int64_t internal_count, int64_t external_count, int64_t internal_id_begin
    ) {
        if (internal_count < 0 || external_count < 0 || internal_id_begin < 0 ||
            external_count > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("invalid external id map size");
        }
        if (ids == nullptr && internal_count != 0 && external_count != 0) {
            throw std::runtime_error("invalid external id map data");
        }

        if (internal_id_begin == 0) {
            if (ids == nullptr && external_count == 0) {
                Clear();
                return;
            }
            SetInternalToExternalIds(ids, internal_count, external_count);
            return;
        }

        if (ids == nullptr && external_count == 0 && !HasInternalToExternalIds() && external_count_ == 0) {
            return;
        }

        auto id_begin = static_cast<size_t>(internal_id_begin);
        auto count = static_cast<size_t>(internal_count);
        if (internal_to_external_ids_.empty() && external_count_ == 0) {
            external_count_ = external_count == 0 ? id_begin + count : static_cast<size_t>(external_count);
            if (external_count_ > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
                id_begin > external_count_) {
                throw std::runtime_error("invalid external id map size");
            }
            internal_to_external_ids_.reserve(id_begin + count);
            external_to_internal_ids_.reserve(id_begin + count);
            for (size_t i = 0; i < id_begin; ++i) {
                AddInternalToExternalId(static_cast<int32_t>(i), static_cast<int32_t>(i));
            }
        } else if (internal_to_external_ids_.size() != id_begin) {
            throw std::runtime_error("external id map append position mismatch");
        }

        auto external_id_begin = external_count_;
        if (external_count != 0) {
            if (static_cast<size_t>(external_count) < external_count_) {
                throw std::runtime_error("invalid external id map size");
            }
            external_count_ = static_cast<size_t>(external_count);
        } else if (count != 0) {
            if (external_count_ > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - count) {
                throw std::runtime_error("invalid external id map size");
            }
            external_count_ += count;
        }

        internal_to_external_ids_.reserve(id_begin + count);
        external_to_internal_ids_.reserve(id_begin + count);
        if (ids != nullptr) {
            for (size_t i = 0; i < count; ++i) {
                AddInternalToExternalId(ids[i], static_cast<int32_t>(id_begin + i));
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                AddInternalToExternalId(static_cast<int32_t>(external_id_begin + i),
                                        static_cast<int32_t>(id_begin + i));
            }
        }
        BuildValidBitmap();
    }

    // Relayout.
    template <typename SourceIdGetter>
    void
    ApplyInternalToExternalRelayout(size_t internal_count, SourceIdGetter source_id_getter) {
        std::vector<int32_t> remapped_ids(internal_count);
        for (size_t i = 0; i < internal_count; ++i) {
            remapped_ids[i] = ToExternalId(source_id_getter(i));
        }
        if (external_count_ == 0 && !remapped_ids.empty()) {
            const auto max_external_id = *std::max_element(remapped_ids.begin(), remapped_ids.end());
            if (max_external_id < 0) {
                throw std::runtime_error("external id map points outside external rows");
            }
            external_count_ = static_cast<size_t>(max_external_id) + 1;
        }
        internal_to_external_ids_ = std::move(remapped_ids);
        RebuildInternalToExternalDerivedData();
    }

    template <typename SourceIdGetter>
    void
    ApplyInternalToEmbListRelayout(size_t internal_count, SourceIdGetter source_id_getter) {
        if (!HasInternalToEmbListIds()) {
            return;
        }
        std::vector<int32_t> remapped_ids(internal_count);
        for (size_t i = 0; i < internal_count; ++i) {
            remapped_ids[i] = internal_id_to_emb_list_id_[source_id_getter(i)];
        }
        internal_id_to_emb_list_id_ = std::move(remapped_ids);
    }

    // Filtering.
    void
    SetOutIdsToBitset(BitsetView& bitset, size_t num_internal_ids = 0,
                      std::optional<size_t> num_filtered_out_ids = std::nullopt,
                      size_t internal_id_offset = 0) const {
        if (HasInternalToEmbListIds()) {
            bitset.set_id_offset(internal_id_offset);
            bitset.set_out_ids(
                internal_id_to_emb_list_id_.data(),
                num_internal_ids == 0 ? internal_id_to_emb_list_id_.size() : num_internal_ids,
                num_filtered_out_ids
            );
            return;
        }
        if (HasInternalToExternalIds()) {
            bitset.set_id_offset(internal_id_offset);
            bitset.set_out_ids(
                internal_to_external_ids_.data(),
                num_internal_ids == 0 ? internal_to_external_ids_.size() : num_internal_ids,
                num_filtered_out_ids
            );
            return;
        }
        const auto id_offset = external_id_offset_ + static_cast<int64_t>(internal_id_offset);
        if (id_offset >= 0) {
            bitset.set_id_offset(static_cast<size_t>(id_offset));
        }
    }

    // ID conversion.
    int32_t
    ToExternalId(size_t internal_id) const {
        if (!HasInternalToExternalIds()) {
            return static_cast<int32_t>(static_cast<int64_t>(internal_id) + external_id_offset_);
        }
        return internal_to_external_ids_[internal_id];
    }

    int64_t
    ToInternalId(int64_t external_id) const {
        if (!HasInternalToExternalIds()) {
            auto internal_id = external_id - external_id_offset_;
            return internal_id < 0 ? -1 : internal_id;
        }
        if (external_id < 0 || external_id > std::numeric_limits<int32_t>::max()) {
            return -1;
        }
        auto iter = external_to_internal_ids_.find(static_cast<int32_t>(external_id));
        return iter == external_to_internal_ids_.end() ? -1 : iter->second;
    }

    template <typename IdType>
    const IdType*
    ToInternalIds(const IdType* external_ids, size_t count, std::vector<IdType>& internal_ids) const {
        if (!HasInternalToExternalIds()) {
            if (!HasExternalIdOffset()) {
                return external_ids;
            }
            internal_ids.resize(count);
            for (size_t i = 0; i < count; ++i) {
                const auto internal_id = static_cast<int64_t>(external_ids[i]) - external_id_offset_;
                internal_ids[i] = internal_id < 0 ? static_cast<IdType>(-1) : static_cast<IdType>(internal_id);
            }
            return internal_ids.data();
        }
        internal_ids.resize(count);
        for (size_t i = 0; i < count; ++i) {
            const auto external_id = static_cast<int64_t>(external_ids[i]);
            if (external_id < 0 || external_id > std::numeric_limits<int32_t>::max()) {
                internal_ids[i] = static_cast<IdType>(-1);
                continue;
            }
            auto iter = external_to_internal_ids_.find(static_cast<int32_t>(external_id));
            internal_ids[i] =
                iter == external_to_internal_ids_.end() ? static_cast<IdType>(-1) : static_cast<IdType>(iter->second);
        }
        return internal_ids.data();
    }

    template <typename IdType>
    IdType
    MapResultId(IdType id) const {
        if constexpr (std::is_signed_v<IdType>) {
            if (id < 0) {
                return id;
            }
        }
        return static_cast<IdType>(ToResultId(static_cast<size_t>(id)));
    }

    template <typename IdType>
    void
    MapResultIds(IdType* ids, size_t count) const {
        if (ids == nullptr || !HasResultIdMap()) {
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            ids[i] = MapResultId(ids[i]);
        }
    }

    template <typename IdType>
    void
    MapResultIds(std::vector<IdType>& ids) const {
        if (!HasResultIdMap()) {
            return;
        }
        for (auto& id : ids) {
            id = MapResultId(id);
        }
    }

    template <typename IdType>
    void
    MapResultIds(std::vector<std::vector<IdType>>& ids) const {
        if (!HasResultIdMap()) {
            return;
        }
        for (auto& row_ids : ids) {
            MapResultIds(row_ids);
        }
    }

    // Serialization. Only vector-level external id mapping is persisted.
    size_t
    BinarySize() const {
        return 2 * sizeof(uint64_t) + internal_to_external_ids_.size() * sizeof(int32_t);
    }

    void
    Serialize(uint8_t* data, size_t size) const {
        if (data == nullptr || size < BinarySize()) {
            throw std::runtime_error("external id map binary buffer is too small");
        }
        auto* ptr = data;
        auto wire_external_count = static_cast<uint64_t>(external_count_);
        auto wire_map_size = static_cast<uint64_t>(internal_to_external_ids_.size());
        std::memcpy(ptr, &wire_external_count, sizeof(uint64_t));
        ptr += sizeof(uint64_t);
        std::memcpy(ptr, &wire_map_size, sizeof(uint64_t));
        ptr += sizeof(uint64_t);
        std::memcpy(ptr, internal_to_external_ids_.data(), internal_to_external_ids_.size() * sizeof(int32_t));
    }

    void
    Deserialize(const uint8_t* data, int64_t size) {
        Clear();
        if (data == nullptr) {
            return;
        }
        if (size < static_cast<int64_t>(2 * sizeof(uint64_t))) {
            throw std::runtime_error("invalid external id map binary size");
        }

        auto* ptr = data;
        uint64_t wire_external_count = 0;
        uint64_t wire_map_size = 0;
        std::memcpy(&wire_external_count, ptr, sizeof(uint64_t));
        ptr += sizeof(uint64_t);
        std::memcpy(&wire_map_size, ptr, sizeof(uint64_t));
        ptr += sizeof(uint64_t);

        auto expected_size = 2 * sizeof(uint64_t) + wire_map_size * sizeof(int32_t);
        if (static_cast<uint64_t>(size) < expected_size) {
            throw std::runtime_error("invalid external id map binary size");
        }

        std::vector<int32_t> ids(wire_map_size);
        std::memcpy(ids.data(), ptr, wire_map_size * sizeof(int32_t));
        SetInternalToExternalIds(std::move(ids), static_cast<size_t>(wire_external_count));
    }

 private:
    bool
    HasExternalIdOffset() const {
        return external_id_offset_ != 0;
    }

    int32_t
    ToResultId(size_t internal_id) const {
        if (HasInternalToEmbListIds()) {
            return internal_id_to_emb_list_id_[internal_id];
        }
        return ToExternalId(internal_id);
    }

    void
    RebuildInternalToExternalDerivedData() {
        external_to_internal_ids_.clear();
        valid_bitmap_.clear();
        external_to_internal_ids_.reserve(internal_to_external_ids_.size());
        for (size_t i = 0; i < internal_to_external_ids_.size(); ++i) {
            AddExternalToInternalId(internal_to_external_ids_[i], static_cast<int32_t>(i));
        }
        BuildValidBitmap();
    }

    void
    AddInternalToExternalId(int32_t external_id, int32_t internal_id) {
        AddExternalToInternalId(external_id, internal_id);
        internal_to_external_ids_.push_back(external_id);
    }

    void
    AddExternalToInternalId(int32_t external_id, int32_t internal_id) {
        if (external_id < 0 || static_cast<size_t>(external_id) >= external_count_) {
            throw std::runtime_error("external id map points outside external rows");
        }
        auto insert_result = external_to_internal_ids_.emplace(external_id, internal_id);
        if (!insert_result.second) {
            throw std::runtime_error("external id map contains duplicate external id");
        }
    }

    void
    BuildValidBitmap() {
        valid_bitmap_.assign((external_count_ + 7) / 8, 0);
        for (auto external_id : internal_to_external_ids_) {
            valid_bitmap_[external_id >> 3] |= static_cast<uint8_t>(1U << (external_id & 7));
        }
    }

    std::vector<int32_t> internal_to_external_ids_;
    std::vector<int32_t> internal_id_to_emb_list_id_;
    std::unordered_map<int32_t, int32_t> external_to_internal_ids_;
    std::vector<uint8_t> valid_bitmap_;
    size_t external_count_ = 0;
    int64_t external_id_offset_ = 0;
};

inline const ExternalIdMap&
EmptyExternalIdMap() {
    static const ExternalIdMap empty_map;
    return empty_map;
}

}  // namespace knowhere

#endif /* EXTERNAL_ID_MAP_H */
