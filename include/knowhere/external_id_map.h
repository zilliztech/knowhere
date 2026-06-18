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
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "knowhere/bitsetview.h"

namespace knowhere {

struct ValidBitmapView {
    const uint8_t* data = nullptr;
    size_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> owner;

    ValidBitmapView() = default;

    ValidBitmapView(const std::vector<uint8_t>& valid_bitmap, size_t external_count, bool copy_data) {
        size = external_count;
        if (valid_bitmap.empty()) {
            return;
        }
        if (copy_data) {
            owner = std::make_shared<std::vector<uint8_t>>(valid_bitmap);
            data = owner->data();
            return;
        }
        data = valid_bitmap.data();
    }

    bool
    empty() const {
        return data == nullptr || size == 0;
    }
};

struct IdArrayView {
    const int32_t* data = nullptr;
    size_t size = 0;
    std::shared_ptr<const std::vector<int32_t>> owner;

    IdArrayView() = default;

    IdArrayView(const std::vector<int32_t>& ids, bool copy_data) {
        if (copy_data) {
            owner = std::make_shared<std::vector<int32_t>>(ids);
            data = owner->data();
            size = owner->size();
            return;
        }
        data = ids.data();
        size = ids.size();
    }
};

// Runtime-only logical-id mapping. Knowhere never serializes this structure into
// index BinarySet payloads; callers persist nullable/valid-row state externally,
// then configure the map exposed by Index/IndexNode before build or load.
struct ExternalIdMap {
 private:
    static constexpr int32_t kInvalidId = -1;
    static constexpr double kDenseExternalToInternalMinRatio = 0.10;

    struct ExternalToInternalStorage {
        void
        AssignVector(size_t size, int32_t default_value) {
            use_map_ = false;
            id_map_ = {};
            ids_.assign(size, default_value);
        }

        void
        AssignMap(size_t reserve_size) {
            use_map_ = true;
            ids_ = {};
            id_map_ = {};
            id_map_.reserve(reserve_size);
        }

        int32_t
        Get(int64_t id) const {
            if (id < 0 || id > std::numeric_limits<int32_t>::max()) {
                return kInvalidId;
            }
            if (use_map_) {
                auto iter = id_map_.find(static_cast<int32_t>(id));
                return iter == id_map_.end() ? kInvalidId : iter->second;
            }
            const auto offset = static_cast<size_t>(id);
            return offset < ids_.size() ? ids_[offset] : kInvalidId;
        }

        void
        Set(int32_t id, int32_t value) {
            if (id < 0) {
                throw std::runtime_error("external id map contains negative id");
            }
            if (use_map_) {
                if (value == kInvalidId) {
                    return;
                }
                auto insert_result = id_map_.emplace(id, value);
                if (!insert_result.second) {
                    throw std::runtime_error("external id map contains duplicate id");
                }
                return;
            }
            const auto offset = static_cast<size_t>(id);
            if (ids_.size() <= offset) {
                ids_.resize(offset + 1, kInvalidId);
            }
            if (ids_[offset] != kInvalidId) {
                throw std::runtime_error("external id map contains duplicate id");
            }
            ids_[offset] = value;
        }

        void
        Add(int32_t id, int32_t value) {
            if (!use_map_) {
                std::unordered_map<int32_t, int32_t> id_map;
                id_map.reserve(ids_.size());
                for (size_t external_id = 0; external_id < ids_.size(); ++external_id) {
                    if (ids_[external_id] != kInvalidId) {
                        id_map.emplace(static_cast<int32_t>(external_id), ids_[external_id]);
                    }
                }
                ids_ = {};
                id_map_ = std::move(id_map);
                use_map_ = true;
            }
            Set(id, value);
        }

        std::vector<int32_t> ids_;
        std::unordered_map<int32_t, int32_t> id_map_;
        bool use_map_ = false;
    };

    template <typename Func>
    decltype(auto)
    Read(Func&& func) const {
        if (!use_lock_) {
            return std::forward<Func>(func)();
        }
        std::shared_lock lock(mutex_);
        return std::forward<Func>(func)();
    }

    template <typename Func>
    decltype(auto)
    Write(Func&& func) {
        if (!use_lock_) {
            return std::forward<Func>(func)();
        }
        std::unique_lock lock(mutex_);
        return std::forward<Func>(func)();
    }

 public:
    explicit ExternalIdMap(bool use_lock = false) : use_lock_(use_lock) {
    }

    ExternalIdMap(const ExternalIdMap& other) {
        other.Read([&] {
            use_lock_ = other.use_lock_;
            internal_to_external_ids_ = other.internal_to_external_ids_;
            internal_id_to_emb_list_id_ = other.internal_id_to_emb_list_id_;
            external_to_internal_ids_ = other.external_to_internal_ids_;
            valid_bitmap_ = other.valid_bitmap_;
            external_count_ = other.external_count_;
        });
    }

    ExternalIdMap(ExternalIdMap&& other) {
        other.Write([&] {
            use_lock_ = other.use_lock_;
            internal_to_external_ids_ = std::move(other.internal_to_external_ids_);
            internal_id_to_emb_list_id_ = std::move(other.internal_id_to_emb_list_id_);
            external_to_internal_ids_ = std::move(other.external_to_internal_ids_);
            valid_bitmap_ = std::move(other.valid_bitmap_);
            external_count_ = other.external_count_;
        });
    }

    ExternalIdMap&
    operator=(const ExternalIdMap& other) {
        if (this == &other) {
            return *this;
        }
        ExternalIdMap snapshot(other);
        Write([&] {
            use_lock_ = snapshot.use_lock_;
            internal_to_external_ids_ = snapshot.internal_to_external_ids_;
            internal_id_to_emb_list_id_ = snapshot.internal_id_to_emb_list_id_;
            external_to_internal_ids_ = snapshot.external_to_internal_ids_;
            valid_bitmap_ = snapshot.valid_bitmap_;
            external_count_ = snapshot.external_count_;
        });
        return *this;
    }

    ExternalIdMap&
    operator=(ExternalIdMap&& other) {
        if (this == &other) {
            return *this;
        }
        ExternalIdMap snapshot(std::move(other));
        Write([&] {
            use_lock_ = snapshot.use_lock_;
            internal_to_external_ids_ = std::move(snapshot.internal_to_external_ids_);
            internal_id_to_emb_list_id_ = std::move(snapshot.internal_id_to_emb_list_id_);
            external_to_internal_ids_ = std::move(snapshot.external_to_internal_ids_);
            valid_bitmap_ = std::move(snapshot.valid_bitmap_);
            external_count_ = snapshot.external_count_;
        });
        return *this;
    }

    int64_t
    GetExternalCount() const {
        return Read([&] { return static_cast<int64_t>(external_count_); });
    }

    ValidBitmapView
    GetValidBitmap(std::optional<bool> copy_data = std::nullopt) const {
        return Read([&]() -> ValidBitmapView {
            if (valid_bitmap_.empty()) {
                return {};
            }
            return {valid_bitmap_, external_count_, copy_data.value_or(use_lock_)};
        });
    }

    IdArrayView
    GetInternalToExternalIds(std::optional<bool> copy_data = std::nullopt) const {
        return Read([&]() -> IdArrayView { return {internal_to_external_ids_, copy_data.value_or(use_lock_)}; });
    }

    IdArrayView
    GetInternalToEmbListIds(std::optional<bool> copy_data = std::nullopt) const {
        return Read([&]() -> IdArrayView { return {internal_id_to_emb_list_id_, copy_data.value_or(use_lock_)}; });
    }

    IdArrayView
    GetInternalToOutputIds(std::optional<bool> copy_data = std::nullopt) const {
        auto emb_list_ids = GetInternalToEmbListIds(copy_data);
        if (emb_list_ids.size != 0) {
            return emb_list_ids;
        }
        return GetInternalToExternalIds(copy_data);
    }

    int32_t
    GetExternalToInternalId(int64_t external_id) const {
        return Read([&] { return external_to_internal_ids_.Get(external_id); });
    }

    int64_t
    MapInternalIdToExternalId(const IdArrayView& ids, int64_t internal_id) const {
        if (internal_id < 0 || ids.size == 0) {
            return internal_id;
        }
        const auto offset = static_cast<size_t>(internal_id);
        return offset < ids.size ? ids.data[offset] : kInvalidId;
    }

    template <typename IdType>
    void
    MapInternalIdsToExternalIds(IdType* ids, size_t count) const {
        if (ids == nullptr) {
            return;
        }
        auto id_map = GetInternalToOutputIds();
        if (id_map.size == 0) {
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            ids[i] = static_cast<IdType>(MapInternalIdToExternalId(id_map, ids[i]));
        }
    }

    template <typename IdType>
    void
    MapInternalIdsToExternalIds(std::vector<std::vector<IdType>>& ids) const {
        auto id_map = GetInternalToOutputIds();
        if (id_map.size == 0) {
            return;
        }
        for (auto& row_ids : ids) {
            for (auto& id : row_ids) {
                id = static_cast<IdType>(MapInternalIdToExternalId(id_map, id));
            }
        }
    }

    int64_t
    MapExternalIdToInternalId(int64_t external_id) const {
        if (GetExternalCount() == 0) {
            return external_id;
        }
        return GetExternalToInternalId(external_id);
    }

    const int64_t*
    MapExternalIdsToInternalIds(const int64_t* ids, size_t count, std::vector<int64_t>& internal_ids) const {
        if (GetExternalCount() == 0) {
            return ids;
        }
        internal_ids.resize(count);
        for (size_t i = 0; i < count; ++i) {
            internal_ids[i] = GetExternalToInternalId(ids[i]);
        }
        return internal_ids.data();
    }

    void
    SetInternalToExternalIds(const int32_t* ids, int64_t internal_count, int64_t external_count) {
        Write([&] {
            if (internal_count < 0 || external_count <= 0 || internal_count > external_count ||
                external_count > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("invalid external id map size");
            }
            const auto internal_count_value = static_cast<size_t>(internal_count);
            external_count_ = static_cast<size_t>(external_count);
            internal_to_external_ids_.clear();
            if (internal_count_value != 0) {
                if (ids == nullptr) {
                    throw std::runtime_error("invalid external id map data");
                }
                internal_to_external_ids_.assign(ids, ids + internal_count_value);
            }
        });
    }

    void
    SetExternalToInternalIds(const int32_t* external_ids, int64_t internal_count, int64_t external_count) {
        Write([&] {
            if (internal_count < 0 || external_count <= 0 || internal_count > external_count ||
                external_count > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("invalid external id map size");
            }
            const auto internal_count_value = static_cast<size_t>(internal_count);
            const auto external_count_value = static_cast<size_t>(external_count);
            external_count_ = external_count_value;
            const auto valid_ratio =
                static_cast<double>(internal_count_value) / static_cast<double>(external_count_value);
            if (valid_ratio <= kDenseExternalToInternalMinRatio) {
                external_to_internal_ids_.AssignMap(internal_count_value);
            } else {
                external_to_internal_ids_.AssignVector(external_count_value, kInvalidId);
            }
            if (internal_count_value == 0) {
                return;
            }
            if (external_ids == nullptr) {
                throw std::runtime_error("invalid external id map data");
            }
            for (size_t internal_id = 0; internal_id < internal_count_value; ++internal_id) {
                const auto external_id = external_ids[internal_id];
                if (external_id < 0 || static_cast<size_t>(external_id) >= external_count_) {
                    throw std::runtime_error("external id map points outside external rows");
                }
                external_to_internal_ids_.Set(external_id, static_cast<int32_t>(internal_id));
            }
        });
    }

    void
    SetValidBitmap(const uint8_t* valid_bitmap, int64_t external_count) {
        Write([&] {
            if (external_count < 0 || external_count > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("invalid external id map size");
            }
            external_count_ = static_cast<size_t>(external_count);
            const auto bitmap_size = (external_count_ + 7) / 8;
            if (bitmap_size == 0) {
                valid_bitmap_.clear();
                return;
            }
            if (valid_bitmap == nullptr) {
                throw std::runtime_error("invalid external id map data");
            }
            valid_bitmap_.assign(valid_bitmap, valid_bitmap + bitmap_size);
            const auto used_bits = external_count_ & 7;
            if (used_bits != 0 && !valid_bitmap_.empty()) {
                const auto mask = static_cast<uint8_t>((1U << used_bits) - 1U);
                valid_bitmap_.back() &= mask;
            }
        });
    }

    void
    BuildIdsFromValidBitmap() {
        Write([&] {
            internal_to_external_ids_.clear();
            external_to_internal_ids_.ids_ = {};
            external_to_internal_ids_.id_map_ = {};
            external_to_internal_ids_.use_map_ = false;
            if (external_count_ == 0 || valid_bitmap_.empty()) {
                return;
            }

            for (size_t external_id = 0; external_id < external_count_; ++external_id) {
                if ((valid_bitmap_[external_id >> 3] & (1U << (external_id & 7))) != 0) {
                    internal_to_external_ids_.push_back(static_cast<int32_t>(external_id));
                }
            }

            const auto internal_count = internal_to_external_ids_.size();
            const auto valid_ratio = static_cast<double>(internal_count) / static_cast<double>(external_count_);
            if (valid_ratio <= kDenseExternalToInternalMinRatio) {
                external_to_internal_ids_.AssignMap(internal_count);
            } else {
                external_to_internal_ids_.AssignVector(external_count_, kInvalidId);
            }
            for (size_t internal_id = 0; internal_id < internal_count; ++internal_id) {
                external_to_internal_ids_.Set(internal_to_external_ids_[internal_id],
                                              static_cast<int32_t>(internal_id));
            }
        });
    }

    void
    BuildEmbListIds(const size_t* emb_list_offsets, int64_t emb_list_count) {
        Write([&] {
            if (emb_list_count < 0) {
                throw std::runtime_error("invalid emb list id map size");
            }
            const auto emb_list_count_value = static_cast<size_t>(emb_list_count);
            if (emb_list_count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error("invalid emb list id map size");
            }
            internal_id_to_emb_list_id_.clear();
            if (emb_list_count_value == 0) {
                return;
            }
            if (emb_list_offsets == nullptr) {
                throw std::runtime_error("invalid emb list id map data");
            }
            if (emb_list_offsets[0] != 0) {
                throw std::runtime_error("invalid emb list id map data");
            }
            if (internal_to_external_ids_.size() < emb_list_count_value) {
                throw std::runtime_error("invalid emb list id map size");
            }
            const auto internal_count = emb_list_offsets[emb_list_count_value];
            if (internal_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error("invalid emb list id map size");
            }
            internal_id_to_emb_list_id_.assign(internal_count, kInvalidId);
            for (size_t emb_list_id = 0; emb_list_id < emb_list_count_value; ++emb_list_id) {
                if (emb_list_offsets[emb_list_id] > emb_list_offsets[emb_list_id + 1]) {
                    throw std::runtime_error("invalid emb list id map data");
                }
                const auto external_emb_list_id = internal_to_external_ids_[emb_list_id];
                for (size_t internal_id = emb_list_offsets[emb_list_id];
                     internal_id < emb_list_offsets[emb_list_id + 1]; ++internal_id) {
                    internal_id_to_emb_list_id_[internal_id] = external_emb_list_id;
                }
            }
        });
    }

    void
    ClearIds() {
        Write([&] {
            internal_to_external_ids_.clear();
            internal_id_to_emb_list_id_.clear();
            external_to_internal_ids_.ids_ = {};
            external_to_internal_ids_.id_map_ = {};
            external_to_internal_ids_.use_map_ = false;
        });
    }

    void
    AddIdsAndBitmap(const int32_t* ids, int64_t count, int64_t external_count) {
        if (count < 0 || external_count < 0 || count > external_count) {
            throw std::runtime_error("invalid external id map size");
        }
        const auto count_value = static_cast<size_t>(count);
        const auto external_count_value = static_cast<size_t>(external_count);
        if (count_value == 0 && external_count_value == 0) {
            return;
        }
        if (count_value != 0 && ids == nullptr) {
            throw std::runtime_error("invalid external id map data");
        }

        size_t internal_id_begin = 0;
        size_t old_external_count = 0;
        Read([&] {
            internal_id_begin = internal_to_external_ids_.size();
            old_external_count = external_count_;
        });
        if (count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - internal_id_begin ||
            external_count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - old_external_count) {
            throw std::runtime_error("invalid external id map size");
        }

        std::vector<int32_t> new_internal_to_external_ids(count_value);
        for (size_t i = 0; i < count_value; ++i) {
            if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= external_count_value) {
                throw std::runtime_error("external id map points outside external rows");
            }
            new_internal_to_external_ids[i] = static_cast<int32_t>(old_external_count + ids[i]);
        }

        Write([&] {
            if (internal_id_begin != internal_to_external_ids_.size() || old_external_count != external_count_) {
                throw std::runtime_error("external id map changed while appending ids");
            }
            external_count_ += external_count_value;
            valid_bitmap_.resize((external_count_ + 7) / 8, 0);

            internal_to_external_ids_.resize(internal_id_begin + count_value);
            for (size_t i = 0; i < count_value; ++i) {
                const auto external_id = new_internal_to_external_ids[i];
                internal_to_external_ids_[internal_id_begin + i] = external_id;
                external_to_internal_ids_.Add(external_id, static_cast<int32_t>(internal_id_begin + i));
                valid_bitmap_[external_id >> 3] |= static_cast<uint8_t>(1U << (external_id & 7));
            }
        });
    }

    void
    AddEmbListIdsAndBitmap(const int32_t* ids, int64_t count, int64_t external_count,
                           const size_t* emb_list_offsets) {
        if (count < 0 || external_count < 0 || count > external_count) {
            throw std::runtime_error("invalid external id map size");
        }
        const auto count_value = static_cast<size_t>(count);
        const auto external_count_value = static_cast<size_t>(external_count);
        if (count_value == 0 && external_count_value == 0) {
            return;
        }
        if (count_value != 0 && (ids == nullptr || emb_list_offsets == nullptr)) {
            throw std::runtime_error("invalid external id map data");
        }

        size_t internal_id_begin = 0;
        size_t vector_id_begin = 0;
        size_t old_external_count = 0;
        Read([&] {
            internal_id_begin = internal_to_external_ids_.size();
            vector_id_begin = internal_id_to_emb_list_id_.size();
            old_external_count = external_count_;
        });
        if (count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - internal_id_begin ||
            external_count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - old_external_count) {
            throw std::runtime_error("invalid external id map size");
        }
        if (count_value == 0) {
            Write([&] {
                if (internal_id_begin != internal_to_external_ids_.size() ||
                    vector_id_begin != internal_id_to_emb_list_id_.size() ||
                    old_external_count != external_count_) {
                    throw std::runtime_error("external id map changed while appending ids");
                }
                external_count_ += external_count_value;
                valid_bitmap_.resize((external_count_ + 7) / 8, 0);
            });
            return;
        }
        if (emb_list_offsets[0] != 0) {
            throw std::runtime_error("invalid emb list id map data");
        }
        for (size_t i = 0; i < count_value; ++i) {
            if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= external_count_value) {
                throw std::runtime_error("external id map points outside external rows");
            }
            if (emb_list_offsets[i] > emb_list_offsets[i + 1]) {
                throw std::runtime_error("invalid emb list id map data");
            }
        }
        const auto vector_count = emb_list_offsets[count_value];
        if (vector_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - vector_id_begin) {
            throw std::runtime_error("invalid emb list id map size");
        }

        std::vector<int32_t> new_internal_to_external_ids(count_value);
        for (size_t i = 0; i < count_value; ++i) {
            new_internal_to_external_ids[i] = static_cast<int32_t>(old_external_count + ids[i]);
        }

        std::vector<int32_t> new_internal_id_to_emb_list_ids(vector_count, kInvalidId);
        for (size_t emb_list_id = 0; emb_list_id < count_value; ++emb_list_id) {
            const auto external_emb_list_id = new_internal_to_external_ids[emb_list_id];
            std::fill(new_internal_id_to_emb_list_ids.begin() + emb_list_offsets[emb_list_id],
                      new_internal_id_to_emb_list_ids.begin() + emb_list_offsets[emb_list_id + 1],
                      external_emb_list_id);
        }

        Write([&] {
            if (internal_id_begin != internal_to_external_ids_.size() ||
                vector_id_begin != internal_id_to_emb_list_id_.size() ||
                old_external_count != external_count_) {
                throw std::runtime_error("external id map changed while appending ids");
            }
            external_count_ += external_count_value;
            valid_bitmap_.resize((external_count_ + 7) / 8, 0);
            internal_to_external_ids_.resize(internal_id_begin + count_value);
            internal_id_to_emb_list_id_.resize(vector_id_begin + vector_count, kInvalidId);

            for (size_t i = 0; i < count_value; ++i) {
                const auto external_id = new_internal_to_external_ids[i];
                internal_to_external_ids_[internal_id_begin + i] = external_id;
                external_to_internal_ids_.Add(external_id, static_cast<int32_t>(internal_id_begin + i));
                valid_bitmap_[external_id >> 3] |= static_cast<uint8_t>(1U << (external_id & 7));
            }
            std::copy(new_internal_id_to_emb_list_ids.begin(), new_internal_id_to_emb_list_ids.end(),
                      internal_id_to_emb_list_id_.begin() + vector_id_begin);
        });
    }

 private:
    mutable std::shared_mutex mutex_;
    bool use_lock_ = false;
    std::vector<int32_t> internal_to_external_ids_;
    std::vector<int32_t> internal_id_to_emb_list_id_;
    ExternalToInternalStorage external_to_internal_ids_;
    std::vector<uint8_t> valid_bitmap_;
    size_t external_count_ = 0;
};

inline const ExternalIdMap&
EmptyExternalIdMap() {
    static const ExternalIdMap empty_map;
    return empty_map;
}

}  // namespace knowhere

#endif /* EXTERNAL_ID_MAP_H */
