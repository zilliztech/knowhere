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

#ifndef ID_MAP_H
#define ID_MAP_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "knowhere/adaptive_store.h"
#include "knowhere/array_store.h"
#include "knowhere/dataset.h"
#include "knowhere/mmap.h"

namespace knowhere {

struct IdMapMmapOptions {
    // Mmap-backed arrays are used by sealed id maps.
    bool enable_in_to_out_ids = false;
    bool enable_out_to_in_ids = false;
    std::string mmap_dir_path;
};

// Owns the nullable ID-domain boundary.
//
// Public ids are row/list ids used by APIs and filter bitsets. Backend ids are
// dense storage ids after null rows or empty lists are compacted away. IdMap
// provides both directions for result mapping, selected-id lookup, and EmbList
// vector-level filtering.
//
// SEALED maps publish immutable arrays, optionally mmap-backed. GROWING maps
// append new public-id ranges while keeping existing backend ids stable.
class IdMap {
 public:
    enum class Type {
        SEALED,
        GROWING,
    };

    // VECTOR maps vector storage ids. EMB_LIST maps base-vector ids to list ids.
    enum class Domain {
        VECTOR,
        EMB_LIST,
    };

    IdMap() {
        SetType(Type::SEALED);
    }

    void
    SetType(Type type) {
        type_ = type;
        in_to_out_ids_.SetType(IdArrayTypeFor(type_));
        in_to_out_ebl_ids_.SetType(IdArrayTypeFor(type_));
        out_to_in_ids_.SetType(IdArrayTypeFor(type_));
        valid_bitmap_.SetType(BitmapTypeFor(type_));
    }

    IdMap(const IdMap&) = delete;
    IdMap(IdMap&& other) noexcept {
        MoveFrom(std::move(other));
    }

    IdMap&
    operator=(const IdMap&) = delete;
    IdMap&
    operator=(IdMap&& other) noexcept {
        if (this != &other) {
            MoveFrom(std::move(other));
        }
        return *this;
    }

    void
    ConfigureMmap(IdMapMmapOptions options) {
        const auto enable_mmap = options.enable_in_to_out_ids || options.enable_out_to_in_ids;
        if (!enable_mmap) {
            return;
        }
        if (type_ != Type::SEALED) {
            throw std::runtime_error("id map mmap is only supported by sealed storage");
        }
        if (mmap_configured_) {
            throw std::runtime_error("id map mmap has already been configured");
        }
        if (HasData()) {
            throw std::runtime_error("id map mmap must be configured before data is written");
        }
        if (enable_mmap && options.mmap_dir_path.empty()) {
            throw std::runtime_error("id map mmap directory is empty");
        }
        std::filesystem::create_directories(options.mmap_dir_path);

        // These files are local backing allocations.
        const auto dir = std::filesystem::path(options.mmap_dir_path);
        in_to_out_ids_mmap_file_paths_ =
            MmapFilePathGenerator(options.enable_in_to_out_ids ? (dir / "in_to_out_ids").string() : std::string{});
        in_to_out_ebl_ids_mmap_file_paths_ =
            MmapFilePathGenerator(options.enable_in_to_out_ids ? (dir / "in_to_out_ebl_ids").string() : std::string{});
        out_to_in_ids_.SetMmapFilePathGenerator(
            MmapFilePathGenerator(options.enable_out_to_in_ids ? (dir / "out_to_in_ids").string() : std::string{}));
        mmap_configured_ = true;
    }

    void
    AddFromData(const IdMapData& data) {
        switch (data.format) {
            case IdMapData::Format::IDS: {
                if (data.out_count == 0) {
                    return;
                }

                const auto old_out_count = type_ == Type::GROWING ? OutCount() : 0;
                const auto old_valid_count = type_ == Type::GROWING ? InCount() : 0;
                const auto in_id_begin = type_ == Type::GROWING ? old_valid_count : 0;

                std::vector<int32_t> ids(data.in_count);
                std::vector<uint8_t> bitmap(BitmapByteSize(data.out_count), 0);
                for (size_t i = 0; i < data.in_count; ++i) {
                    const auto local_out_id = static_cast<size_t>(data.out_ids[i]);
                    ids[i] = static_cast<int32_t>(old_out_count + local_out_id);
                    SetPackedBit(bitmap, local_out_id);
                }

                if (type_ == Type::GROWING) {
                    valid_bitmap_.Append(bitmap.data(), data.out_count);
                    in_to_out_ids_.Append(ids.data(), ids.size());
                    out_to_in_ids_.Append(ids.data(), ids.size(), old_out_count, data.out_count, in_id_begin);
                    PublishCounts(old_out_count + data.out_count, old_valid_count + ids.size());
                } else {
                    valid_bitmap_.Set(bitmap.data(), data.out_count);
                    in_to_out_ids_.Set(ids.data(), ids.size(), in_to_out_ids_mmap_file_paths_.Next(this));
                    PublishCounts(data.out_count, data.in_count);
                    if (ids.empty()) {
                        return;
                    }
                    out_to_in_ids_.Set(ids.data(), ids.size(), data.out_count);
                }
                return;
            }
            case IdMapData::Format::PACKED_BITMAP: {
                if (data.out_count == 0) {
                    return;
                }

                if (type_ == Type::SEALED) {
                    valid_bitmap_.Set(data.valid_bitmap, data.out_count);
                    PublishCounts(data.out_count, CountPackedBitmap(data.valid_bitmap, data.out_count));
                    return;
                }

                const auto old_out_count = OutCount();
                const auto old_valid_count = InCount();
                const auto in_id_begin = old_valid_count;

                std::vector<int32_t> ids;
                ids.reserve(data.out_count);
                for (size_t out_id = 0; out_id < data.out_count; ++out_id) {
                    if (PackedBit(data.valid_bitmap, out_id)) {
                        ids.push_back(static_cast<int32_t>(old_out_count + out_id));
                    }
                }

                valid_bitmap_.Append(data.valid_bitmap, data.out_count);
                in_to_out_ids_.Append(ids.data(), ids.size());
                out_to_in_ids_.Append(ids.data(), ids.size(), old_out_count, data.out_count, in_id_begin);
                PublishCounts(old_out_count + data.out_count, old_valid_count + ids.size());
                return;
            }
            case IdMapData::Format::BOOL_ARRAY: {
                if (data.out_count == 0) {
                    return;
                }

                auto bitmap = PackBoolArray(data.valid_data, data.out_count);
                if (type_ == Type::SEALED) {
                    valid_bitmap_.Set(bitmap.data(), data.out_count);
                    PublishCounts(data.out_count, CountPackedBitmap(bitmap.data(), data.out_count));
                    return;
                }

                const auto old_out_count = OutCount();
                const auto old_valid_count = InCount();
                const auto in_id_begin = old_valid_count;

                std::vector<int32_t> ids;
                ids.reserve(data.out_count);
                for (size_t out_id = 0; out_id < data.out_count; ++out_id) {
                    if (data.valid_data[out_id]) {
                        ids.push_back(static_cast<int32_t>(old_out_count + out_id));
                    }
                }

                valid_bitmap_.Append(bitmap.data(), data.out_count);
                in_to_out_ids_.Append(ids.data(), ids.size());
                out_to_in_ids_.Append(ids.data(), ids.size(), old_out_count, data.out_count, in_id_begin);
                PublishCounts(old_out_count + data.out_count, old_valid_count + ids.size());
                return;
            }
            default:
                throw std::runtime_error("unsupported id map data format");
        }
    }

    void
    FinalizeVectorIds() {
        // Derives vector maps from public-id validity input.
        const auto out_count = OutCount();
        if (out_count == 0 || valid_bitmap_.empty()) {
            valid_count_.store(0, std::memory_order_release);
            return;
        }
        if (!in_to_out_ids_.empty()) {
            return;
        }

        std::vector<int32_t> ids;
        ids.reserve(InCount());
        for (size_t out_id = 0; out_id < out_count; ++out_id) {
            if (PackedBit(valid_bitmap_, out_id)) {
                ids.push_back(static_cast<int32_t>(out_id));
            }
        }
        valid_count_.store(ids.size(), std::memory_order_release);
        if (ids.empty()) {
            return;
        }
        if (type_ == Type::GROWING) {
            in_to_out_ids_.Append(ids.data(), ids.size());
        } else {
            in_to_out_ids_.Set(ids.data(), ids.size(), in_to_out_ids_mmap_file_paths_.Next(this));
        }
        out_to_in_ids_.Set(ids.data(), ids.size(), out_count);
    }

    void
    FinalizeEmbListIds(const size_t* ebl_offsets, size_t ebl_count) {
        // Base-vector id -> public-list id map for list-domain filters.
        if (ebl_count == 0) {
            return;
        }

        const auto in_count = ebl_offsets[ebl_count];

        std::vector<int32_t> ids(in_count, kInvalidId);
        for (size_t ebl_id = 0; ebl_id < ebl_count; ++ebl_id) {
            const auto out_id = in_to_out_ids_.empty() ? static_cast<int32_t>(ebl_id) : in_to_out_ids_[ebl_id];
            std::fill(ids.begin() + ebl_offsets[ebl_id], ids.begin() + ebl_offsets[ebl_id + 1], out_id);
        }
        if (type_ == Type::GROWING) {
            in_to_out_ebl_ids_.Append(ids.data(), ids.size());
        } else {
            in_to_out_ebl_ids_.Set(ids.data(), ids.size(), in_to_out_ebl_ids_mmap_file_paths_.Next(this));
        }
    }

    void
    AppendEmbListIds(int64_t ebl_id_begin, const size_t* ebl_offsets, int64_t ebl_count) {
        // Appends base-vector ids for a growing EmbList batch.
        const auto list_count = static_cast<size_t>(ebl_count);
        if (list_count == 0) {
            return;
        }

        const auto append_count = ebl_offsets[list_count];

        std::vector<int32_t> ids(append_count, kInvalidId);
        for (size_t ebl_id = 0; ebl_id < list_count; ++ebl_id) {
            const auto out_id = static_cast<int32_t>(ebl_id_begin + static_cast<int64_t>(ebl_id));
            std::fill(ids.begin() + ebl_offsets[ebl_id], ids.begin() + ebl_offsets[ebl_id + 1], out_id);
        }
        in_to_out_ebl_ids_.Append(ids.data(), ids.size());
    }

    size_t
    OutCount() const {
        return out_count_.load(std::memory_order_acquire);
    }

    size_t
    InCount() const {
        return valid_count_.load(std::memory_order_acquire);
    }

    bool
    IsValidOutId(int64_t out_id) const {
        if (out_id < 0) {
            return false;
        }
        const auto offset = static_cast<size_t>(out_id);
        const auto out_count = OutCount();
        if (out_count != 0 && offset >= out_count) {
            return false;
        }
        return valid_bitmap_.empty() || PackedBit(valid_bitmap_, offset);
    }

    BitmapArray
    ValidBitmap() const {
        return valid_bitmap_.Prefix(OutCount());
    }

    IdArray
    InToOutIds(Domain domain = Domain::VECTOR) const {
        if (domain == Domain::EMB_LIST) {
            return in_to_out_ebl_ids_;
        }
        return in_to_out_ids_.Prefix(InCount());
    }

    int64_t
    MapInToOut(int64_t in_id, Domain domain = Domain::VECTOR) const {
        const auto& ids = domain == Domain::EMB_LIST ? in_to_out_ebl_ids_ : in_to_out_ids_;
        if (in_id < 0 || ids.empty()) {
            return in_id;
        }
        const auto offset = static_cast<size_t>(in_id);
        const auto count = InToOutCount(domain);
        return offset < count ? ids[offset] : kInvalidId;
    }

    template <typename IdType>
    void
    MapInToOut(IdType* ids, size_t count, Domain domain = Domain::VECTOR) const {
        const auto& id_map = domain == Domain::EMB_LIST ? in_to_out_ebl_ids_ : in_to_out_ids_;
        if (ids == nullptr || id_map.empty()) {
            return;
        }
        const auto id_map_count = InToOutCount(domain);
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] < 0) {
                continue;
            }
            const auto offset = static_cast<size_t>(ids[i]);
            ids[i] = static_cast<IdType>(offset < id_map_count ? id_map[offset] : kInvalidId);
        }
    }

    int64_t
    MapOutToIn(int64_t out_id) const {
        return OutCount() == 0 ? out_id : static_cast<int64_t>(GetOutToInId(out_id));
    }

    const int64_t*
    MapOutToIn(const int64_t* ids, size_t count, std::vector<int64_t>& in_ids) const {
        if (OutCount() == 0) {
            return ids;
        }
        in_ids.resize(count);
        for (size_t i = 0; i < count; ++i) {
            in_ids[i] = GetOutToInId(ids[i]);
        }
        return in_ids.data();
    }

 private:
    static constexpr int32_t kInvalidId = -1;

    static size_t
    BitmapByteSize(size_t bit_count) {
        return (bit_count + 7) / 8;
    }

    static bool
    PackedBit(const uint8_t* bitmap, size_t bit) {
        return (bitmap[bit >> 3] & (1U << (bit & 7))) != 0;
    }

    static bool
    PackedBit(const BitmapArray& bitmap, size_t bit) {
        return bit < bitmap.size() && (bitmap[bit >> 3] & (1U << (bit & 7))) != 0;
    }

    static void
    SetPackedBit(std::vector<uint8_t>& bitmap, size_t bit) {
        bitmap[bit >> 3] |= static_cast<uint8_t>(1U << (bit & 7));
    }

    static size_t
    CountPackedBitmap(const uint8_t* valid_bitmap, size_t out_count) {
        // Count only the logical public-id range. Padding bits in the final
        // byte/word are ignored even if the caller's buffer contains garbage.
        if (out_count == 0) {
            return 0;
        }
        size_t count = 0;
        const auto full_words = out_count / 64;
        for (size_t i = 0; i < full_words; ++i) {
            uint64_t word = 0;
            std::memcpy(&word, valid_bitmap + i * sizeof(uint64_t), sizeof(uint64_t));
            count += static_cast<size_t>(__builtin_popcountll(static_cast<unsigned long long>(word)));
        }

        const auto tail_bits = out_count % 64;
        if (tail_bits != 0) {
            uint64_t word = 0;
            std::memcpy(&word, valid_bitmap + full_words * sizeof(uint64_t), BitmapByteSize(tail_bits));
            word &= (uint64_t{1} << tail_bits) - 1;
            count += static_cast<size_t>(__builtin_popcountll(static_cast<unsigned long long>(word)));
        }
        return count;
    }

    static std::vector<uint8_t>
    PackBoolArray(const bool* valid_data, size_t out_count) {
        std::vector<uint8_t> bitmap(BitmapByteSize(out_count), 0);
        for (size_t out_id = 0; out_id < out_count; ++out_id) {
            if (valid_data[out_id]) {
                SetPackedBit(bitmap, out_id);
            }
        }
        return bitmap;
    }

    int32_t
    GetOutToInId(int64_t out_id) const {
        if (out_id < 0 || out_id > std::numeric_limits<int32_t>::max()) {
            return kInvalidId;
        }
        const auto out_count = OutCount();
        if (out_count != 0 && static_cast<size_t>(out_id) >= out_count) {
            return kInvalidId;
        }
        return out_to_in_ids_.Get(static_cast<int32_t>(out_id));
    }

    static ArrayStore<int32_t>::Type
    IdArrayTypeFor(Type type) {
        return type == Type::GROWING ? ArrayStore<int32_t>::Type::APPEND_ARRAY : ArrayStore<int32_t>::Type::ARRAY;
    }

    static BitmapArray::Type
    BitmapTypeFor(Type type) {
        return type == Type::GROWING ? BitmapArray::Type::APPEND_ARRAY : BitmapArray::Type::ARRAY;
    }

    bool
    HasData() const {
        return OutCount() != 0 || InCount() != 0 || !in_to_out_ids_.empty() || !in_to_out_ebl_ids_.empty() ||
               !valid_bitmap_.empty();
    }

    void
    PublishCounts(size_t out_count, size_t valid_count) {
        valid_count_.store(valid_count, std::memory_order_release);
        out_count_.store(out_count, std::memory_order_release);
    }

    size_t
    InToOutCount(Domain domain) const {
        return domain == Domain::EMB_LIST ? in_to_out_ebl_ids_.size() : InCount();
    }

    void
    MoveFrom(IdMap&& other) noexcept {
        type_ = other.type_;
        in_to_out_ids_ = std::move(other.in_to_out_ids_);
        in_to_out_ebl_ids_ = std::move(other.in_to_out_ebl_ids_);
        out_to_in_ids_ = std::move(other.out_to_in_ids_);
        valid_bitmap_ = std::move(other.valid_bitmap_);
        out_count_.store(other.OutCount(), std::memory_order_release);
        valid_count_.store(other.InCount(), std::memory_order_release);
        in_to_out_ids_mmap_file_paths_ = std::move(other.in_to_out_ids_mmap_file_paths_);
        in_to_out_ebl_ids_mmap_file_paths_ = std::move(other.in_to_out_ebl_ids_mmap_file_paths_);
        mmap_configured_ = other.mmap_configured_;
    }

    Type type_ = Type::SEALED;
    // Compact vector internal id -> public row id.
    IdArray in_to_out_ids_;
    // Compact base-vector internal id -> public embedding-list id. Only
    // populated for EmbList strategies whose base index filters at vector level.
    IdArray in_to_out_ebl_ids_;
    // Public row/list id -> compact vector/list id for selected-id APIs.
    AdaptiveStore<int32_t> out_to_in_ids_;
    // Public-id validity over [0, out_count_). Empty means identity/non-nullable.
    BitmapArray valid_bitmap_;
    // Size of the public id domain.
    std::atomic<size_t> out_count_{0};
    // Number of compact vector/list ids represented by this map.
    std::atomic<size_t> valid_count_{0};
    MmapFilePathGenerator in_to_out_ids_mmap_file_paths_;
    MmapFilePathGenerator in_to_out_ebl_ids_mmap_file_paths_;
    bool mmap_configured_ = false;
};

}  // namespace knowhere

#endif /* ID_MAP_H */
