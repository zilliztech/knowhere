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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "knowhere/bitsetview.h"

namespace knowhere {

struct IdArrayBuffer {
    explicit IdArrayBuffer(size_t capacity) : capacity(capacity), data(std::make_unique<int32_t[]>(capacity)) {
    }

    size_t capacity;
    std::unique_ptr<int32_t[]> data;
};

class IdArraySnapshot {
 public:
    IdArraySnapshot() = default;

    IdArraySnapshot(std::shared_ptr<const IdArrayBuffer> buffer, size_t size)
        : buffer_(std::move(buffer)), size_(size) {
    }

    IdArraySnapshot(const int32_t* data, size_t size) : data_(data), size_(size) {
    }

    const int32_t*
    data() const {
        if (data_ != nullptr) {
            return data_;
        }
        return buffer_ == nullptr || size_ == 0 ? nullptr : buffer_->data.get();
    }

    size_t
    size() const {
        return size_;
    }

    bool
    empty() const {
        return data() == nullptr || size_ == 0;
    }

    int32_t
    operator[](size_t offset) const {
        return data()[offset];
    }

 private:
    std::shared_ptr<const IdArrayBuffer> buffer_;
    const int32_t* data_ = nullptr;
    size_t size_ = 0;
};

struct BitmapBuffer {
    explicit BitmapBuffer(size_t capacity) : capacity(capacity), data(std::make_unique<uint8_t[]>(capacity)) {
        std::fill(data.get(), data.get() + capacity, 0);
    }

    size_t capacity;
    std::unique_ptr<uint8_t[]> data;
};

class BitmapSnapshot {
 public:
    BitmapSnapshot() = default;

    BitmapSnapshot(std::shared_ptr<const BitmapBuffer> buffer, size_t bit_count)
        : buffer_(std::move(buffer)), bit_count_(bit_count) {
    }

    const uint8_t*
    data() const {
        return buffer_ == nullptr || bit_count_ == 0 ? nullptr : buffer_->data.get();
    }

    size_t
    size() const {
        return bit_count_;
    }

    bool
    empty() const {
        return data() == nullptr || bit_count_ == 0;
    }

 private:
    std::shared_ptr<const BitmapBuffer> buffer_;
    size_t bit_count_ = 0;
};

class IdMapSnapshot {
 public:
    IdMapSnapshot() = default;

    IdMapSnapshot(IdArraySnapshot in_to_out_ids, IdArraySnapshot in_to_out_ebl_ids, BitmapSnapshot valid_bitmap,
                  size_t out_count)
        : in_to_out_ids_(std::move(in_to_out_ids)),
          in_to_out_ebl_ids_(std::move(in_to_out_ebl_ids)),
          valid_bitmap_(std::move(valid_bitmap)),
          out_count_(out_count) {
    }

    int64_t
    GetCount() const {
        return static_cast<int64_t>(out_count_);
    }

    const BitmapSnapshot&
    GetValidBitmap() const {
        return valid_bitmap_;
    }

    const IdArraySnapshot&
    GetInToOutIds() const {
        return in_to_out_ids_;
    }

    const IdArraySnapshot&
    GetInToOutEblIds() const {
        return in_to_out_ebl_ids_;
    }

    int64_t
    MapInToOut(int64_t in_id) const {
        return MapInToOut(in_to_out_ids_, in_id);
    }

    int64_t
    MapInToOut(const IdArraySnapshot& ids, int64_t in_id) const {
        if (in_id < 0 || ids.empty()) {
            return in_id;
        }
        return ids[static_cast<size_t>(in_id)];
    }

    template <typename IdType>
    void
    MapInToOut(IdType* ids, size_t count) const {
        MapInToOut(in_to_out_ids_, ids, count);
    }

    template <typename IdType>
    void
    MapInToOut(const IdArraySnapshot& id_map, IdType* ids, size_t count) const {
        if (ids == nullptr) {
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            ids[i] = static_cast<IdType>(MapInToOut(id_map, ids[i]));
        }
    }

 private:
    IdArraySnapshot in_to_out_ids_;
    IdArraySnapshot in_to_out_ebl_ids_;
    BitmapSnapshot valid_bitmap_;
    size_t out_count_ = 0;
};

// Runtime-only logical-id mapping. Knowhere never serializes this structure into
// index BinarySet payloads; callers persist nullable/valid-row state externally,
// then configure the map exposed by Index/IndexNode before build or load.
struct IdMap {
 private:
    static constexpr int32_t kInvalidId = -1;
    static constexpr double kDenseOutToInMinRatio = 0.10;

    struct OutToInStorage {
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
                for (size_t out_id = 0; out_id < ids_.size(); ++out_id) {
                    if (ids_[out_id] != kInvalidId) {
                        id_map.emplace(static_cast<int32_t>(out_id), ids_[out_id]);
                    }
                }
                ids_ = {};
                id_map_ = std::move(id_map);
                use_map_ = true;
            }
            Set(id, value);
        }

        void
        Clear() {
            ids_ = {};
            id_map_ = {};
            use_map_ = false;
        }

        std::vector<int32_t> ids_;
        std::unordered_map<int32_t, int32_t> id_map_;
        bool use_map_ = false;
    };

    struct IdArrayStorage {
        IdArrayStorage() = default;

        IdArrayStorage(const IdArrayStorage& other) {
            Assign(other.data(), other.size());
        }

        IdArrayStorage(IdArrayStorage&& other) noexcept = default;

        IdArrayStorage&
        operator=(const IdArrayStorage& other) {
            if (this != &other) {
                Assign(other.data(), other.size());
            }
            return *this;
        }

        IdArrayStorage&
        operator=(IdArrayStorage&& other) noexcept = default;

        const int32_t*
        data() const {
            return ids_ == nullptr || size_ == 0 ? nullptr : ids_->data.get();
        }

        size_t
        size() const {
            return size_;
        }

        bool
        empty() const {
            return size_ == 0;
        }

        int32_t
        operator[](size_t offset) const {
            return ids_->data[offset];
        }

        void
        Clear() {
            ids_.reset();
            size_ = 0;
        }

        void
        Assign(const int32_t* data, size_t size) {
            if (size == 0) {
                Clear();
                return;
            }
            if (data == nullptr) {
                throw std::runtime_error("invalid external id map data");
            }
            auto ids = NewBuffer(size);
            std::copy(data, data + size, ids->data.get());
            ids_ = std::move(ids);
            size_ = size;
        }

        void
        Assign(size_t size, int32_t value) {
            if (size == 0) {
                Clear();
                return;
            }
            auto ids = NewBuffer(size);
            std::fill(ids->data.get(), ids->data.get() + size, value);
            ids_ = std::move(ids);
            size_ = size;
        }

        void
        PushBack(int32_t value) {
            EnsureCapacity(size_ + 1);
            ids_->data[size_] = value;
            ++size_;
        }

        void
        Set(size_t offset, int32_t value) {
            if (offset >= size_) {
                throw std::runtime_error("invalid external id map size");
            }
            ids_->data[offset] = value;
        }

        void
        Resize(size_t size, int32_t value) {
            EnsureCapacity(size);
            if (size > size_) {
                std::fill(ids_->data.get() + size_, ids_->data.get() + size, value);
            }
            size_ = size;
        }

        template <typename Iter>
        void
        CopyTo(size_t offset, Iter begin, Iter end) {
            const auto count = static_cast<size_t>(std::distance(begin, end));
            if (offset + count > size_) {
                throw std::runtime_error("invalid external id map size");
            }
            std::copy(begin, end, ids_->data.get() + offset);
        }

        IdArraySnapshot
        Snapshot() const {
            return {ids_, size_};
        }

     private:
        static size_t
        GrowCapacity(size_t size) {
            size_t capacity = 8192;
            while (capacity < size) {
                capacity <<= 1;
            }
            return capacity;
        }

        static std::shared_ptr<IdArrayBuffer>
        NewBuffer(size_t size) {
            return std::make_shared<IdArrayBuffer>(GrowCapacity(size));
        }

        void
        EnsureCapacity(size_t size) {
            if (size == 0) {
                return;
            }
            if (ids_ != nullptr && ids_->capacity >= size) {
                return;
            }
            auto ids = NewBuffer(size);
            if (ids_ != nullptr && size_ != 0) {
                std::copy(ids_->data.get(), ids_->data.get() + size_, ids->data.get());
            }
            ids_ = std::move(ids);
        }

        std::shared_ptr<IdArrayBuffer> ids_;
        size_t size_ = 0;
    };

    struct BitmapStorage {
        BitmapStorage() = default;

        BitmapStorage(const BitmapStorage& other) {
            Assign(other.data(), other.size());
        }

        BitmapStorage(BitmapStorage&& other) noexcept = default;

        BitmapStorage&
        operator=(const BitmapStorage& other) {
            if (this != &other) {
                Assign(other.data(), other.size());
            }
            return *this;
        }

        BitmapStorage&
        operator=(BitmapStorage&& other) noexcept = default;

        const uint8_t*
        data() const {
            return bitmap_ == nullptr || bit_count_ == 0 ? nullptr : bitmap_->data.get();
        }

        size_t
        size() const {
            return bit_count_;
        }

        bool
        empty() const {
            return bitmap_ == nullptr || bit_count_ == 0;
        }

        bool
        Test(size_t bit) const {
            return bit < bit_count_ && (bitmap_->data[bit >> 3] & (1U << (bit & 7))) != 0;
        }

        void
        Clear() {
            bitmap_.reset();
            bit_count_ = 0;
        }

        void
        Assign(const uint8_t* data, size_t bit_count) {
            if (bit_count == 0) {
                Clear();
                return;
            }
            if (data == nullptr) {
                throw std::runtime_error("invalid external id map data");
            }
            auto bitmap = NewBuffer(ByteSize(bit_count));
            std::memcpy(bitmap->data.get(), data, ByteSize(bit_count));
            bitmap_ = std::move(bitmap);
            bit_count_ = bit_count;
            MaskTail();
        }

        void
        Resize(size_t bit_count) {
            if (bit_count == 0) {
                Clear();
                return;
            }
            const auto old_bit_count = bit_count_;
            const auto old_byte_count = ByteSize(old_bit_count);
            const auto new_byte_count = ByteSize(bit_count);
            const auto should_copy_tail =
                bit_count > old_bit_count && (old_bit_count & 7U) != 0 && bitmap_ != nullptr && bitmap_.use_count() > 1;
            EnsureCapacity(bit_count, should_copy_tail);
            if (new_byte_count > old_byte_count) {
                std::fill(bitmap_->data.get() + old_byte_count, bitmap_->data.get() + new_byte_count, 0);
            }
            bit_count_ = bit_count;
            MaskTail();
        }

        void
        Set(size_t bit) {
            if (bit >= bit_count_) {
                throw std::runtime_error("invalid external id map size");
            }
            bitmap_->data[bit >> 3] |= static_cast<uint8_t>(1U << (bit & 7));
        }

        BitmapSnapshot
        Snapshot() const {
            return {bitmap_, bit_count_};
        }

     private:
        static size_t
        ByteSize(size_t bit_count) {
            return (bit_count + 7) / 8;
        }

        static size_t
        GrowCapacity(size_t size) {
            size_t capacity = 1024;
            while (capacity < size) {
                capacity <<= 1;
            }
            return capacity;
        }

        static std::shared_ptr<BitmapBuffer>
        NewBuffer(size_t size) {
            return std::make_shared<BitmapBuffer>(GrowCapacity(size));
        }

        void
        EnsureCapacity(size_t bit_count, bool force_copy) {
            const auto byte_count = ByteSize(bit_count);
            if (byte_count == 0) {
                return;
            }
            if (!force_copy && bitmap_ != nullptr && bitmap_->capacity >= byte_count) {
                return;
            }
            auto bitmap = NewBuffer(byte_count);
            if (bitmap_ != nullptr && bit_count_ != 0) {
                std::memcpy(bitmap->data.get(), bitmap_->data.get(), ByteSize(bit_count_));
            }
            bitmap_ = std::move(bitmap);
        }

        void
        MaskTail() {
            const auto used_bits = bit_count_ & 7U;
            if (used_bits != 0 && bitmap_ != nullptr) {
                const auto mask = static_cast<uint8_t>((1U << used_bits) - 1U);
                bitmap_->data[ByteSize(bit_count_) - 1] &= mask;
            }
        }

        std::shared_ptr<BitmapBuffer> bitmap_;
        size_t bit_count_ = 0;
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
    explicit IdMap(bool use_lock = false) : use_lock_(use_lock) {
    }

    void
    SetUseLock(bool use_lock) {
        if (use_lock_) {
            std::unique_lock lock(mutex_);
            use_lock_ = use_lock;
            return;
        }
        use_lock_ = use_lock;
    }

    bool
    UseLock() const {
        if (!use_lock_) {
            return false;
        }
        std::shared_lock lock(mutex_);
        return use_lock_;
    }

    IdMap(const IdMap& other) {
        other.Read([&] {
            use_lock_ = other.use_lock_;
            in_to_out_ids_ = other.in_to_out_ids_;
            in_to_out_ebl_ids_ = other.in_to_out_ebl_ids_;
            out_to_in_ids_ = other.out_to_in_ids_;
            valid_bitmap_ = other.valid_bitmap_;
            out_count_ = other.out_count_;
        });
    }

    IdMap(IdMap&& other) {
        other.Write([&] {
            use_lock_ = other.use_lock_;
            in_to_out_ids_ = std::move(other.in_to_out_ids_);
            in_to_out_ebl_ids_ = std::move(other.in_to_out_ebl_ids_);
            out_to_in_ids_ = std::move(other.out_to_in_ids_);
            valid_bitmap_ = std::move(other.valid_bitmap_);
            out_count_ = other.out_count_;
        });
    }

    IdMap&
    operator=(const IdMap& other) {
        if (this == &other) {
            return *this;
        }
        IdMap snapshot(other);
        Write([&] {
            use_lock_ = snapshot.use_lock_;
            in_to_out_ids_ = snapshot.in_to_out_ids_;
            in_to_out_ebl_ids_ = snapshot.in_to_out_ebl_ids_;
            out_to_in_ids_ = snapshot.out_to_in_ids_;
            valid_bitmap_ = snapshot.valid_bitmap_;
            out_count_ = snapshot.out_count_;
        });
        return *this;
    }

    IdMap&
    operator=(IdMap&& other) {
        if (this == &other) {
            return *this;
        }
        IdMap snapshot(std::move(other));
        Write([&] {
            use_lock_ = snapshot.use_lock_;
            in_to_out_ids_ = std::move(snapshot.in_to_out_ids_);
            in_to_out_ebl_ids_ = std::move(snapshot.in_to_out_ebl_ids_);
            out_to_in_ids_ = std::move(snapshot.out_to_in_ids_);
            valid_bitmap_ = std::move(snapshot.valid_bitmap_);
            out_count_ = snapshot.out_count_;
        });
        return *this;
    }

    IdMapSnapshot
    GetSnapshot() const {
        return Read([&]() -> IdMapSnapshot {
            return {in_to_out_ids_.Snapshot(), in_to_out_ebl_ids_.Snapshot(), valid_bitmap_.Snapshot(), out_count_};
        });
    }

    int64_t
    MapOutToIn(int64_t out_id) const {
        return Read([&] {
            if (out_count_ == 0) {
                return out_id;
            }
            return static_cast<int64_t>(out_to_in_ids_.Get(out_id));
        });
    }

    const int64_t*
    MapOutToIn(const int64_t* ids, size_t count, std::vector<int64_t>& in_ids) const {
        return Read([&]() -> const int64_t* {
            if (out_count_ == 0) {
                return ids;
            }
            in_ids.resize(count);
            for (size_t i = 0; i < count; ++i) {
                in_ids[i] = out_to_in_ids_.Get(ids[i]);
            }
            return in_ids.data();
        });
    }

    void
    SetValidBitmap(const uint8_t* valid_bitmap, int64_t out_count) {
        Write([&] {
            if (out_count < 0 || out_count > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("invalid external id map size");
            }
            out_count_ = static_cast<size_t>(out_count);
            valid_bitmap_.Assign(valid_bitmap, out_count_);
        });
    }

    void
    SetValidBitmap(const bool* valid_data, int64_t out_count) {
        if (out_count < 0 || out_count > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("invalid external id map size");
        }
        const auto out_count_value = static_cast<size_t>(out_count);
        if (out_count_value != 0 && valid_data == nullptr) {
            throw std::runtime_error("invalid external id map data");
        }

        std::vector<uint8_t> valid_bitmap((out_count_value + 7) / 8, 0);
        for (size_t out_id = 0; out_id < out_count_value; ++out_id) {
            if (valid_data[out_id]) {
                valid_bitmap[out_id >> 3] |= static_cast<uint8_t>(1U << (out_id & 7));
            }
        }
        SetValidBitmap(valid_bitmap.data(), out_count);
    }

    void
    BuildIdsFromValidBitmap() {
        Write([&] {
            in_to_out_ids_.Clear();
            in_to_out_ebl_ids_.Clear();
            out_to_in_ids_.Clear();
            if (out_count_ == 0 || valid_bitmap_.empty()) {
                return;
            }

            for (size_t out_id = 0; out_id < out_count_; ++out_id) {
                if (valid_bitmap_.Test(out_id)) {
                    in_to_out_ids_.PushBack(static_cast<int32_t>(out_id));
                }
            }

            const auto in_count = in_to_out_ids_.size();
            const auto valid_ratio = static_cast<double>(in_count) / static_cast<double>(out_count_);
            if (valid_ratio <= kDenseOutToInMinRatio) {
                out_to_in_ids_.AssignMap(in_count);
            } else {
                out_to_in_ids_.AssignVector(out_count_, kInvalidId);
            }
            for (size_t in_id = 0; in_id < in_count; ++in_id) {
                out_to_in_ids_.Set(in_to_out_ids_[in_id], static_cast<int32_t>(in_id));
            }
        });
    }

    void
    BuildInToOutEblIds(const size_t* ebl_offsets, int64_t ebl_count) {
        Write([&] {
            if (ebl_count < 0) {
                throw std::runtime_error("invalid emb list id map size");
            }
            const auto ebl_count_value = static_cast<size_t>(ebl_count);
            if (ebl_count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error("invalid emb list id map size");
            }
            in_to_out_ebl_ids_.Clear();
            if (ebl_count_value == 0) {
                return;
            }
            if (ebl_offsets == nullptr) {
                throw std::runtime_error("invalid emb list id map data");
            }
            if (ebl_offsets[0] != 0) {
                throw std::runtime_error("invalid emb list id map data");
            }
            if (!in_to_out_ids_.empty() && in_to_out_ids_.size() < ebl_count_value) {
                throw std::runtime_error("invalid emb list id map size");
            }
            const auto in_count = ebl_offsets[ebl_count_value];
            if (in_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error("invalid emb list id map size");
            }
            in_to_out_ebl_ids_.Assign(in_count, kInvalidId);
            for (size_t ebl_id = 0; ebl_id < ebl_count_value; ++ebl_id) {
                if (ebl_offsets[ebl_id] > ebl_offsets[ebl_id + 1]) {
                    throw std::runtime_error("invalid emb list id map data");
                }
                const auto out_ebl_id = in_to_out_ids_.empty() ? static_cast<int32_t>(ebl_id) : in_to_out_ids_[ebl_id];
                for (size_t in_id = ebl_offsets[ebl_id]; in_id < ebl_offsets[ebl_id + 1]; ++in_id) {
                    in_to_out_ebl_ids_.Set(in_id, out_ebl_id);
                }
            }
        });
    }

    void
    ClearIds() {
        Write([&] {
            in_to_out_ids_.Clear();
            in_to_out_ebl_ids_.Clear();
            out_to_in_ids_.Clear();
        });
    }

    void
    ClearEblIds() {
        Write([&] { in_to_out_ebl_ids_.Clear(); });
    }

    void
    AddIdsAndBitmap(const int32_t* out_ids, int64_t in_count, int64_t out_count) {
        if (in_count < 0 || out_count < 0 || in_count > out_count) {
            throw std::runtime_error("invalid external id map size");
        }
        const auto in_count_value = static_cast<size_t>(in_count);
        const auto out_count_value = static_cast<size_t>(out_count);
        if (in_count_value == 0 && out_count_value == 0) {
            return;
        }
        if (in_count_value != 0 && out_ids == nullptr) {
            throw std::runtime_error("invalid external id map data");
        }

        size_t in_id_begin = 0;
        size_t old_out_count = 0;
        Read([&] {
            in_id_begin = in_to_out_ids_.size();
            old_out_count = out_count_;
        });
        if (in_count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - in_id_begin ||
            out_count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - old_out_count) {
            throw std::runtime_error("invalid external id map size");
        }

        std::vector<int32_t> new_in_to_out_ids(in_count_value);
        for (size_t i = 0; i < in_count_value; ++i) {
            if (out_ids[i] < 0 || static_cast<size_t>(out_ids[i]) >= out_count_value) {
                throw std::runtime_error("external id map points outside external rows");
            }
            new_in_to_out_ids[i] = static_cast<int32_t>(old_out_count + out_ids[i]);
        }

        Write([&] {
            if (in_id_begin != in_to_out_ids_.size() || old_out_count != out_count_) {
                throw std::runtime_error("external id map changed while appending ids");
            }
            out_count_ += out_count_value;
            valid_bitmap_.Resize(out_count_);

            in_to_out_ids_.Resize(in_id_begin + in_count_value, kInvalidId);
            for (size_t i = 0; i < in_count_value; ++i) {
                const auto out_id = new_in_to_out_ids[i];
                in_to_out_ids_.Set(in_id_begin + i, out_id);
                out_to_in_ids_.Add(out_id, static_cast<int32_t>(in_id_begin + i));
                valid_bitmap_.Set(out_id);
            }
        });
    }

    void
    AddIdsAndBitmap(const bool* valid_data, int64_t out_count) {
        if (out_count < 0) {
            throw std::runtime_error("invalid external id map size");
        }
        const auto out_count_value = static_cast<size_t>(out_count);
        if (out_count_value == 0) {
            return;
        }
        if (valid_data == nullptr) {
            throw std::runtime_error("invalid external id map data");
        }

        size_t in_id_begin = 0;
        size_t old_out_count = 0;
        Read([&] {
            in_id_begin = in_to_out_ids_.size();
            old_out_count = out_count_;
        });
        if (out_count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - old_out_count) {
            throw std::runtime_error("invalid external id map size");
        }

        size_t in_count_value = 0;
        for (size_t out_id = 0; out_id < out_count_value; ++out_id) {
            if (valid_data[out_id]) {
                ++in_count_value;
            }
        }
        if (in_count_value > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - in_id_begin) {
            throw std::runtime_error("invalid external id map size");
        }

        Write([&] {
            if (in_id_begin != in_to_out_ids_.size() || old_out_count != out_count_) {
                throw std::runtime_error("external id map changed while appending ids");
            }
            out_count_ += out_count_value;
            valid_bitmap_.Resize(out_count_);

            in_to_out_ids_.Resize(in_id_begin + in_count_value, kInvalidId);
            size_t valid_offset = 0;
            for (size_t out_id = 0; out_id < out_count_value; ++out_id) {
                if (!valid_data[out_id]) {
                    continue;
                }
                const auto mapped_out_id = static_cast<int32_t>(old_out_count + out_id);
                const auto mapped_in_id = static_cast<int32_t>(in_id_begin + valid_offset);
                in_to_out_ids_.Set(in_id_begin + valid_offset, mapped_out_id);
                out_to_in_ids_.Add(mapped_out_id, mapped_in_id);
                valid_bitmap_.Set(mapped_out_id);
                ++valid_offset;
            }
        });
    }

    void
    AddInToOutEblIds(int64_t ebl_id_begin, const size_t* ebl_offsets, int64_t ebl_count) {
        if (ebl_id_begin < 0 || ebl_count < 0) {
            throw std::runtime_error("invalid emb list id map size");
        }
        const auto ebl_count_value = static_cast<size_t>(ebl_count);
        if (ebl_count_value == 0) {
            return;
        }
        if (ebl_offsets == nullptr || ebl_offsets[0] != 0) {
            throw std::runtime_error("invalid emb list id map data");
        }
        const auto max_id = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        if (ebl_id_begin > max_id || ebl_count > max_id - ebl_id_begin + 1) {
            throw std::runtime_error("invalid emb list id map size");
        }

        size_t ebl_map_begin = 0;
        Read([&] { ebl_map_begin = in_to_out_ebl_ids_.size(); });

        const auto appended_in_count = ebl_offsets[ebl_count_value];
        if (appended_in_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - ebl_map_begin) {
            throw std::runtime_error("invalid emb list id map size");
        }

        std::vector<int32_t> new_in_to_out_ebl_ids(appended_in_count, kInvalidId);
        for (size_t ebl_id = 0; ebl_id < ebl_count_value; ++ebl_id) {
            if (ebl_offsets[ebl_id] > ebl_offsets[ebl_id + 1]) {
                throw std::runtime_error("invalid emb list id map data");
            }
            const auto out_id = static_cast<int32_t>(ebl_id_begin + static_cast<int64_t>(ebl_id));
            std::fill(new_in_to_out_ebl_ids.begin() + ebl_offsets[ebl_id],
                      new_in_to_out_ebl_ids.begin() + ebl_offsets[ebl_id + 1], out_id);
        }

        Write([&] {
            if (ebl_map_begin != in_to_out_ebl_ids_.size()) {
                throw std::runtime_error("external id map changed while appending ids");
            }
            in_to_out_ebl_ids_.Resize(ebl_map_begin + appended_in_count, kInvalidId);
            in_to_out_ebl_ids_.CopyTo(ebl_map_begin, new_in_to_out_ebl_ids.begin(), new_in_to_out_ebl_ids.end());
        });
    }

 private:
    mutable std::shared_mutex mutex_;
    bool use_lock_ = false;
    IdArrayStorage in_to_out_ids_;
    IdArrayStorage in_to_out_ebl_ids_;
    OutToInStorage out_to_in_ids_;
    BitmapStorage valid_bitmap_;
    size_t out_count_ = 0;
};

}  // namespace knowhere

#endif /* ID_MAP_H */
