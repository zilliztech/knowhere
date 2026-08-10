// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef ARRAY_STORE_H
#define ARRAY_STORE_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "knowhere/mmap.h"

namespace knowhere {

// Contiguous sealed array allocation, heap-backed or mmap-backed.
template <typename T>
struct ArrayData {
    explicit ArrayData(size_t size) : size(size), data(std::make_unique<T[]>(size)) {
    }

    ArrayData(const T* view_data, size_t size) : size(size), view_data(view_data) {
    }

    ArrayData(size_t size, const std::string& filepath)
        : size(size), mmap_region(MmapRegion::Create(filepath, size * sizeof(T))) {
    }

    T*
    mutable_data() {
        if (view_data != nullptr) {
            throw std::runtime_error("array data view is read only");
        }
        return mmap_region == nullptr ? data.get() : static_cast<T*>(mmap_region->data());
    }

    const T*
    data_ptr() const {
        if (view_data != nullptr) {
            return view_data;
        }
        return mmap_region == nullptr ? data.get() : static_cast<const T*>(mmap_region->data());
    }

    size_t size;
    const T* view_data = nullptr;
    std::unique_ptr<T[]> data;
    std::shared_ptr<MmapRegion> mmap_region;
};

template <typename T>
class AppendArrayData {
 public:
    AppendArrayData() {
        for (auto& chunk : chunk_ptrs_) {
            chunk.store(nullptr, std::memory_order_relaxed);
        }
    }

    AppendArrayData(const AppendArrayData&) = delete;
    AppendArrayData&
    operator=(const AppendArrayData&) = delete;

    void
    Append(const T* data, size_t count) {
        const auto begin = committed_count_.load(std::memory_order_acquire);
        if (count > std::numeric_limits<size_t>::max() - begin) {
            throw std::runtime_error("append array size overflows");
        }
        const auto end = begin + count;
        EnsureCapacity(end);

        auto remaining = count;
        auto offset = begin;
        auto* source = data;
        while (remaining != 0) {
            const auto chunk_id = ChunkIndex(offset);
            const auto chunk_offset = offset - ChunkBegin(chunk_id);
            const auto write_count = std::min(remaining, ChunkSize(chunk_id) - chunk_offset);
            auto* chunk = chunk_ptrs_[chunk_id].load(std::memory_order_acquire);
            std::copy(source, source + write_count, chunk + chunk_offset);
            source += write_count;
            offset += write_count;
            remaining -= write_count;
        }

        committed_count_.store(end, std::memory_order_release);
    }

    size_t
    size() const {
        return committed_count_.load(std::memory_order_acquire);
    }

    T
    Get(size_t offset) const {
        const auto chunk_id = ChunkIndex(offset);
        const auto* chunk = chunk_ptrs_[chunk_id].load(std::memory_order_acquire);
        return chunk[offset - ChunkBegin(chunk_id)];
    }

 private:
    static constexpr size_t kFirstChunkBits = 10;
    static constexpr size_t kFirstChunkSize = size_t{1} << kFirstChunkBits;
    static constexpr size_t kMaxChunks = std::numeric_limits<size_t>::digits - kFirstChunkBits;

    static size_t
    ChunkIndex(size_t offset) {
        const auto block = (offset >> kFirstChunkBits) + 1;
        const auto chunk_id = static_cast<size_t>(std::numeric_limits<unsigned long long>::digits - 1 -
                                                  __builtin_clzll(static_cast<unsigned long long>(block)));
        if (chunk_id >= kMaxChunks) {
            throw std::runtime_error("append array capacity is exhausted");
        }
        return chunk_id;
    }

    static size_t
    ChunkBegin(size_t chunk_id) {
        return (kFirstChunkSize << chunk_id) - kFirstChunkSize;
    }

    static size_t
    ChunkSize(size_t chunk_id) {
        return kFirstChunkSize << chunk_id;
    }

    void
    EnsureCapacity(size_t count) {
        if (count == 0) {
            return;
        }
        const auto last_chunk_id = ChunkIndex(count - 1);
        for (size_t chunk_id = 0; chunk_id <= last_chunk_id; ++chunk_id) {
            if (chunks_[chunk_id] != nullptr) {
                continue;
            }
            chunks_[chunk_id] = std::make_unique<T[]>(ChunkSize(chunk_id));
            chunk_ptrs_[chunk_id].store(chunks_[chunk_id].get(), std::memory_order_release);
        }
    }

    std::array<std::unique_ptr<T[]>, kMaxChunks> chunks_;
    std::array<std::atomic<T*>, kMaxChunks> chunk_ptrs_;
    std::atomic<size_t> committed_count_{0};
};

template <typename T>
class ArrayStore {
 public:
    enum class Type {
        ARRAY,
        APPEND_ARRAY,
    };

    // ARRAY stores a complete sealed array through Set(). APPEND_ARRAY stores
    // growing data through append-only chunks.
    ArrayStore() = default;
    ArrayStore(const T* data, size_t count) {
        SetView(data, count);
    }
    ArrayStore(const ArrayStore& other) {
        CopyFrom(other);
    }
    ArrayStore(ArrayStore&&) noexcept = default;
    ArrayStore&
    operator=(const ArrayStore& other) {
        if (this != &other) {
            CopyFrom(other);
        }
        return *this;
    }
    ArrayStore&
    operator=(ArrayStore&&) noexcept = default;

    void
    SetType(Type type) {
        type_ = type;
        if (type_ == Type::APPEND_ARRAY && append_array_ == nullptr) {
            append_array_ = std::make_shared<AppendArrayData<T>>();
        }
        visible_count_ = kDynamicCount;
    }

    void
    SetView(const T* data, size_t count) {
        if (count != 0 && data == nullptr) {
            throw std::runtime_error("array store view data is null");
        }
        Clear();
        type_ = Type::ARRAY;
        if (count == 0) {
            return;
        }
        array_ = std::make_shared<ArrayData<T>>(data, count);
    }

    void
    Set(const T* data, size_t count, const std::string& filepath = std::string{}) {
        if (count != 0 && data == nullptr) {
            throw std::runtime_error("array store data is null");
        }
        Clear();
        type_ = Type::ARRAY;
        if (count == 0) {
            return;
        }
        array_ = NewArray(count, filepath);
        for (size_t i = 0; i < count; ++i) {
            array_->mutable_data()[i] = data[i];
        }
    }

    void
    Append(const T* data, size_t count) {
        if (count != 0 && data == nullptr) {
            throw std::runtime_error("array store data is null");
        }
        if (count == 0) {
            return;
        }
        if (type_ != Type::APPEND_ARRAY || append_array_ == nullptr || visible_count_ != kDynamicCount) {
            throw std::runtime_error("array store append requires append storage");
        }
        append_array_->Append(data, count);
    }

    bool
    empty() const {
        return size() == 0;
    }

    size_t
    size() const {
        const auto storage_count = StorageSize();
        return visible_count_ == kDynamicCount ? storage_count : std::min(visible_count_, storage_count);
    }

    bool
    is_array() const {
        return type_ == Type::ARRAY;
    }

    bool
    is_append_array() const {
        return type_ == Type::APPEND_ARRAY;
    }

    const T*
    data() const {
        return type_ == Type::ARRAY && array_ != nullptr ? array_->data_ptr() : nullptr;
    }

    T
    operator[](size_t offset) const {
        if (type_ == Type::ARRAY) {
            return array_->data_ptr()[offset];
        }
        return append_array_->Get(offset);
    }

    T&
    operator[](size_t offset) {
        if (type_ == Type::ARRAY) {
            return array_->mutable_data()[offset];
        }
        throw std::runtime_error("append array storage is read only");
    }

    void
    Clear() {
        array_.reset();
        append_array_.reset();
        visible_count_ = kDynamicCount;
    }

    ArrayStore
    Prefix(size_t count) const {
        ArrayStore copy(*this);
        copy.visible_count_ = std::min(count, copy.size());
        return copy;
    }

 private:
    static constexpr size_t kDynamicCount = std::numeric_limits<size_t>::max();

    void
    CopyFrom(const ArrayStore& other) {
        type_ = other.type_;
        array_ = other.array_;
        append_array_ = other.append_array_;
        visible_count_ = other.size();
    }

    static std::shared_ptr<ArrayData<T>>
    NewArray(size_t capacity, const std::string& filepath) {
        if (filepath.empty()) {
            return std::make_shared<ArrayData<T>>(capacity);
        }
        return std::make_shared<ArrayData<T>>(capacity, filepath);
    }

    size_t
    StorageSize() const {
        if (type_ == Type::ARRAY) {
            return array_ == nullptr ? 0 : array_->size;
        }
        return append_array_ == nullptr ? 0 : append_array_->size();
    }

    Type type_ = Type::ARRAY;
    std::shared_ptr<ArrayData<T>> array_;
    std::shared_ptr<AppendArrayData<T>> append_array_;
    size_t visible_count_ = kDynamicCount;
};

using IdArray = ArrayStore<int32_t>;

struct BitmapRecord {
    size_t bit_begin = 0;
    size_t bit_count = 0;
    std::shared_ptr<const std::vector<uint8_t>> bytes;

    bool
    Contains(size_t bit) const {
        return bit >= bit_begin && bit < bit_begin + bit_count;
    }

    bool
    Test(size_t bit) const {
        const auto local_bit = bit - bit_begin;
        return ((*bytes)[local_bit >> 3] & (1U << (local_bit & 7))) != 0;
    }
};

class AppendBitmapData {
 public:
    AppendBitmapData() {
        records_.SetType(ArrayStore<BitmapRecord>::Type::APPEND_ARRAY);
    }

    void
    Append(const uint8_t* data, size_t bit_count) {
        const auto bit_begin = bit_count_.load(std::memory_order_acquire);
        const auto bytes = std::make_shared<const std::vector<uint8_t>>(data, data + ByteSize(bit_count));
        const BitmapRecord record{bit_begin, bit_count, bytes};
        records_.Append(&record, 1);
        bit_count_.store(bit_begin + bit_count, std::memory_order_release);
    }

    size_t
    size() const {
        return bit_count_.load(std::memory_order_acquire);
    }

    uint8_t
    GetByte(size_t byte_offset) const {
        const auto bit_begin = byte_offset << 3;
        const auto bit_end = size();
        uint8_t value = 0;
        for (size_t bit = 0; bit < 8; ++bit) {
            const auto absolute_bit = bit_begin + bit;
            if (absolute_bit >= bit_end) {
                break;
            }
            if (Test(absolute_bit)) {
                value |= static_cast<uint8_t>(1U << bit);
            }
        }
        return value;
    }

 private:
    static size_t
    ByteSize(size_t bit_count) {
        return (bit_count + 7) / 8;
    }

    bool
    Test(size_t bit) const {
        auto left = static_cast<size_t>(0);
        auto right = records_.size();
        while (left < right) {
            const auto middle = left + (right - left) / 2;
            if (records_[middle].bit_begin <= bit) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        if (left == 0) {
            return false;
        }

        const auto record = records_[left - 1];
        return record.Contains(bit) && record.Test(bit);
    }

    ArrayStore<BitmapRecord> records_;
    std::atomic<size_t> bit_count_{0};
};

// Packed public-id validity bitmap. size() returns the logical bit count.
class BitmapArray {
 public:
    using Type = ArrayStore<uint8_t>::Type;

    BitmapArray() {
        bytes_.SetType(Type::ARRAY);
    }

    BitmapArray(const BitmapArray& other) {
        CopyFrom(other);
    }

    BitmapArray(BitmapArray&&) noexcept = default;

    BitmapArray&
    operator=(const BitmapArray& other) {
        if (this != &other) {
            CopyFrom(other);
        }
        return *this;
    }

    BitmapArray&
    operator=(BitmapArray&&) noexcept = default;

    void
    SetType(Type type) {
        type_ = type;
        if (type_ == Type::ARRAY) {
            bytes_.SetType(Type::ARRAY);
            bit_count_ = 0;
            return;
        }
        if (append_data_ == nullptr) {
            append_data_ = std::make_shared<AppendBitmapData>();
        }
        bit_count_ = kDynamicBitCount;
    }

    const uint8_t*
    data() const {
        return type_ == Type::ARRAY && bit_count_ != 0 ? bytes_.data() : nullptr;
    }

    size_t
    size() const {
        if (type_ == Type::ARRAY) {
            return bit_count_;
        }
        if (append_data_ == nullptr) {
            return 0;
        }
        return bit_count_ == kDynamicBitCount ? append_data_->size() : bit_count_;
    }

    bool
    empty() const {
        return size() == 0;
    }

    uint8_t
    operator[](size_t offset) const {
        return type_ == Type::ARRAY ? bytes_[offset] : append_data_->GetByte(offset);
    }

    void
    Set(const uint8_t* data, size_t bit_count) {
        Clear();
        if (bit_count == 0) {
            return;
        }
        if (data == nullptr) {
            throw std::runtime_error("bitmap array data is null");
        }
        type_ = Type::ARRAY;
        bytes_.Set(data, ByteSize(bit_count));
        bit_count_ = bit_count;
        MaskTail();
    }

    void
    Append(const uint8_t* data, size_t bit_count) {
        // Append is bit-oriented; batches may end in the middle of a byte.
        if (bit_count == 0) {
            return;
        }
        if (data == nullptr) {
            throw std::runtime_error("bitmap array data is null");
        }
        if (type_ != Type::APPEND_ARRAY || append_data_ == nullptr || bit_count_ != kDynamicBitCount) {
            throw std::runtime_error("bitmap array append requires append storage");
        }
        append_data_->Append(data, bit_count);
    }

    void
    Clear() {
        bytes_.Clear();
        bit_count_ = 0;
        append_data_.reset();
    }

    BitmapArray
    Prefix(size_t bit_count) const {
        BitmapArray copy(*this);
        copy.bit_count_ = std::min(bit_count, copy.size());
        return copy;
    }

 private:
    static constexpr size_t kDynamicBitCount = std::numeric_limits<size_t>::max();

    static size_t
    ByteSize(size_t bit_count) {
        return (bit_count + 7) / 8;
    }

    void
    CopyFrom(const BitmapArray& other) {
        bytes_ = other.bytes_;
        type_ = other.type_;
        bit_count_ = type_ == Type::APPEND_ARRAY ? other.size() : other.bit_count_;
        append_data_ = other.append_data_;
    }

    void
    MaskTail() {
        const auto used_bits = bit_count_ & 7U;
        if (used_bits == 0 || bit_count_ == 0) {
            return;
        }
        const auto byte = ByteSize(bit_count_) - 1;
        bytes_[byte] = static_cast<uint8_t>(bytes_[byte] & static_cast<uint8_t>((1U << used_bits) - 1U));
    }

    ArrayStore<uint8_t> bytes_;
    Type type_ = Type::ARRAY;
    size_t bit_count_ = 0;
    std::shared_ptr<AppendBitmapData> append_data_;
};

}  // namespace knowhere

#endif /* ARRAY_STORE_H */
