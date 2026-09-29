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
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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

namespace detail {

inline bool
PackedBit(const uint8_t* bitmap, size_t bit) {
    return (bitmap[bit >> 3] & (1U << (bit & 7))) != 0;
}

inline void
ClearPackedBits(uint8_t* bitmap, size_t bit_begin, size_t bit_count) {
    while (bit_count != 0 && (bit_begin & 7U) != 0) {
        bitmap[bit_begin >> 3] &= static_cast<uint8_t>(~(1U << (bit_begin & 7U)));
        ++bit_begin;
        --bit_count;
    }

    const auto byte_count = bit_count >> 3;
    if (byte_count != 0) {
        std::memset(bitmap + (bit_begin >> 3), 0, byte_count);
        bit_begin += byte_count << 3;
        bit_count -= byte_count << 3;
    }

    while (bit_count != 0) {
        bitmap[bit_begin >> 3] &= static_cast<uint8_t>(~(1U << (bit_begin & 7U)));
        ++bit_begin;
        --bit_count;
    }
}

inline uint8_t
ReadShiftedByte(const uint8_t* source, unsigned shift) {
    if (shift == 0) {
        return source[0];
    }
    return static_cast<uint8_t>((source[0] >> shift) | (source[1] << (8U - shift)));
}

// ANDs whole target bytes with a possibly bit-shifted source stream.  The
// caller guarantees that source contains the extra carry byte when shift is
// non-zero.  x86 and AArch64 use their baseline vector ISA, with AVX2 selected
// when the including translation unit is compiled for it.
inline void
AndPackedBytes(uint8_t* target, const uint8_t* source, size_t byte_count, unsigned shift) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && defined(__AVX2__)
    {
        const auto shift_right = _mm_cvtsi64_si128(static_cast<long long>(shift));
        const auto shift_left = _mm_cvtsi64_si128(static_cast<long long>(64U - shift));
        while (byte_count >= 32) {
            const auto input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
            __m256i validity = input;
            if (shift != 0) {
                auto next = _mm256_permute4x64_epi64(input, _MM_SHUFFLE(3, 3, 2, 1));
                const auto carry = _mm256_set_epi64x(static_cast<long long>(source[32]), 0, 0, 0);
                next = _mm256_blend_epi32(next, carry, 0xC0);
                validity = _mm256_or_si256(_mm256_srl_epi64(input, shift_right), _mm256_sll_epi64(next, shift_left));
            }
            const auto current = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(target));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(target), _mm256_and_si256(current, validity));
            target += 32;
            source += 32;
            byte_count -= 32;
        }
    }
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && defined(__SSE2__)
    {
        const auto shift_right = _mm_cvtsi64_si128(static_cast<long long>(shift));
        const auto shift_left = _mm_cvtsi64_si128(static_cast<long long>(64U - shift));
        while (byte_count >= 16) {
            const auto input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
            __m128i validity = input;
            if (shift != 0) {
                const auto carry = _mm_slli_si128(_mm_cvtsi64_si128(source[16]), 8);
                const auto next = _mm_unpackhi_epi64(input, carry);
                validity = _mm_or_si128(_mm_srl_epi64(input, shift_right), _mm_sll_epi64(next, shift_left));
            }
            const auto current = _mm_loadu_si128(reinterpret_cast<const __m128i*>(target));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(target), _mm_and_si128(current, validity));
            target += 16;
            source += 16;
            byte_count -= 16;
        }
    }
#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && defined(__ARM_NEON)
    {
        const auto shift_right = vdupq_n_s64(-static_cast<int64_t>(shift));
        const auto shift_left = vdupq_n_s64(static_cast<int64_t>(64U - shift));
        while (byte_count >= 16) {
            const auto input_bytes = vld1q_u8(source);
            auto validity = vreinterpretq_u64_u8(input_bytes);
            if (shift != 0) {
                const uint64x2_t carry = {static_cast<uint64_t>(source[16]), 0};
                const auto next = vextq_u64(validity, carry, 1);
                validity = vorrq_u64(vshlq_u64(validity, shift_right), vshlq_u64(next, shift_left));
            }
            const auto current = vld1q_u8(target);
            vst1q_u8(target, vandq_u8(current, vreinterpretq_u8_u64(validity)));
            target += 16;
            source += 16;
            byte_count -= 16;
        }
    }
#endif

    while (byte_count != 0) {
        *target &= ReadShiftedByte(source, shift);
        ++target;
        ++source;
        --byte_count;
    }
}

inline void
AndPackedBits(uint8_t* target, size_t target_bit_begin, const uint8_t* source, size_t source_bit_begin,
              size_t bit_count) {
    while (bit_count != 0 && (target_bit_begin & 7U) != 0) {
        if (!PackedBit(source, source_bit_begin)) {
            target[target_bit_begin >> 3] &= static_cast<uint8_t>(~(1U << (target_bit_begin & 7U)));
        }
        ++target_bit_begin;
        ++source_bit_begin;
        --bit_count;
    }

    const auto byte_count = bit_count >> 3;
    if (byte_count != 0) {
        AndPackedBytes(target + (target_bit_begin >> 3), source + (source_bit_begin >> 3), byte_count,
                       static_cast<unsigned>(source_bit_begin & 7U));
        target_bit_begin += byte_count << 3;
        source_bit_begin += byte_count << 3;
        bit_count -= byte_count << 3;
    }

    while (bit_count != 0) {
        if (!PackedBit(source, source_bit_begin)) {
            target[target_bit_begin >> 3] &= static_cast<uint8_t>(~(1U << (target_bit_begin & 7U)));
        }
        ++target_bit_begin;
        ++source_bit_begin;
        --bit_count;
    }
}

}  // namespace detail

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

    bool
    Test(size_t bit) const {
        if (bit >= size()) {
            return false;
        }
        const auto records = records_.Prefix(records_.size());
        const auto record_id = FindRecord(records, bit);
        return record_id < records.size() && records[record_id].Contains(bit) && records[record_id].Test(bit);
    }

    template <typename Visitor>
    void
    Visit(size_t bit_begin, size_t bit_count, Visitor&& visitor) const {
        const auto visible_count = size();
        if (bit_count == 0 || bit_begin >= visible_count) {
            return;
        }
        const auto bit_end = bit_begin + std::min(bit_count, visible_count - bit_begin);
        const auto records = records_.Prefix(records_.size());
        auto record_id = FindRecord(records, bit_begin);
        while (record_id < records.size()) {
            const auto record = records[record_id];
            if (record.bit_begin >= bit_end) {
                break;
            }
            const auto record_begin = std::max(bit_begin, record.bit_begin);
            const auto record_end = std::min(bit_end, record.bit_begin + record.bit_count);
            for (auto bit = record_begin; bit < record_end; ++bit) {
                visitor(bit, record.Test(bit));
            }
            ++record_id;
        }
    }

    void
    AndRange(size_t bit_begin, size_t bit_count, uint8_t* target, size_t target_bit_begin) const {
        const auto visible_count = size();
        if (bit_count == 0 || bit_begin >= visible_count) {
            return;
        }
        const auto bit_end = bit_begin + std::min(bit_count, visible_count - bit_begin);
        const auto records = records_.Prefix(records_.size());
        auto record_id = FindRecord(records, bit_begin);
        while (record_id < records.size()) {
            const auto record = records[record_id];
            if (record.bit_begin >= bit_end) {
                break;
            }
            const auto record_begin = std::max(bit_begin, record.bit_begin);
            const auto record_end = std::min(bit_end, record.bit_begin + record.bit_count);
            detail::AndPackedBits(target, target_bit_begin + record_begin - bit_begin, record.bytes->data(),
                                  record_begin - record.bit_begin, record_end - record_begin);
            ++record_id;
        }
    }

 private:
    static size_t
    ByteSize(size_t bit_count) {
        return (bit_count + 7) / 8;
    }

    static size_t
    FindRecord(const ArrayStore<BitmapRecord>& records, size_t bit) {
        auto left = static_cast<size_t>(0);
        auto right = records.size();
        while (left < right) {
            const auto middle = left + (right - left) / 2;
            const auto& record = records[middle];
            if (record.bit_begin + record.bit_count <= bit) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        return left;
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

    bool
    Test(size_t bit) const {
        if (bit >= size()) {
            return false;
        }
        if (type_ == Type::ARRAY) {
            return (bytes_[bit >> 3] & (1U << (bit & 7))) != 0;
        }
        return append_data_->Test(bit);
    }

    template <typename Visitor>
    void
    Visit(size_t bit_begin, size_t bit_count, Visitor&& visitor) const {
        const auto visible_count = size();
        if (bit_count == 0 || bit_begin >= visible_count) {
            return;
        }
        const auto visit_count = std::min(bit_count, visible_count - bit_begin);
        if (type_ == Type::APPEND_ARRAY) {
            append_data_->Visit(bit_begin, visit_count, std::forward<Visitor>(visitor));
            return;
        }
        const auto bit_end = bit_begin + visit_count;
        for (auto bit = bit_begin; bit < bit_end; ++bit) {
            visitor(bit, Test(bit));
        }
    }

    void
    AndRange(size_t bit_begin, size_t bit_count, uint8_t* target, size_t target_bit_begin = 0) const {
        const auto visible_count = size();
        if (bit_count == 0 || bit_begin >= visible_count) {
            return;
        }
        const auto visit_count = std::min(bit_count, visible_count - bit_begin);
        if (type_ == Type::APPEND_ARRAY) {
            append_data_->AndRange(bit_begin, visit_count, target, target_bit_begin);
            return;
        }
        detail::AndPackedBits(target, target_bit_begin, bytes_.data(), bit_begin, visit_count);
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
