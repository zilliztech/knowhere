#ifndef PTRHASH_PTRHASH_HPP
#define PTRHASH_PTRHASH_HPP

// Based on PtrHash, a minimal perfect hashing scheme.
// Paper: "PtrHash: Minimal Perfect Hashing at RAM Throughput"
// https://arxiv.org/abs/2502.15539
// Reference implementation:
// https://github.com/RagnarGrootKoerkamp/PtrHash
//
// Usage:
//   std::vector<uint64_t> keys = {10, 20, 30};
//   auto hash = ptrhash::PtrHash::build(keys);
//   size_t index = hash.index(20);  // in [0, hash.n()) for keys used to build the hash.
//
// PtrHash does not store the original keys and cannot prove membership by itself.
// If queries may contain keys outside the build set, keep an index-addressed key
// or fingerprint table and verify the candidate returned by index():
//
//   std::vector<uint64_t> keys_by_index(hash.n());
//   for (uint64_t key : keys) {
//       keys_by_index[hash.index(key)] = key;
//   }
//
//   size_t candidate = hash.index(query);
//   bool found = keys_by_index[candidate] == query;
//
// The serialized data contains the pilots and remap table required by index().
// It can be persisted with hash.save(path) and restored with PtrHash::load(path).

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ptrhash {

enum class BucketFunction : uint64_t {
    Linear = 0,
    SquareEps = 1,
    CubicEps = 2,
};

struct PtrHashParams {
    double alpha = 0.99;
    double lambda = 3.0;
    uint64_t seed = 0x3141592653589793ull;
    size_t max_seed_attempts = 256;
    uint16_t max_pilot = 255;
    // 0 means auto. Set to 1 to force single-core construction.
    size_t build_threads = 0;
    BucketFunction bucket_function = BucketFunction::Linear;
};

namespace detail {

// Magic identifies the local PtrHash serialization family. The format revision
// is stored separately in kVersion.
constexpr uint8_t kMagic[8] = {'P', 'T', 'R', 'H', 'A', 'S', 'H', '\0'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderSize = 8 + 4 + 4 + 8 * 9;
constexpr uint64_t kMix = 0x517cc1b727220a95ull;
constexpr uint32_t kRemapU32 = 4;
constexpr uint32_t kBucketFunctionShift = 8;
constexpr uint32_t kKeyHashKindShift = 16;

enum class KeyHashKind : uint32_t {
    Integer = 0,
    String = 1,
    Hash64 = 2,
};

inline uint64_t mul_high(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
#else
    const uint64_t a_lo = static_cast<uint32_t>(a);
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = static_cast<uint32_t>(b);
    const uint64_t b_hi = b >> 32;
    const uint64_t p0 = a_lo * b_lo;
    const uint64_t p1 = a_lo * b_hi;
    const uint64_t p2 = a_hi * b_lo;
    const uint64_t p3 = a_hi * b_hi;
    const uint64_t carry = ((p0 >> 32) + static_cast<uint32_t>(p1) + static_cast<uint32_t>(p2)) >> 32;
    return p3 + (p1 >> 32) + (p2 >> 32) + carry;
#endif
}

inline size_t fast_reduce(uint64_t d, uint64_t h) {
    return static_cast<size_t>(mul_high(d, h));
}

inline uint64_t fastmod32_multiplier(size_t d) {
    return (std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(d)) + 1;
}

inline size_t fastmod32_reduce(uint64_t d, uint64_t m, uint64_t h) {
    const uint64_t lowbits = m * h;
#if defined(__SIZEOF_INT128__)
    return static_cast<size_t>((static_cast<unsigned __int128>(lowbits) * d) >> 64);
#else
    return fast_reduce(d, lowbits);
#endif
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

inline uint64_t hash_key(uint64_t key, uint64_t seed) {
#if defined(__SIZEOF_INT128__)
    const auto r = static_cast<unsigned __int128>(key ^ seed) * kMix;
    const auto low = static_cast<uint64_t>(r);
    const auto high = static_cast<uint64_t>(r >> 64);
    return (low ^ high) * kMix;
#else
    return splitmix64(key ^ seed);
#endif
}

inline uint64_t read_u64_unaligned(const uint8_t* p) {
    uint64_t value = 0;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(&value, p, sizeof(value));
#else
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
#endif
    return value;
}

inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

inline uint64_t fmix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

inline uint64_t hash_bytes(std::string_view key, uint64_t seed) {
    const auto* p = reinterpret_cast<const uint8_t*>(key.data());
    const size_t len = key.size();
    constexpr uint64_t c1 = 0x87c37b91114253d5ull;
    constexpr uint64_t c2 = 0x4cf5ad432745937full;
    uint64_t h1 = seed;
    uint64_t h2 = seed ^ (static_cast<uint64_t>(len) * kMix);

    size_t remaining = len;
    while (remaining >= 16) {
        uint64_t k1 = read_u64_unaligned(p);
        uint64_t k2 = read_u64_unaligned(p + 8);

        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;
        h2 = rotl64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;

        p += 16;
        remaining -= 16;
    }

    uint64_t k1 = 0;
    uint64_t k2 = 0;
    const size_t first_tail = std::min<size_t>(remaining, 8);
    for (size_t i = 0; i < first_tail; ++i) {
        k1 |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    for (size_t i = 8; i < remaining; ++i) {
        k2 |= static_cast<uint64_t>(p[i]) << (8 * (i - 8));
    }
    if (k2 != 0) {
        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;
    }
    if (k1 != 0) {
        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;
    }

    h1 ^= static_cast<uint64_t>(len);
    h2 ^= static_cast<uint64_t>(len);
    h1 += h2;
    h2 += h1;
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 += h2;
    return h1;
}

template<typename Key>
inline typename std::enable_if<std::is_integral<Key>::value && sizeof(Key) <= sizeof(uint64_t), uint64_t>::type
hash_key_for(Key key, uint64_t seed) {
    return hash_key(static_cast<uint64_t>(key), seed);
}

inline uint64_t hash_key_for(std::string_view key, uint64_t seed) {
    return hash_bytes(key, seed);
}

inline uint64_t hash_pilot(uint64_t pilot, uint64_t seed) {
    return kMix * (pilot ^ seed);
}

inline bool likely(bool value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_expect(value, true);
#else
    return value;
#endif
}

inline uint64_t bucket_transform(uint64_t x, BucketFunction bucket_function) {
    switch (bucket_function) {
        case BucketFunction::Linear:
            return x;
        case BucketFunction::SquareEps:
            return mul_high(x, x) / 256 * 255 + x / 256;
        case BucketFunction::CubicEps:
            return mul_high(mul_high(x, x), (x >> 1) | (1ull << 63)) / 256 * 255 + x / 256;
    }
    throw std::invalid_argument("unknown bucket function");
}

inline void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

inline void append_u64(std::vector<uint8_t>& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

inline uint32_t read_u32(const uint8_t* p) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(p[i]) << (8 * i);
    }
    return value;
}

inline uint64_t read_u64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return value;
}

inline size_t ceil_division_as_size(double numerator, double denominator) {
    if (!std::isfinite(denominator) || !(denominator > 0.0)) {
        throw std::invalid_argument("PtrHash parameter must be positive");
    }
    if (!std::isfinite(numerator) || numerator < 0.0) {
        throw std::overflow_error("PtrHash size overflow");
    }
    const double value = numerator / denominator;
    if (!std::isfinite(value) || value < 0.0 || !(value < static_cast<double>(std::numeric_limits<size_t>::max()))) {
        throw std::overflow_error("PtrHash size overflow");
    }
    const size_t truncated = static_cast<size_t>(value);
    if (static_cast<double>(truncated) == value) {
        return truncated;
    }
    if (truncated == std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("PtrHash size overflow");
    }
    return truncated + 1;
}

inline size_t checked_multiply_as_size(size_t lhs, size_t rhs, const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

inline size_t choose_parts(size_t n, double alpha) {
    if (n == 0) {
        return 0;
    }
    const double eps = (1.0 - alpha) / 2.0;
    const double x = static_cast<double>(n) * eps * eps / 2.0;
    if (!(x > std::exp(1.0))) {
        return 1;
    }
    double target_parts = x / std::log(x);
    if (!(target_parts >= 1.0)) {
        target_parts = 1.0;
    }
    const size_t compression_parts = std::max<size_t>(1, static_cast<size_t>(std::floor(target_parts)));
    const size_t parallel_parts = 1;
    return std::max<size_t>(compression_parts, std::max<size_t>(1, parallel_parts));
}

}  // namespace detail

class PtrHashView {
public:
    PtrHashView() = default;

    static PtrHashView from_bytes(const void* data, size_t size) {
        if (data == nullptr && size != 0) {
            throw std::invalid_argument("PtrHashView data is null");
        }
        const auto* bytes = static_cast<const uint8_t*>(data);
        if (size < detail::kHeaderSize) {
            throw std::invalid_argument("PtrHash data is truncated");
        }
        if (!std::equal(detail::kMagic, detail::kMagic + 8, bytes)) {
            throw std::invalid_argument("PtrHash magic mismatch");
        }
        if (detail::read_u32(bytes + 8) != detail::kVersion) {
            throw std::invalid_argument("unsupported PtrHash version");
        }

        const uint32_t flags = detail::read_u32(bytes + 12);
        const uint32_t remap_width = flags & 0xffu;
        const auto bucket_function = static_cast<BucketFunction>((flags >> detail::kBucketFunctionShift) & 0xffu);
        const auto key_hash_kind = static_cast<detail::KeyHashKind>((flags >> detail::kKeyHashKindShift) & 0xffu);
        if (remap_width != detail::kRemapU32) {
            throw std::invalid_argument("unsupported PtrHash remap width");
        }
        (void)detail::bucket_transform(0, bucket_function);
        switch (key_hash_kind) {
            case detail::KeyHashKind::Integer:
            case detail::KeyHashKind::String:
            case detail::KeyHashKind::Hash64:
                break;
            default:
                throw std::invalid_argument("unsupported PtrHash key hash kind");
        }

        PtrHashView view;
        view.data_ = bytes;
        view.capacity_ = size;
        view.bucket_function_ = bucket_function;
        view.key_hash_kind_ = key_hash_kind;
        const uint8_t* cursor = bytes + 16;
        view.n_ = static_cast<size_t>(detail::read_u64(cursor));
        cursor += 8;
        view.slots_total_ = static_cast<size_t>(detail::read_u64(cursor));
        cursor += 8;
        view.buckets_total_ = static_cast<size_t>(detail::read_u64(cursor));
        cursor += 8;
        view.seed_ = detail::read_u64(cursor);
        cursor += 8;
        view.pilot_count_ = static_cast<size_t>(detail::read_u64(cursor));
        cursor += 8;
        view.remap_count_ = static_cast<size_t>(detail::read_u64(cursor));
        cursor += 8;
        view.parts_ = static_cast<size_t>(detail::read_u64(cursor));
        cursor += 8;
        view.slots_per_part_ = static_cast<size_t>(detail::read_u64(cursor));
        cursor += 8;
        view.buckets_per_part_ = static_cast<size_t>(detail::read_u64(cursor));
        view.rem_slots_m_ = view.slots_per_part_ == 0 ? 0 : detail::fastmod32_multiplier(view.slots_per_part_);
        for (size_t pilot = 0; pilot < view.pilot_hashes_.size(); ++pilot) {
            view.pilot_hashes_[pilot] = detail::hash_pilot(pilot, view.seed_);
        }

        if (view.n_ == 0 &&
            (view.slots_total_ != 0 || view.buckets_total_ != 0 || view.pilot_count_ != 0 || view.remap_count_ != 0 ||
             view.parts_ != 0 || view.slots_per_part_ != 0 || view.buckets_per_part_ != 0)) {
            throw std::invalid_argument("PtrHash empty layout mismatch");
        }
        if (view.parts_ == 0 && view.n_ != 0) {
            throw std::invalid_argument("PtrHash part count is invalid");
        }
        if (view.n_ != 0 && (view.parts_ == 0 || view.slots_per_part_ == 0 || view.buckets_per_part_ == 0 ||
                             view.slots_total_ == 0 || view.buckets_total_ == 0 || view.pilot_count_ == 0)) {
            throw std::invalid_argument("PtrHash non-empty layout has zero counts");
        }
        if (view.parts_ != 0 && view.slots_per_part_ != 0 &&
            view.parts_ > std::numeric_limits<size_t>::max() / view.slots_per_part_) {
            throw std::overflow_error("PtrHash slot layout overflow");
        }
        if (view.parts_ != 0 && view.buckets_per_part_ != 0 &&
            view.parts_ > std::numeric_limits<size_t>::max() / view.buckets_per_part_) {
            throw std::overflow_error("PtrHash bucket layout overflow");
        }
        if (view.parts_ != 0 && view.slots_total_ != view.parts_ * view.slots_per_part_) {
            throw std::invalid_argument("PtrHash slot layout mismatch");
        }
        if (view.parts_ != 0 && view.buckets_total_ != view.parts_ * view.buckets_per_part_) {
            throw std::invalid_argument("PtrHash bucket layout mismatch");
        }
        if (view.pilot_count_ != view.buckets_total_) {
            throw std::invalid_argument("PtrHash pilot count mismatch");
        }
        if (view.slots_total_ < view.n_) {
            throw std::invalid_argument("PtrHash slot count is invalid");
        }
        if (view.remap_count_ != view.slots_total_ - view.n_) {
            throw std::invalid_argument("PtrHash remap count mismatch");
        }
        const size_t pilot_bytes = view.pilot_count_;
        if (view.pilot_count_ != 0 && pilot_bytes / sizeof(uint8_t) != view.pilot_count_) {
            throw std::overflow_error("PtrHash pilot size overflow");
        }
        const size_t remap_bytes = view.remap_count_ * sizeof(uint32_t);
        if (view.remap_count_ != 0 && remap_bytes / sizeof(uint32_t) != view.remap_count_) {
            throw std::overflow_error("PtrHash remap size overflow");
        }
        if (pilot_bytes > std::numeric_limits<size_t>::max() - detail::kHeaderSize ||
            remap_bytes > std::numeric_limits<size_t>::max() - detail::kHeaderSize - pilot_bytes) {
            throw std::overflow_error("PtrHash serialized size overflow");
        }
        view.serialized_size_ = detail::kHeaderSize + pilot_bytes + remap_bytes;
        if (view.serialized_size_ > size) {
            throw std::invalid_argument("PtrHash data is truncated");
        }
        view.pilots_ = bytes + detail::kHeaderSize;
        view.remap_ = view.pilots_ + pilot_bytes;
        for (size_t i = 0; i < view.remap_count_; ++i) {
            if (detail::read_u32(view.remap_ + i * sizeof(uint32_t)) >= view.n_) {
                throw std::invalid_argument("PtrHash remap entry is invalid");
            }
        }
        return view;
    }

    size_t n() const {
        return n_;
    }

    size_t max_index() const {
        return slots_total_;
    }

    size_t bucket_count() const {
        return buckets_total_;
    }

    size_t serialized_size() const {
        return serialized_size_;
    }

    size_t index_no_remap(uint64_t key) const {
        require_key_hash_kind(detail::KeyHashKind::Integer);
        return index_no_remap_hx(detail::hash_key(key, seed_));
    }

    size_t index_no_remap(std::string_view key) const {
        require_key_hash_kind(detail::KeyHashKind::String);
        return index_no_remap_hx(detail::hash_key_for(key, seed_));
    }

    size_t index_no_remap_hash(uint64_t hash) const {
        require_key_hash_kind(detail::KeyHashKind::Hash64);
        return index_no_remap_hx(detail::hash_key(hash, seed_));
    }

    size_t index(uint64_t key) const {
        return index_from_slot(index_no_remap(key));
    }

    size_t index(std::string_view key) const {
        return index_from_slot(index_no_remap(key));
    }

    size_t index_hash(uint64_t hash) const {
        return index_from_slot(index_no_remap_hash(hash));
    }

private:
    void require_key_hash_kind(detail::KeyHashKind expected) const {
        if (key_hash_kind_ != expected) {
            throw std::invalid_argument("PtrHash key type does not match this data");
        }
    }

    size_t index_no_remap_hx(uint64_t hx) const {
        if (n_ == 0) {
            throw std::out_of_range("cannot query an empty PtrHash");
        }
        const size_t part = detail::fast_reduce(static_cast<uint64_t>(parts_), hx);
        const size_t bucket =
            bucket_function_ == BucketFunction::Linear
                ? detail::fast_reduce(static_cast<uint64_t>(buckets_total_), hx)
                : part * buckets_per_part_ +
                      detail::fast_reduce(
                          static_cast<uint64_t>(buckets_per_part_),
                          detail::bucket_transform(detail::splitmix64(hx ^ 0x243f6a8885a308d3ull), bucket_function_)
                      );
        const uint64_t pilot = pilots_[bucket];
        const size_t slot_in_part =
            detail::fastmod32_reduce(static_cast<uint64_t>(slots_per_part_), rem_slots_m_, hx ^ pilot_hashes_[pilot]);
        return part * slots_per_part_ + slot_in_part;
    }

    size_t index_from_slot(size_t slot) const {
        if (detail::likely(slot < n_)) {
            return slot;
        }
        return static_cast<size_t>(detail::read_u32(remap_ + (slot - n_) * 4));
    }

    const uint8_t* data_ = nullptr;
    const uint8_t* pilots_ = nullptr;
    const uint8_t* remap_ = nullptr;
    size_t capacity_ = 0;
    size_t serialized_size_ = 0;
    size_t n_ = 0;
    size_t slots_total_ = 0;
    size_t buckets_total_ = 0;
    size_t pilot_count_ = 0;
    size_t remap_count_ = 0;
    size_t parts_ = 0;
    size_t slots_per_part_ = 0;
    size_t buckets_per_part_ = 0;
    uint64_t rem_slots_m_ = 0;
    uint64_t seed_ = 0;
    std::array<uint64_t, 256> pilot_hashes_{};
    BucketFunction bucket_function_ = BucketFunction::Linear;
    detail::KeyHashKind key_hash_kind_ = detail::KeyHashKind::Integer;
};

class PtrHash {
public:
    PtrHash() = default;

    PtrHash(const PtrHash& other) : storage_(other.storage_) {
        reset_view();
    }

    PtrHash& operator=(const PtrHash& other) {
        if (this != &other) {
            storage_ = other.storage_;
            reset_view();
        }
        return *this;
    }

    PtrHash(PtrHash&& other) noexcept : storage_(std::move(other.storage_)) {
        reset_view();
    }

    PtrHash& operator=(PtrHash&& other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            reset_view();
        }
        return *this;
    }

    template<
        typename Key,
        typename std::enable_if<std::is_integral<Key>::value && sizeof(Key) <= sizeof(uint64_t), int>::type = 0>
    static PtrHash build(const std::vector<Key>& keys, const PtrHashParams& params = PtrHashParams()) {
        return build_impl(keys, params, detail::KeyHashKind::Integer);
    }

    static PtrHash build(const std::vector<std::string>& keys, const PtrHashParams& params = PtrHashParams()) {
        return build_impl(keys, params, detail::KeyHashKind::String);
    }

    static PtrHash build(const std::vector<std::string_view>& keys, const PtrHashParams& params = PtrHashParams()) {
        return build_impl(keys, params, detail::KeyHashKind::String);
    }

    static PtrHash build_hashes(const std::vector<uint64_t>& hashes, const PtrHashParams& params = PtrHashParams()) {
        return build_impl(hashes, params, detail::KeyHashKind::Hash64);
    }

private:
    using BucketId = uint32_t;

    template<typename Key>
    static PtrHash
    build_impl(const std::vector<Key>& keys, const PtrHashParams& params, detail::KeyHashKind key_hash_kind) {
        validate_params(params);
        validate_unique(keys);
        if (keys.empty()) {
            return from_parts(0, 0, 0, 0, 0, 0, params.seed, params.bucket_function, key_hash_kind, {}, {});
        }

        const size_t n = keys.size();
        if (n > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::overflow_error("this compact serialized format supports up to 2^32-1 keys");
        }
        const size_t parts = detail::choose_parts(n, params.alpha);

        for (size_t attempt = 0; attempt < params.max_seed_attempts; ++attempt) {
            const uint64_t seed = detail::splitmix64(params.seed + attempt);
            size_t slots_total = 0;
            size_t buckets_total = 0;
            size_t slots_per_part = 0;
            size_t buckets_per_part = 0;
            std::vector<uint8_t> pilots;
            std::vector<uint32_t> remap;
            if (try_build(
                    keys,
                    parts,
                    params.alpha,
                    params.lambda,
                    seed,
                    params.max_pilot,
                    params.build_threads,
                    params.bucket_function,
                    slots_total,
                    buckets_total,
                    slots_per_part,
                    buckets_per_part,
                    pilots,
                    remap
                )) {
                return from_parts(
                    n,
                    slots_total,
                    buckets_total,
                    parts,
                    slots_per_part,
                    buckets_per_part,
                    seed,
                    params.bucket_function,
                    key_hash_kind,
                    std::move(pilots),
                    std::move(remap)
                );
            }
        }
        throw std::runtime_error("unable to construct PtrHash with the requested parameters");
    }

public:
    static PtrHash deserialize(const void* data, size_t size) {
        PtrHashView view = PtrHashView::from_bytes(data, size);
        const auto* bytes = static_cast<const uint8_t*>(data);
        PtrHash hash;
        hash.storage_.assign(bytes, bytes + view.serialized_size());
        hash.reset_view();
        return hash;
    }

    static PtrHash load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open PtrHash file");
        }
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return deserialize(bytes.data(), bytes.size());
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("failed to create PtrHash file");
        }
        out.write(reinterpret_cast<const char*>(storage_.data()), static_cast<std::streamsize>(storage_.size()));
        if (!out.good()) {
            throw std::runtime_error("failed to write PtrHash file");
        }
    }

    size_t n() const {
        return view_.n();
    }

    size_t max_index() const {
        return view_.max_index();
    }

    size_t index_no_remap(uint64_t key) const {
        return view_.index_no_remap(key);
    }

    size_t index_no_remap(std::string_view key) const {
        return view_.index_no_remap(key);
    }

    size_t index_no_remap_hash(uint64_t hash) const {
        return view_.index_no_remap_hash(hash);
    }

    size_t index(uint64_t key) const {
        return view_.index(key);
    }

    size_t index(std::string_view key) const {
        return view_.index(key);
    }

    size_t index_hash(uint64_t hash) const {
        return view_.index_hash(hash);
    }

    const PtrHashView& view() const {
        return view_;
    }

    const std::vector<uint8_t>& serialize() const {
        return storage_;
    }

private:
    static void validate_params(const PtrHashParams& params) {
        if (!std::isfinite(params.alpha) || !(params.alpha > 0.0 && params.alpha <= 1.0)) {
            throw std::invalid_argument("alpha must be in (0, 1]");
        }
        if (!std::isfinite(params.lambda) || !(params.lambda > 0.0)) {
            throw std::invalid_argument("lambda must be positive");
        }
        if (params.max_pilot > std::numeric_limits<uint8_t>::max()) {
            throw std::invalid_argument("max_pilot must fit in the u8 serialized pilot format");
        }
    }

    template<typename Key>
    static void validate_unique(const std::vector<Key>& keys) {
        if (std::adjacent_find(keys.begin(), keys.end(), std::greater_equal<Key>()) == keys.end()) {
            return;
        }
        std::vector<Key> sorted = keys;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument("PtrHash requires unique keys");
        }
    }

    template<typename Key>
    static bool try_build(
        const std::vector<Key>& keys,
        size_t parts,
        double alpha,
        double lambda,
        uint64_t seed,
        uint16_t max_pilot,
        size_t build_threads,
        BucketFunction bucket_function,
        size_t& slots_total,
        size_t& buckets_total,
        size_t& slots_per_part,
        size_t& buckets_per_part,
        std::vector<uint8_t>& pilots,
        std::vector<uint32_t>& remap
    ) {
        const size_t keys_per_part = std::max<size_t>(1, (keys.size() + parts - 1) / parts);
        slots_per_part = std::max<size_t>(1, detail::ceil_division_as_size(static_cast<double>(keys_per_part), alpha));
        if ((slots_per_part & (slots_per_part - 1)) == 0) {
            ++slots_per_part;
        }
        const size_t bucket_base = detail::ceil_division_as_size(static_cast<double>(keys_per_part), lambda);
        if (bucket_base > std::numeric_limits<size_t>::max() - 3) {
            throw std::overflow_error("PtrHash size overflow");
        }
        buckets_per_part = std::max<size_t>(1, bucket_base + 3);
        if (buckets_per_part > static_cast<size_t>(std::numeric_limits<BucketId>::max())) {
            throw std::overflow_error("too many buckets per part for compact build state");
        }
        slots_total = detail::checked_multiply_as_size(parts, slots_per_part, "PtrHash slot layout overflow");
        buckets_total = detail::checked_multiply_as_size(parts, buckets_per_part, "PtrHash bucket layout overflow");
        const uint64_t rem_slots_m = detail::fastmod32_multiplier(slots_per_part);
        std::array<uint64_t, 256> pilot_hashes{};
        for (size_t pilot = 0; pilot <= max_pilot; ++pilot) {
            pilot_hashes[pilot] = detail::hash_pilot(pilot, seed);
        }

        std::vector<uint32_t> bucket_starts(buckets_total + 1, 0);
        std::vector<uint64_t> bucket_hashes(keys.size());
        fill_buckets(
            keys,
            parts,
            buckets_per_part,
            buckets_total,
            seed,
            bucket_function,
            build_threads,
            bucket_starts,
            bucket_hashes
        );

        pilots.assign(buckets_total, 0);
        std::vector<uint8_t> taken(slots_total, 0);
        std::atomic<size_t> next_part{0};
        std::atomic<bool> ok{true};
        const size_t thread_count = effective_thread_count(build_threads, parts);
        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        try {
            for (size_t thread = 0; thread < thread_count; ++thread) {
                workers.emplace_back([&] {
                    while (ok.load(std::memory_order_relaxed)) {
                        const size_t part = next_part.fetch_add(1, std::memory_order_relaxed);
                        if (part >= parts) {
                            break;
                        }
                        if (!build_part(
                                part,
                                buckets_per_part,
                                slots_per_part,
                                rem_slots_m,
                                max_pilot,
                                pilot_hashes,
                                bucket_hashes,
                                bucket_starts,
                                pilots,
                                taken
                            )) {
                            ok.store(false, std::memory_order_relaxed);
                            break;
                        }
                    }
                });
            }
        } catch (...) {
            join_workers(workers);
            throw;
        }
        join_workers(workers);
        if (!ok.load(std::memory_order_relaxed)) {
            return false;
        }

        const size_t remap_count = slots_total - keys.size();
        std::vector<uint32_t> free_minimal;
        free_minimal.reserve(remap_count);
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!taken[i]) {
                free_minimal.push_back(static_cast<uint32_t>(i));
            }
        }

        remap.assign(remap_count, 0);
        size_t free_cursor = 0;
        for (size_t slot = keys.size(); slot < slots_total; ++slot) {
            if (taken[slot]) {
                if (free_cursor >= free_minimal.size()) {
                    return false;
                }
                remap[slot - keys.size()] = free_minimal[free_cursor++];
            }
        }
        return free_cursor == free_minimal.size();
    }

    static void join_workers(std::vector<std::thread>& workers) noexcept {
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    static size_t hardware_threads() {
        return std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
    }

    static size_t effective_thread_count(size_t requested, size_t limit) {
        const size_t wanted = requested == 0 ? hardware_threads() : requested;
        return std::max<size_t>(1, std::min<size_t>(wanted, std::max<size_t>(1, limit)));
    }

    static size_t build_thread_count(size_t n, size_t requested) {
        return effective_thread_count(requested, (n + 99999) / 100000);
    }

    static size_t bucket_for_hash(
        uint64_t hx,
        size_t parts,
        size_t buckets_per_part,
        size_t buckets_total,
        BucketFunction bucket_function
    ) {
        if (bucket_function == BucketFunction::Linear) {
            return detail::fast_reduce(static_cast<uint64_t>(buckets_total), hx);
        }
        return detail::fast_reduce(static_cast<uint64_t>(parts), hx) * buckets_per_part +
               detail::fast_reduce(
                   static_cast<uint64_t>(buckets_per_part),
                   detail::bucket_transform(detail::splitmix64(hx ^ 0x243f6a8885a308d3ull), bucket_function)
               );
    }

    template<typename Key>
    static void fill_buckets(
        const std::vector<Key>& keys,
        size_t parts,
        size_t buckets_per_part,
        size_t buckets_total,
        uint64_t seed,
        BucketFunction bucket_function,
        size_t build_threads,
        std::vector<uint32_t>& bucket_starts,
        std::vector<uint64_t>& bucket_hashes
    ) {
        const size_t thread_count = build_thread_count(keys.size(), build_threads);
        if (thread_count == 1) {
            for (const auto& key : keys) {
                const uint64_t hx = detail::hash_key_for(key, seed);
                ++bucket_starts[bucket_for_hash(hx, parts, buckets_per_part, buckets_total, bucket_function) + 1];
            }
            for (size_t i = 1; i < bucket_starts.size(); ++i) {
                bucket_starts[i] += bucket_starts[i - 1];
            }
            std::vector<uint32_t> cursor = bucket_starts;
            for (const auto& key : keys) {
                const uint64_t hx = detail::hash_key_for(key, seed);
                const size_t bucket = bucket_for_hash(hx, parts, buckets_per_part, buckets_total, bucket_function);
                bucket_hashes[cursor[bucket]++] = hx;
            }
            return;
        }

        std::vector<std::atomic<uint32_t>> counts(buckets_total);
        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        try {
            for (size_t thread = 0; thread < thread_count; ++thread) {
                const size_t begin = keys.size() * thread / thread_count;
                const size_t end = keys.size() * (thread + 1) / thread_count;
                workers.emplace_back([&, begin, end] {
                    for (size_t i = begin; i < end; ++i) {
                        const uint64_t hx = detail::hash_key_for(keys[i], seed);
                        const size_t bucket =
                            bucket_for_hash(hx, parts, buckets_per_part, buckets_total, bucket_function);
                        counts[bucket].fetch_add(1, std::memory_order_relaxed);
                    }
                });
            }
        } catch (...) {
            join_workers(workers);
            throw;
        }
        join_workers(workers);

        for (size_t i = 0; i < buckets_total; ++i) {
            bucket_starts[i + 1] = bucket_starts[i] + counts[i].load(std::memory_order_relaxed);
        }

        std::vector<std::atomic<uint32_t>> cursor(buckets_total);
        for (size_t i = 0; i < buckets_total; ++i) {
            cursor[i].store(bucket_starts[i], std::memory_order_relaxed);
        }

        workers.clear();
        try {
            for (size_t thread = 0; thread < thread_count; ++thread) {
                const size_t begin = keys.size() * thread / thread_count;
                const size_t end = keys.size() * (thread + 1) / thread_count;
                workers.emplace_back([&, begin, end] {
                    for (size_t i = begin; i < end; ++i) {
                        const uint64_t hx = detail::hash_key_for(keys[i], seed);
                        const size_t bucket =
                            bucket_for_hash(hx, parts, buckets_per_part, buckets_total, bucket_function);
                        const uint32_t pos = cursor[bucket].fetch_add(1, std::memory_order_relaxed);
                        bucket_hashes[pos] = hx;
                    }
                });
            }
        } catch (...) {
            join_workers(workers);
            throw;
        }
        join_workers(workers);
    }

    static size_t slot_in_part_hp(uint64_t hx, uint64_t rem_slots_m, uint64_t pilot_hash, size_t slots_per_part) {
        return detail::fastmod32_reduce(static_cast<uint64_t>(slots_per_part), rem_slots_m, hx ^ pilot_hash);
    }

    static size_t
    slot_in_part(uint64_t hx, uint64_t seed, uint64_t rem_slots_m, uint16_t pilot, size_t slots_per_part) {
        return slot_in_part_hp(hx, rem_slots_m, detail::hash_pilot(pilot, seed), slots_per_part);
    }

    static bool bucket_slots(
        const std::vector<uint64_t>& hashes,
        size_t begin,
        size_t end,
        uint64_t rem_slots_m,
        uint64_t pilot_hash,
        size_t slots_per_part,
        std::vector<size_t>& out
    ) {
        out.clear();
        out.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            const uint64_t hx = hashes[i];
            const size_t slot = slot_in_part_hp(hx, rem_slots_m, pilot_hash, slots_per_part);
            if (std::find(out.begin(), out.end(), slot) != out.end()) {
                return false;
            }
            out.push_back(slot);
        }
        return true;
    }

    static bool bucket_slots_available(
        const std::vector<uint64_t>& hashes,
        size_t begin,
        size_t end,
        uint64_t rem_slots_m,
        uint64_t pilot_hash,
        size_t slots_per_part,
        const uint8_t* taken_part
    ) {
        size_t i = begin;
        const size_t unrolled_end = begin + ((end - begin) / 4) * 4;
        for (; i < unrolled_end; i += 4) {
            const size_t slot0 = slot_in_part_hp(hashes[i], rem_slots_m, pilot_hash, slots_per_part);
            const size_t slot1 = slot_in_part_hp(hashes[i + 1], rem_slots_m, pilot_hash, slots_per_part);
            const size_t slot2 = slot_in_part_hp(hashes[i + 2], rem_slots_m, pilot_hash, slots_per_part);
            const size_t slot3 = slot_in_part_hp(hashes[i + 3], rem_slots_m, pilot_hash, slots_per_part);
            if (taken_part[slot0] || taken_part[slot1] || taken_part[slot2] || taken_part[slot3]) {
                return false;
            }
        }
        for (; i < end; ++i) {
            const size_t slot = slot_in_part_hp(hashes[i], rem_slots_m, pilot_hash, slots_per_part);
            if (taken_part[slot]) {
                return false;
            }
        }
        return true;
    }

    static bool try_take_bucket_slots(
        const std::vector<uint64_t>& hashes,
        size_t begin,
        size_t end,
        uint64_t rem_slots_m,
        uint64_t pilot_hash,
        size_t slots_per_part,
        uint8_t* taken_part,
        std::vector<size_t>& out
    ) {
        out.clear();
        out.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            const size_t slot = slot_in_part_hp(hashes[i], rem_slots_m, pilot_hash, slots_per_part);
            if (taken_part[slot]) {
                for (size_t taken_slot : out) {
                    taken_part[taken_slot] = 0;
                }
                return false;
            }
            taken_part[slot] = 1;
            out.push_back(slot);
        }
        return true;
    }

    static bool contains_recent(const std::array<BucketId, 16>& recent, BucketId bucket) {
        return std::find(recent.begin(), recent.end(), bucket) != recent.end();
    }

    static bool build_part(
        size_t part,
        size_t buckets_per_part,
        size_t slots_per_part,
        uint64_t rem_slots_m,
        uint16_t max_pilot,
        const std::array<uint64_t, 256>& pilot_hashes,
        const std::vector<uint64_t>& bucket_hashes,
        const std::vector<uint32_t>& bucket_starts,
        std::vector<uint8_t>& pilots,
        std::vector<uint8_t>& taken
    ) {
        const size_t bucket_offset = part * buckets_per_part;
        const size_t slot_offset = part * slots_per_part;
        uint8_t* const taken_part = taken.data() + slot_offset;
        std::vector<BucketId> order(buckets_per_part);
        for (size_t i = 0; i < buckets_per_part; ++i) {
            order[i] = static_cast<BucketId>(i);
        }
        std::stable_sort(order.begin(), order.end(), [&](BucketId a, BucketId b) {
            return bucket_starts[bucket_offset + a + 1] - bucket_starts[bucket_offset + a] >
                   bucket_starts[bucket_offset + b + 1] - bucket_starts[bucket_offset + b];
        });

        std::vector<BucketId> slot_bucket(slots_per_part, bucket_npos());
        std::vector<size_t> candidate_slots;
        std::vector<size_t> remove_slots;
        std::array<BucketId, 16> recent{};

        auto bucket_len = [&](BucketId b) {
            return static_cast<size_t>(bucket_starts[bucket_offset + b + 1] - bucket_starts[bucket_offset + b]);
        };

        for (BucketId new_bucket : order) {
            if (bucket_len(new_bucket) == 0) {
                pilots[bucket_offset + new_bucket] = 0;
                continue;
            }

            std::priority_queue<std::pair<size_t, BucketId>> stack;
            stack.emplace(bucket_len(new_bucket), new_bucket);
            recent.fill(bucket_npos());
            size_t recent_idx = 0;
            recent[recent_idx] = new_bucket;
            size_t evictions = 0;

            while (!stack.empty()) {
                const BucketId bucket = stack.top().second;
                stack.pop();
                const size_t begin = bucket_starts[bucket_offset + bucket];
                const size_t end = bucket_starts[bucket_offset + bucket + 1];

                bool placed = false;
                for (uint32_t pilot_u32 = 0; pilot_u32 <= max_pilot; ++pilot_u32) {
                    const auto pilot = static_cast<uint16_t>(pilot_u32);
                    const uint64_t pilot_hash = pilot_hashes[pilot_u32];
                    if (!bucket_slots_available(
                            bucket_hashes, begin, end, rem_slots_m, pilot_hash, slots_per_part, taken_part
                        )) {
                        continue;
                    }
                    if (!try_take_bucket_slots(
                            bucket_hashes,
                            begin,
                            end,
                            rem_slots_m,
                            pilot_hash,
                            slots_per_part,
                            taken_part,
                            candidate_slots
                        )) {
                        continue;
                    }
                    pilots[bucket_offset + bucket] = static_cast<uint8_t>(pilot);
                    for (size_t slot : candidate_slots) {
                        slot_bucket[slot] = bucket;
                    }
                    placed = true;
                    break;
                }
                if (placed) {
                    continue;
                }

                size_t best_score = std::numeric_limits<size_t>::max();
                uint16_t best_pilot = 0;
                bool have_best = false;
                for (uint32_t pilot_u32 = 0; pilot_u32 <= max_pilot; ++pilot_u32) {
                    const auto pilot = static_cast<uint16_t>(pilot_u32);
                    const uint64_t pilot_hash = pilot_hashes[pilot_u32];
                    if (!bucket_slots(
                            bucket_hashes, begin, end, rem_slots_m, pilot_hash, slots_per_part, candidate_slots
                        )) {
                        continue;
                    }
                    size_t score = 0;
                    bool skip = false;
                    for (size_t slot : candidate_slots) {
                        const BucketId other = slot_bucket[slot];
                        if (other == bucket_npos()) {
                            continue;
                        }
                        if (contains_recent(recent, other)) {
                            skip = true;
                            break;
                        }
                        const size_t len = bucket_len(other);
                        score += len * len;
                        if (score >= best_score) {
                            skip = true;
                            break;
                        }
                    }
                    if (!skip) {
                        best_score = score;
                        best_pilot = pilot;
                        have_best = true;
                    }
                }
                if (!have_best) {
                    return false;
                }

                if (!bucket_slots(
                        bucket_hashes,
                        begin,
                        end,
                        rem_slots_m,
                        pilot_hashes[best_pilot],
                        slots_per_part,
                        candidate_slots
                    )) {
                    return false;
                }
                pilots[bucket_offset + bucket] = static_cast<uint8_t>(best_pilot);
                for (size_t slot : candidate_slots) {
                    const BucketId other = slot_bucket[slot];
                    if (other != bucket_npos() && other != bucket) {
                        stack.emplace(bucket_len(other), other);
                        ++evictions;
                        if (evictions > 10 * slots_per_part) {
                            return false;
                        }
                        const size_t other_begin = bucket_starts[bucket_offset + other];
                        const size_t other_end = bucket_starts[bucket_offset + other + 1];
                        const auto other_pilot = static_cast<uint16_t>(pilots[bucket_offset + other]);
                        if (!bucket_slots(
                                bucket_hashes,
                                other_begin,
                                other_end,
                                rem_slots_m,
                                pilot_hashes[other_pilot],
                                slots_per_part,
                                remove_slots
                            )) {
                            return false;
                        }
                        for (size_t remove_slot : remove_slots) {
                            if (slot_bucket[remove_slot] == other) {
                                slot_bucket[remove_slot] = bucket_npos();
                                taken_part[remove_slot] = false;
                            }
                        }
                    }
                    slot_bucket[slot] = bucket;
                    taken_part[slot] = true;
                }

                recent_idx = (recent_idx + 1) % recent.size();
                recent[recent_idx] = bucket;
            }
        }
        return true;
    }

    static constexpr BucketId bucket_npos() {
        return std::numeric_limits<BucketId>::max();
    }

    static PtrHash from_parts(
        size_t n,
        size_t slots_total,
        size_t buckets_total,
        size_t parts,
        size_t slots_per_part,
        size_t buckets_per_part,
        uint64_t seed,
        BucketFunction bucket_function,
        detail::KeyHashKind key_hash_kind,
        std::vector<uint8_t> pilots,
        std::vector<uint32_t> remap
    ) {
        PtrHash hash;
        const size_t remap_bytes =
            detail::checked_multiply_as_size(remap.size(), sizeof(uint32_t), "PtrHash serialized size overflow");
        if (pilots.size() > std::numeric_limits<size_t>::max() - detail::kHeaderSize ||
            remap_bytes > std::numeric_limits<size_t>::max() - detail::kHeaderSize - pilots.size()) {
            throw std::overflow_error("PtrHash serialized size overflow");
        }
        hash.storage_.reserve(detail::kHeaderSize + pilots.size() + remap_bytes);
        hash.storage_.insert(hash.storage_.end(), detail::kMagic, detail::kMagic + 8);
        detail::append_u32(hash.storage_, detail::kVersion);
        const uint32_t flags = detail::kRemapU32 |
                               (static_cast<uint32_t>(bucket_function) << detail::kBucketFunctionShift) |
                               (static_cast<uint32_t>(key_hash_kind) << detail::kKeyHashKindShift);
        detail::append_u32(hash.storage_, flags);
        detail::append_u64(hash.storage_, static_cast<uint64_t>(n));
        detail::append_u64(hash.storage_, static_cast<uint64_t>(slots_total));
        detail::append_u64(hash.storage_, static_cast<uint64_t>(buckets_total));
        detail::append_u64(hash.storage_, seed);
        detail::append_u64(hash.storage_, static_cast<uint64_t>(pilots.size()));
        detail::append_u64(hash.storage_, static_cast<uint64_t>(remap.size()));
        detail::append_u64(hash.storage_, static_cast<uint64_t>(parts));
        detail::append_u64(hash.storage_, static_cast<uint64_t>(slots_per_part));
        detail::append_u64(hash.storage_, static_cast<uint64_t>(buckets_per_part));
        hash.storage_.insert(hash.storage_.end(), pilots.begin(), pilots.end());
        for (uint32_t value : remap) {
            detail::append_u32(hash.storage_, value);
        }
        hash.reset_view();
        return hash;
    }

    void reset_view() {
        if (storage_.empty()) {
            view_ = PtrHashView();
        } else {
            view_ = PtrHashView::from_bytes(storage_.data(), storage_.size());
        }
    }

    std::vector<uint8_t> storage_;
    PtrHashView view_;
};

template<typename Key, typename Hasher>
class PtrHashWithHasher {
public:
    PtrHashWithHasher() = default;

    PtrHashWithHasher(PtrHash hash, Hasher hasher) : hash_(std::move(hash)), hasher_(std::move(hasher)) {}

    static PtrHashWithHasher
    build(const std::vector<Key>& keys, Hasher hasher = Hasher(), const PtrHashParams& params = PtrHashParams()) {
        std::vector<uint64_t> hashes;
        hashes.reserve(keys.size());
        for (const auto& key : keys) {
            hashes.push_back(to_u64_hash(hasher(key)));
        }
        return PtrHashWithHasher(PtrHash::build_hashes(hashes, params), std::move(hasher));
    }

    static PtrHashWithHasher deserialize(const void* data, size_t size, Hasher hasher = Hasher()) {
        return PtrHashWithHasher(PtrHash::deserialize(data, size), std::move(hasher));
    }

    static PtrHashWithHasher load(const std::string& path, Hasher hasher = Hasher()) {
        return PtrHashWithHasher(PtrHash::load(path), std::move(hasher));
    }

    void save(const std::string& path) const {
        hash_.save(path);
    }

    size_t n() const {
        return hash_.n();
    }

    size_t max_index() const {
        return hash_.max_index();
    }

    size_t index_no_remap(const Key& key) const {
        return hash_.index_no_remap_hash(hash_key(key));
    }

    size_t index(const Key& key) const {
        return hash_.index_hash(hash_key(key));
    }

    const PtrHash& raw() const {
        return hash_;
    }

    const PtrHashView& view() const {
        return hash_.view();
    }

    const std::vector<uint8_t>& serialize() const {
        return hash_.serialize();
    }

private:
    template<typename Value>
    static uint64_t to_u64_hash(Value value) {
        using Decayed = typename std::decay<Value>::type;
        static_assert(
            std::is_integral<Decayed>::value && sizeof(Decayed) <= sizeof(uint64_t),
            "PtrHashWithHasher hasher must return an integral value "
            "that fits in uint64_t"
        );
        return static_cast<uint64_t>(value);
    }

    uint64_t hash_key(const Key& key) const {
        return to_u64_hash(hasher_(key));
    }

    PtrHash hash_;
    Hasher hasher_;
};

class MappedPtrHash {
public:
    MappedPtrHash() = default;
    MappedPtrHash(const MappedPtrHash&) = delete;
    MappedPtrHash& operator=(const MappedPtrHash&) = delete;

    MappedPtrHash(MappedPtrHash&& other) noexcept {
        move_from(std::move(other));
    }

    MappedPtrHash& operator=(MappedPtrHash&& other) noexcept {
        if (this != &other) {
            close();
            move_from(std::move(other));
        }
        return *this;
    }

    ~MappedPtrHash() {
        close();
    }

    static MappedPtrHash open(const std::string& path, size_t offset = 0) {
#if defined(__unix__) || defined(__APPLE__)
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open PtrHash mmap file");
        }
        struct stat st;
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("failed to stat PtrHash mmap file");
        }
        if (st.st_size < 0) {
            ::close(fd);
            throw std::invalid_argument("PtrHash mmap file size is invalid");
        }
        const auto file_size = static_cast<uintmax_t>(st.st_size);
        if (file_size > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
            ::close(fd);
            throw std::overflow_error("PtrHash mmap file is too large");
        }
        const size_t length = static_cast<size_t>(file_size);
        if (offset > length) {
            ::close(fd);
            throw std::invalid_argument("PtrHash mmap offset is past end of file");
        }
        if (length == 0) {
            ::close(fd);
            throw std::invalid_argument("PtrHash mmap file is empty");
        }
        void* mapping = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (mapping == MAP_FAILED) {
            throw std::runtime_error("failed to mmap PtrHash file");
        }
        MappedPtrHash result;
        result.mapping_ = mapping;
        result.mapping_size_ = length;
        try {
            result.view_ = PtrHashView::from_bytes(static_cast<const uint8_t*>(mapping) + offset, length - offset);
        } catch (...) {
            result.close();
            throw;
        }
        return result;
#else
        (void)path;
        (void)offset;
        throw std::runtime_error("mmap loading is only available on POSIX platforms");
#endif
    }

    const PtrHashView& view() const {
        return view_;
    }

    size_t n() const {
        return view_.n();
    }

    size_t max_index() const {
        return view_.max_index();
    }

    size_t index_no_remap(uint64_t key) const {
        return view_.index_no_remap(key);
    }

    size_t index_no_remap(std::string_view key) const {
        return view_.index_no_remap(key);
    }

    size_t index_no_remap_hash(uint64_t hash) const {
        return view_.index_no_remap_hash(hash);
    }

    size_t index(uint64_t key) const {
        return view_.index(key);
    }

    size_t index(std::string_view key) const {
        return view_.index(key);
    }

    size_t index_hash(uint64_t hash) const {
        return view_.index_hash(hash);
    }

private:
    void close() noexcept {
#if defined(__unix__) || defined(__APPLE__)
        if (mapping_ != nullptr && mapping_size_ != 0) {
            ::munmap(mapping_, mapping_size_);
        }
#endif
        mapping_ = nullptr;
        mapping_size_ = 0;
        view_ = PtrHashView();
    }

    void move_from(MappedPtrHash&& other) noexcept {
        mapping_ = other.mapping_;
        mapping_size_ = other.mapping_size_;
        view_ = other.view_;
        other.mapping_ = nullptr;
        other.mapping_size_ = 0;
        other.view_ = PtrHashView();
    }

    void* mapping_ = nullptr;
    size_t mapping_size_ = 0;
    PtrHashView view_;
};

}  // namespace ptrhash

#endif  // PTRHASH_PTRHASH_HPP
