#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "io/memory_io.h"
#include "knowhere/log.h"

namespace knowhere::sparse::inverted {

inline constexpr uint32_t kInvertedIndexFileFormatVersion = 1;
inline constexpr size_t kInvertedIndexHeaderReservedBytes = 16;
inline constexpr size_t kInvertedIndexFileHeaderSize = sizeof(uint32_t) * 4 + kInvertedIndexHeaderReservedBytes;
inline constexpr size_t kInvertedIndexSectionCountSize = sizeof(uint32_t);

enum class InvertedIndexSectionType : uint32_t {
    POSTING_LISTS = 0,
    METRIC_PARAMS = 1,
    DIM_MAP_REVERSE = 2,
    ROW_SUMS = 3,
    MAX_SCORES_PER_DIM = 4,
    BLOCK_MAX_SCORES = 5,
    PROMETHEUS_BUILD_STATS = 6,
    DIM_MAP_MPHF = 7,
};

struct InvertedIndexSectionHeader {
    InvertedIndexSectionType type;
    uint64_t offset;
    uint64_t size;
};

inline uint64_t
first_section_offset(uint32_t nr_sections) {
    return kInvertedIndexFileHeaderSize + kInvertedIndexSectionCountSize +
           sizeof(InvertedIndexSectionHeader) * nr_sections;
}

inline uint64_t
align_offset(uint64_t offset, size_t alignment) {
    if (alignment <= 1) {
        return offset;
    }
    const auto mask = static_cast<uint64_t>(alignment - 1);
    return (offset + mask) & ~mask;
}

inline uint64_t
align_section_offset(uint64_t offset, InvertedIndexSectionType type) {
    switch (type) {
        case InvertedIndexSectionType::POSTING_LISTS:
            return align_offset(offset + sizeof(uint32_t), alignof(size_t)) - sizeof(uint32_t);
        case InvertedIndexSectionType::DIM_MAP_REVERSE:
            return align_offset(offset, alignof(uint32_t));
        case InvertedIndexSectionType::ROW_SUMS:
        case InvertedIndexSectionType::MAX_SCORES_PER_DIM:
            return align_offset(offset, alignof(float));
        case InvertedIndexSectionType::BLOCK_MAX_SCORES:
            return align_offset(offset + sizeof(size_t) + sizeof(uint32_t), alignof(size_t)) - sizeof(size_t) -
                   sizeof(uint32_t);
        default:
            return offset;
    }
}

inline void
assign_section_offset(InvertedIndexSectionHeader& section_header, uint64_t& used_offset) {
    used_offset = align_section_offset(used_offset, section_header.type);
    section_header.offset = used_offset;
    used_offset += section_header.size;
}

inline void
write_padding_until(MemoryIOWriter& writer, uint64_t target_offset) {
    auto current_offset = static_cast<uint64_t>(writer.tellg());
    assert(current_offset <= target_offset);
    if (current_offset >= target_offset) {
        return;
    }

    std::array<uint8_t, 64> padding{};
    while (current_offset < target_offset) {
        const auto bytes = static_cast<size_t>(std::min<uint64_t>(padding.size(), target_offset - current_offset));
        writer.write(padding.data(), bytes);
        current_offset += bytes;
    }
}

inline std::vector<InvertedIndexSectionHeader>
read_section_headers(MemoryIOReader& reader, uint32_t nr_sections) {
    std::vector<InvertedIndexSectionHeader> section_headers(nr_sections);
    for (auto& section_header : section_headers) {
        reader.read(&section_header, sizeof(InvertedIndexSectionHeader));
    }
    return section_headers;
}

inline const InvertedIndexSectionHeader*
find_section_header(const std::vector<InvertedIndexSectionHeader>& section_headers, InvertedIndexSectionType type) {
    auto it = std::find_if(section_headers.cbegin(), section_headers.cend(),
                           [type](const auto& section_header) { return section_header.type == type; });
    return it == section_headers.cend() ? nullptr : &*it;
}

inline void
log_uint32_stats(const char* index_name, const char* stat_name, const std::vector<uint32_t>& values) {
    if (values.empty()) {
        LOG_KNOWHERE_INFO_ << index_name << "::deserialize " << stat_name << " stats: count=0";
        return;
    }

    uint64_t sum = 0;
    uint32_t min_value = std::numeric_limits<uint32_t>::max();
    uint32_t max_value = 0;
    for (uint32_t value : values) {
        sum += value;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    LOG_KNOWHERE_INFO_ << index_name << "::deserialize " << stat_name << " stats: count=" << values.size()
                       << " min=" << min_value << " max=" << max_value
                       << " avg=" << (static_cast<double>(sum) / static_cast<double>(values.size())) << " sum=" << sum;
}

}  // namespace knowhere::sparse::inverted
