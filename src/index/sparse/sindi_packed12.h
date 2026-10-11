// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace knowhere::sparse::inverted::sindi {

// Two little-endian 12-bit words in three bytes; odd final words use two bytes.
inline size_t
packed12_bytes(size_t count) {
    if (count > (std::numeric_limits<size_t>::max() / 3) * 2) {
        throw std::overflow_error("Packed posting length overflow");
    }

    return (count / 2) * 3 + (count % 2) * 2;
}

inline uint16_t
unpack12(const uint8_t* bytes, size_t position) {
    const size_t offset = (position / 2) * 3 + (position & 1);
    return ((uint16_t(bytes[offset]) | (uint16_t(bytes[offset + 1]) << 8)) >> ((position & 1) * 4)) & 4095;
}

// Sequential writes or independently owned complete pairs only: adjacent words share a byte.
inline void
pack12(uint8_t* bytes, size_t position, uint16_t code) {
    if (code > 4095) {
        throw std::invalid_argument("U12 overflow");
    }

    const size_t offset = (position / 2) * 3;
    if (position & 1) {
        bytes[offset + 1] = (bytes[offset + 1] & 15) | ((code & 15) << 4);
        bytes[offset + 2] = code >> 4;
    } else {
        bytes[offset] = code & 255;
        bytes[offset + 1] = code >> 8;
    }
}

}  // namespace knowhere::sparse::inverted::sindi
