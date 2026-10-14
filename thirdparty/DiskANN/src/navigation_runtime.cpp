// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "diskann/navigation_runtime.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include "diskann/ann_exception.h"
#include "diskann/pq_navigation.h"

namespace diskann {
  namespace {
    constexpr std::array<uint8_t, 8> kMagic = {'K', 'H', 'N', 'A',
                                               'V', 'I', 'D', 'X'};
    constexpr uint32_t               kVersion = 1;
    constexpr uint32_t               kMaxCodecLength = 32;

    [[noreturn]] void FileError(const std::string& path) {
      auto error = std::system_error(std::make_error_code(std::errc::io_error),
                                     "Cannot access navigation metadata");
      throw FileException(path, error, __FUNCSIG__, __FILE__, __LINE__);
    }

    uint32_t ReadU32(const uint8_t* bytes) {
      return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
             uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
    }

    void WriteU32(uint8_t* bytes, uint32_t value) {
      for (unsigned i = 0; i < 4; ++i)
        bytes[i] = uint8_t(value >> (8 * i));
    }
  }  // namespace

  std::string GetNavigationMetadataFilename(const std::string& prefix) {
    return prefix + "_navigation.meta";
  }

  void WriteNavigationMetadata(const std::string& prefix,
                               const std::string& codec) {
    if (codec.empty() || codec.size() > kMaxCodecLength)
      throw NavigationTypeError("Invalid navigation codec name length");
    std::array<uint8_t, 16> header{};
    std::copy(kMagic.begin(), kMagic.end(), header.begin());
    WriteU32(header.data() + 8, kVersion);
    WriteU32(header.data() + 12, uint32_t(codec.size()));
    const auto    path = GetNavigationMetadataFilename(prefix);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    file.write(codec.data(), codec.size());
    file.close();
    if (!file)
      FileError(path);
  }

  std::string ReadNavigationCodec(const NavigationMetadataContext& context) {
    const auto          path = GetNavigationMetadataFilename(context.prefix);
    std::optional<bool> present;
    if (context.exists) {
      present = context.exists(path);
    } else {
      std::error_code error;
      present = std::filesystem::exists(path, error);
      if (error)
        present.reset();
    }
    if (!present.has_value())
      FileError(path);
    if (!present.value())
      return "PQ";
    if (context.load_file && !context.load_file(path))
      FileError(path);

    std::ifstream file(path, std::ios::binary);
    if (!file)
      FileError(path);
    std::array<uint8_t, 16> header{};
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    if (file.bad())
      FileError(path);
    if (file.gcount() != std::streamsize(header.size()) ||
        !std::equal(kMagic.begin(), kMagic.end(), header.begin()))
      throw NavigationTypeError("Invalid navigation metadata header");
    if (ReadU32(header.data() + 8) != kVersion)
      throw NavigationTypeError("Unsupported navigation metadata version");
    const auto length = ReadU32(header.data() + 12);
    if (length == 0 || length > kMaxCodecLength)
      throw NavigationTypeError("Invalid navigation codec name length");
    std::string codec(length, '\0');
    file.read(codec.data(), length);
    if (file.bad())
      FileError(path);
    if (file.gcount() != std::streamsize(length))
      throw NavigationTypeError("Truncated navigation codec name");
    const auto trailing = file.peek();
    if (file.bad())
      FileError(path);
    if (trailing != std::char_traits<char>::eof())
      throw NavigationTypeError("Invalid navigation metadata size");
    return codec;
  }

  std::unique_ptr<NavigationStore> CreateNavigationStore(
      const std::string& codec) {
    if (codec == "PQ")
      return std::make_unique<PQNavigationStore>();
    throw NavigationTypeError("Unsupported DiskANN navigation codec: " + codec);
  }
}  // namespace diskann
