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

#ifndef MMAP_H
#define MMAP_H

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace knowhere {

// File-backed writable memory used as transient backing storage for derived
// arrays. The file is removed with the region; durability still belongs to the
// caller's normal index/metadata serialization.
class MmapRegion {
 public:
    static std::shared_ptr<MmapRegion>
    Create(const std::string& filepath, size_t byte_size) {
        if (filepath.empty()) {
            throw std::runtime_error("mmap region filepath is empty");
        }
        if (byte_size == 0) {
            throw std::runtime_error("mmap region size is zero");
        }
        const auto parent_path = std::filesystem::path(filepath).parent_path();
        if (!parent_path.empty()) {
            std::filesystem::create_directories(parent_path);
        }

        const int fd = ::open(filepath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) {
            throw std::runtime_error(std::string("failed to create mmap region file: ") + std::strerror(errno));
        }

        const auto close_fd = [&] { ::close(fd); };
        if (::ftruncate(fd, static_cast<off_t>(byte_size)) != 0) {
            const auto err = errno;
            close_fd();
            std::error_code ec;
            std::filesystem::remove(filepath, ec);
            throw std::runtime_error(std::string("failed to resize mmap region file: ") + std::strerror(err));
        }

        void* data = ::mmap(nullptr, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const auto mmap_errno = errno;
        close_fd();
        if (data == MAP_FAILED) {
            std::error_code ec;
            std::filesystem::remove(filepath, ec);
            throw std::runtime_error(std::string("failed to mmap region file: ") + std::strerror(mmap_errno));
        }
        return std::shared_ptr<MmapRegion>(new MmapRegion(filepath, byte_size, data));
    }

    ~MmapRegion() {
        if (data_ != nullptr) {
            ::munmap(data_, byte_size_);
        }
        if (!filepath_.empty()) {
            std::error_code ec;
            std::filesystem::remove(filepath_, ec);
        }
    }

    void*
    data() const {
        return data_;
    }

 private:
    MmapRegion(std::string filepath, size_t byte_size, void* data)
        : filepath_(std::move(filepath)), byte_size_(byte_size), data_(data) {
    }

    std::string filepath_;
    size_t byte_size_;
    void* data_;
};

// Generates distinct backing file names for repeated derived-array builds under
// the same IdMap object.
class MmapFilePathGenerator {
 public:
    MmapFilePathGenerator() = default;

    explicit MmapFilePathGenerator(std::string prefix) : prefix_(std::move(prefix)) {
    }

    bool
    empty() const {
        return prefix_.empty();
    }

    std::string
    Next(const void* owner) {
        if (empty()) {
            return {};
        }
        return prefix_ + "." + std::to_string(reinterpret_cast<uintptr_t>(owner)) + "." + std::to_string(++generation_);
    }

 private:
    std::string prefix_;
    size_t generation_ = 0;
};

}  // namespace knowhere

#endif /* MMAP_H */
