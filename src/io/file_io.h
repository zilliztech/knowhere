// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once
#include <fcntl.h>
#include <unistd.h>

#include <stdexcept>
#include <string>

namespace knowhere {
struct FileReader {
    FileReader(const std::string& filename, bool auto_remove = false) {
        fd_ = open(filename.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open file");
        }

        size_ = lseek(fd_, 0, SEEK_END);
        lseek(fd_, 0, SEEK_SET);

        if (auto_remove) {
            unlink(filename.c_str());
        }
    }

    ~FileReader() {
        if (fd_ != -1) {
            close(fd_);
            fd_ = -1;
        }
    }

    int
    descriptor() const {
        return fd_;
    }

    size_t
    size() const {
        return size_;
    }

    ssize_t
    read(char* dst, size_t n) {
        return ::read(fd_, dst, n);
    }

    off_t
    seek(off_t offset) {
        return lseek(fd_, offset, SEEK_SET);
    }

    off_t
    advance(off_t offset) {
        return lseek(fd_, offset, SEEK_CUR);
    }

    off_t
    offset() {
        return lseek(fd_, 0, SEEK_CUR);
    }

 private:
    int fd_ = -1;
    size_t size_;
};
}  // namespace knowhere
