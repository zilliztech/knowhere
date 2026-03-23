#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <boost/core/span.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <new>
#include <vector>

#include "index/sparse/aligned_allocator.h"

namespace knowhere::sparse::inverted {

class BinaryContainer {
 public:
    virtual ~BinaryContainer() = default;
    virtual void
    append(const uint8_t* values, size_t count) = 0;
    virtual void
    write_at(size_t position, const uint8_t* values, size_t count) = 0;
    virtual void
    resize(size_t new_size) = 0;
    virtual uint8_t*
    data() = 0;
    virtual void
    seal() = 0;
    [[nodiscard]] virtual size_t
    size() const = 0;
};

class FileBinaryContainer : public BinaryContainer {
 public:
    explicit FileBinaryContainer(const std::filesystem::path& filepath) : filepath_(filepath) {
        std::filesystem::create_directories(filepath.parent_path());
        file_.open(filepath, std::ios::binary | std::ios::in | std::ios::out);
        if (!file_) {
            std::ofstream create_file(filepath, std::ios::binary);
            if (!create_file) {
                throw std::runtime_error("Failed to create file: " + filepath.string());
            }
            create_file.close();
            file_.open(filepath, std::ios::binary | std::ios::in | std::ios::out);
            if (!file_) {
                throw std::runtime_error("Failed to open file: " + filepath.string());
            }
        }
        file_.seekg(0, std::ios::end);
        size_ = file_.tellg();
        file_.seekg(0);
    }

    void
    append(const uint8_t* values, size_t count) override {
        if (sealed_) {
            throw std::runtime_error("FileBinaryContainer is sealed");
        }
        if (count == 0) {
            return;
        }
        file_.seekp(size_);
        file_.write(reinterpret_cast<const char*>(values), count);
        size_ += count;
    }

    void
    write_at(size_t position, const uint8_t* values, size_t count) override {
        if (count == 0) {
            return;
        }
        if (position + count > size_) {
            throw std::out_of_range("write_at out of range");
        }
        file_.seekp(position);
        file_.write(reinterpret_cast<const char*>(values), count);
    }

    void
    resize(size_t new_size) override {
        if (sealed_) {
            throw std::runtime_error("FileBinaryContainer is sealed");
        }
        file_.close();
        std::filesystem::resize_file(filepath_, new_size);
        file_.open(filepath_, std::ios::binary | std::ios::in | std::ios::out);
        size_ = new_size;
    }

    void
    seal() override {
        if (!sealed_) {
            file_.flush();
            file_.close();
            sealed_ = true;
        }
    }

    [[nodiscard]] size_t
    size() const override {
        return size_;
    }

    [[nodiscard]] uint8_t*
    data() override {
        if (!sealed_) {
            throw std::runtime_error("FileBinaryContainer is not sealed");
        }
        if (data_ != nullptr) {
            return static_cast<uint8_t*>(data_);
        }
        fd_ = ::open(filepath_.c_str(), O_RDWR);
        if (fd_ == -1) {
            throw std::runtime_error("Failed to open file: " + filepath_.string());
        }
        data_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap file");
        }
        return static_cast<uint8_t*>(data_);
    }

    ~FileBinaryContainer() override {
        if (!sealed_) {
            if (file_.is_open()) {
                file_.close();
            }
        } else {
            if (fd_ != -1) {
                ::munmap(data_, size_);
                ::close(fd_);
                data_ = nullptr;
            }
        }
        std::filesystem::remove(filepath_);
    }

 private:
    std::filesystem::path filepath_;
    std::fstream file_;
    bool sealed_{false};
    int fd_{-1};
    void* data_{nullptr};
    size_t size_{};
};

class MemBinaryContainer : public BinaryContainer {
    static constexpr size_t kAlignment = 128;
    static constexpr size_t kAlignmentMask = kAlignment - 1;

 public:
    explicit MemBinaryContainer() = default;

    void
    append(const uint8_t* values, size_t count) override {
        if (sealed_) {
            throw std::runtime_error("MemBinaryContainer is sealed");
        }
        if (bytes_count_ + count > data_.size()) {
            auto aligned_size = ((bytes_count_ + count + kAlignmentMask) & ~kAlignmentMask);
            data_.resize(aligned_size);
        }
        std::copy(values, values + count, data_.begin() + bytes_count_);
        bytes_count_ += count;
    }

    void
    write_at(size_t position, const uint8_t* values, size_t count) override {
        if (position + count > bytes_count_) {
            throw std::out_of_range("write_at out of range");
        }
        std::copy(values, values + count, data_.begin() + position);
    }

    void
    resize(size_t new_size) override {
        if (sealed_) {
            throw std::runtime_error("MemBinaryContainer is sealed");
        }
        auto aligned_size = ((new_size + kAlignmentMask) & ~kAlignmentMask);
        data_.resize(aligned_size);
        bytes_count_ = new_size;
    }

    [[nodiscard]] size_t
    size() const override {
        return bytes_count_;
    }

    void
    seal() override {
        sealed_ = true;
    }

    [[nodiscard]] uint8_t*
    data() override {
        return data_.data();
    }

 private:
    std::vector<uint8_t, aligned_allocator<uint8_t, kAlignment>> data_;
    size_t bytes_count_{0};
    bool sealed_{false};
};

class BlockMaxDataCursor {
 public:
    BlockMaxDataCursor(boost::span<uint32_t> block_max_ids, boost::span<float> block_max_scores)
        : block_max_ids_(block_max_ids), block_max_scores_(block_max_scores) {
        cur_vec_id_ = block_max_ids_[0];
    }

    void
    next_geq(uint32_t lower_bound) {
        while (cur_pos_ + 1 < block_max_ids_.size() && block_max_ids_[cur_pos_] < lower_bound) {
            cur_pos_++;
        }
        cur_vec_id_ = block_max_ids_[cur_pos_];
    }

    [[nodiscard]] float
    score() const {
        return block_max_scores_[cur_pos_];
    }

    [[nodiscard]] uint32_t
    vec_id() const {
        return cur_vec_id_;
    }

 private:
    uint32_t cur_vec_id_{0};
    uint32_t cur_pos_{0};
    boost::span<uint32_t> block_max_ids_;
    boost::span<float> block_max_scores_;
};

struct BlockMaxData {
    std::unique_ptr<BinaryContainer> container_;
    boost::span<uint32_t> block_max_ids_;
    boost::span<float> block_max_scores_;
    boost::span<size_t> block_offsets_;
    size_t block_size_;
};

}  // namespace knowhere::sparse::inverted
