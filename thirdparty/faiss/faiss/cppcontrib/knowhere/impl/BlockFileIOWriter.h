#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <faiss/impl/io.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// not thread safe, keeping a header in the file
struct BlockFileIOWriter : FileIOWriter {
    size_t block_size;
    std::unique_ptr<char[]> block_buf = nullptr;
    size_t current_block_id = 0;
    size_t block_buf_ofs = 0;

    BlockFileIOWriter(
            FILE* wf,
            size_t block_size = 8 * 1024,
            size_t header_size = 8 * 1024);

    BlockFileIOWriter(
            const char* fname,
            size_t block_size = 8 * 1024,
            size_t header_size = 8 * 1024);

    ~BlockFileIOWriter() override;

    size_t operator()(const void* ptr, size_t size, size_t nitems) override;

    size_t write(const char* ptr, size_t bytes);
    // go back to the head
    size_t write_header(const char* ptr, size_t bytes);

    void flush();

    size_t tellg() {
        return current_block_id * block_size + block_buf_ofs;
    }

    size_t flush_and_write(const char* ptr, size_t bytes);

    size_t get_current_block_id() {
        return current_block_id;
    }
};

}
}
}

