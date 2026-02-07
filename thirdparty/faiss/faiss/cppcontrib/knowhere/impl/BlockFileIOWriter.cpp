#include <faiss/cppcontrib/knowhere/impl/BlockFileIOWriter.h>

#include <cstdio>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>



namespace faiss::cppcontrib::knowhere {

namespace {

size_t round_up(const size_t X, const size_t Y) {
    return ((((X) / (Y)) + ((X) % (Y) != 0)) * (Y));
}

}

BlockFileIOWriter::BlockFileIOWriter(
        FILE* wf,
        size_t block_size,
        size_t header_size)
        : FileIOWriter(wf), block_size(block_size) {
    header_size = round_up(header_size, block_size);
    block_buf = std::make_unique<char[]>(header_size);
    block_buf_ofs = 0;
    // write a placeholder for file header
    fwrite(block_buf.get(), sizeof(char), header_size, f);
    current_block_id = 1;
}

BlockFileIOWriter::BlockFileIOWriter(
        const char* fname,
        size_t block_size,
        size_t header_size)
        : FileIOWriter(fname), block_size(block_size) {
    header_size = round_up(header_size, block_size);
    block_buf = std::make_unique<char[]>(header_size);
    block_buf_ofs = 0;
    // write a placeholder for file header
    fwrite(block_buf.get(), sizeof(char), header_size, f);
    current_block_id = 1;
}

BlockFileIOWriter::~BlockFileIOWriter() {
    flush();
}

size_t BlockFileIOWriter::write(const char* ptr, size_t bytes_size) {
    if (block_buf_ofs + bytes_size <= block_size) {
        memcpy(block_buf.get() + block_buf_ofs, ptr, bytes_size);
        block_buf_ofs += bytes_size;
        if (block_buf_ofs == block_size) {
            flush();
        }
    } else {
        size_t cur_pos = 0;
        while (cur_pos < bytes_size) {
            size_t copy_size =
                    std::min(block_size - block_buf_ofs, bytes_size - cur_pos);
            memcpy(block_buf.get() + block_buf_ofs, ptr + cur_pos, copy_size);
            cur_pos += copy_size;
            block_buf_ofs += copy_size;
            if (block_buf_ofs == block_size) {
                flush();
            }
        }
    }
    return bytes_size;
}

void BlockFileIOWriter::flush() {
    if (block_buf_ofs != 0) {
        fwrite(block_buf.get(), sizeof(char), block_size, f);
        current_block_id++;
        block_buf_ofs = 0;
    }
}

size_t BlockFileIOWriter::flush_and_write(const char* ptr, size_t bytes) {
    flush();
    return write(ptr, bytes);
}

size_t BlockFileIOWriter::operator()(
        const void* ptr,
        size_t size,
        size_t nitems) {
    return write((const char*)ptr, size * nitems);
}

size_t BlockFileIOWriter::write_header(const char* ptr, size_t bytes) {
    FAISS_THROW_IF_MSG(
            bytes > block_size,
            "header size should not larger than a block size");
    fseek(f, 0, SEEK_SET);
    fwrite(ptr, sizeof(char), bytes, f);
    fseek(f, 0, SEEK_END);
    return bytes;
}

}


