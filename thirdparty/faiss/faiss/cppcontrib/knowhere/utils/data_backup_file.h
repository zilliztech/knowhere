#pragma once

#include <shared_mutex>
#include <memory>
#include <cstdio>
#include <fstream>

#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/**
 * @brief BackDataFileHandler is a temporary file structure, and the temperary file only exist in the life cycle of the struct. 
 *  It is used to back up the original data of the index for the needs of data reconstruction with no loss.
 */
class DataBackFileHandler {
public:
    DataBackFileHandler(std::string prefix, size_t block_size);
    ~DataBackFileHandler();
    void ReadDataBlock(char* data, size_t block_id);
    void AppendDataBlock(const char* data);
    inline bool FileExist();

private:
    std::shared_mutex file_mtx_;
    size_t buffer_size_;
    size_t buffer_res_size_;
    std::unique_ptr<char[]> buffer_;
    std::string raw_data_file_name_;
    size_t block_size_;
    size_t file_block_num_;
    size_t buffer_block_num_;
    size_t buffer_max_block_num_;
};

}
}
} // namespace  faiss