#include <cstring>
#include <cmath>
#include <mutex>

#include <faiss/cppcontrib/knowhere/utils/data_backup_file.h>



namespace faiss::cppcontrib::knowhere {

namespace {
constexpr const char* kRawDataFileName = "/ivf_sq_cc_raw_data.bin";
constexpr const size_t kDataBackFileHandlerDefaulBufferSize = 8192;
constexpr const size_t kDataBackFileHandlerMinFlushNum = 4;
} // namespace

DataBackFileHandler::DataBackFileHandler(
        std::string prefix,
        size_t block_size) {
   FAISS_THROW_IF_NOT(block_size != 0);
    std::unique_lock<std::shared_mutex> lock(file_mtx_);
    raw_data_file_name_ = prefix + kRawDataFileName;
    std::fstream file;
    file.open(raw_data_file_name_.c_str(), std::fstream::out | std::fstream::trunc);
    FAISS_THROW_IF_NOT(file.is_open());
    file.close();
    this->block_size_ = block_size;
    this->file_block_num_ = 0;
    this->buffer_size_ = std::min(
            kDataBackFileHandlerDefaulBufferSize,
            kDataBackFileHandlerMinFlushNum * block_size);
    this->buffer_ = std::make_unique<char[]>(this->buffer_size_);
    this->buffer_res_size_ = this->buffer_size_;
    this->buffer_block_num_ = 0;
    this->buffer_max_block_num_ =
            std::floor((float)buffer_size_ / (float)block_size);
    memset(this->buffer_.get(), 0, this->buffer_size_);
}

DataBackFileHandler::~DataBackFileHandler() {
    if (FileExist()) {
        std::remove(raw_data_file_name_.c_str());
    }
}

bool DataBackFileHandler::FileExist() {
    std::ifstream file(raw_data_file_name_.c_str());
    return file.good();
}

void DataBackFileHandler::ReadDataBlock(
        char* data,
        size_t blk_id) {
    std::shared_lock<std::shared_mutex> lock(file_mtx_);
    FAISS_THROW_IF_NOT(blk_id < this->buffer_block_num_ + this->file_block_num_);
    if (blk_id >= this->file_block_num_) {
        auto buffer_blk_id = blk_id - this->file_block_num_;
        std::memcpy(
                data, buffer_.get() + buffer_blk_id * block_size_, block_size_);
    } else {
        std::ifstream reader(raw_data_file_name_.c_str(), std::ios::binary);
        reader.seekg(blk_id * block_size_);
        reader.read(data, block_size_);
    }
}

void DataBackFileHandler::AppendDataBlock(
        const char* data) {
    std::unique_lock<std::shared_mutex> lock(file_mtx_);
    std::memcpy(buffer_.get() + buffer_block_num_ * block_size_, data, block_size_);
    buffer_block_num_++;
    if (buffer_block_num_ == buffer_max_block_num_) {
        std::ofstream writer(raw_data_file_name_.c_str(), std::ios::app);
        writer.write(buffer_.get(), buffer_max_block_num_ * block_size_);
        writer.flush();
        buffer_block_num_ = 0;
        file_block_num_ += buffer_max_block_num_;
    }
}

}


