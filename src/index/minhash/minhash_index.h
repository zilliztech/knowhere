// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef MINHASH_TREE_H
#define MINHASH_TREE_H
#include "diskann/utils.h"
#include "faiss/impl/io.h"
#include "io/file_io.h"
#include "io/memory_io.h"
#include "knowhere/comp/bloomfilter.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "simd/hook.h"
namespace knowhere {
using Idx = int64_t;
using KeyType = uint64_t;
using ValueType = Idx;

struct MinHashIndexBuildParams {
    std::string data_path;
    std::string index_file_path;
    size_t band;
    size_t block_size;
};

struct MinHashIndexLoadParams {
    std::string index_file_path;
    bool enable_mmap = false;
    bool global_bloom_filter = false;
    float false_positive_prob = 0.01;
};
// struct MinHashIndexSearchParams {
//     size_t k;
//     size_t refine_factor;
// };

struct KVPair {
    KeyType Key;
    ValueType Value;
};
// index of each band
class MinHashBandIndex {
 public:
    MinHashBandIndex() {
    }
    static size_t
    FormatAndSave(faiss::BlockFileIOWriter& writer, const std::shared_ptr<KVPair[]> sorted_kv, const size_t block_size,
                  const size_t rows);

    Status
    Load(FileReader& reader, size_t rows, bool mmap_enable, BloomFilter<KeyType>& bloom_filter);

    ValueType
    Search(KeyType key);

    ~MinHashBandIndex() {
        if (mmap_enable_) {
            munmap(data_, block_size_ * block_num_);
        }
    }

 private:
    std::vector<KeyType> mins_;
    std::vector<KeyType> maxs_;
    std::vector<size_t> num_in_a_blk_;
    size_t block_size_;
    size_t block_num_;
    bool mmap_enable_ = false;
    char* data_ = nullptr;
    std::unique_ptr<char[]> owned_data_ = nullptr;
};

/* all index meta and codes will maintain as blocks in file*/
// todo: hold raw data for higher recall
class MinHashIndex {
 public:
    MinHashIndex(const size_t dim) : dim_(dim) {
    }
    MinHashIndex() = default;
    static Status
    BuildAndSave(MinHashIndexBuildParams* params);
    Status
    Load(MinHashIndexLoadParams* params);
    void
    Search(const float* query, float* distances, Idx* labels);
    size_t
    Count() {
        return ntotal_;
    };
    size_t
    GetDim() {
        return dim_;
    }
    size_t
    Size() {
        return ntotal_ * band_ * sizeof(KeyType);
    }

 private:
    std::unique_ptr<MinHashBandIndex[]> band_index_;
    bool is_loaded_ = false;
    size_t ntotal_;
    size_t dim_;
    size_t block_size_;
    size_t band_;
    // char* raw_data_ = nullptr;  // mmap mode, use IO object later
    std::vector<BloomFilter<KeyType>> bloom_;
};

// todo: thread pool version
namespace {
constexpr int MMAP_IO_FLAGS = MAP_POPULATE | MAP_SHARED;
inline KeyType
caculate_hash(const float* data, size_t dim, size_t band) {
    auto sub_dim = dim / band;
    return hash_vec(data + sub_dim * band, sub_dim);
}
inline int
find_hash_key(const KeyType* hash_arr, const size_t n, const uint64_t key) {
    if (n < 256) {
        auto result = std::lower_bound(hash_arr, hash_arr + n, key);
        if (result != hash_arr + n) {
            return result - hash_arr;
        } else {
            return -1;
        }
    } else {
        return faiss::binary_search(hash_arr, n, key);
    }
}

std::shared_ptr<KVPair[]>
gen_transposed_hash_kv(const float* data, size_t rows, size_t dim, size_t band) {
    auto res_kv = std::shared_ptr<KVPair[]>(new KVPair[band * rows]);
    auto sub_dim = dim / band;
    for (size_t i = 0; i < rows; i++) {
        const float* data_i = data + dim * i;
        for (size_t j = 0; j < band; j++) {
            KVPair kv = {hash_vec(data_i + j * sub_dim, sub_dim), i};
            res_kv.get()[j * rows + i] = kv;
        }
    }
    return res_kv;
}

void
sort_kv(const std::shared_ptr<KVPair[]> kv_code, size_t rows, size_t band) {
    for (size_t i = 0; i < band; i++) {
        std::sort(kv_code.get() + rows * i, kv_code.get() + rows * (i + 1),
                  [](const KVPair& a, const KVPair& b) { return a.Key < b.Key; });
    }
}
}  // namespace

size_t
MinHashBandIndex::FormatAndSave(faiss::BlockFileIOWriter& writer, const std::shared_ptr<KVPair[]> sorted_kv,
                                const size_t block_size, const size_t rows) {
    size_t max_num_of_a_block = block_size / sizeof(KVPair);
    size_t blocks_num = (rows + max_num_of_a_block - 1) / max_num_of_a_block;
    std::vector<KeyType> mins;
    std::vector<KeyType> maxs;
    std::vector<size_t> num_in_a_blk;
    mins.resize(blocks_num);
    maxs.resize(blocks_num);
    num_in_a_blk.resize(blocks_num);
    std::unique_ptr<KeyType[]> block_key_buf = std::make_unique<KeyType[]>(blocks_num);
    std::unique_ptr<ValueType[]> block_val_buf = std::make_unique<ValueType[]>(blocks_num);
    writer.flush();
    size_t data_pos = writer.get_current_block_id();
    for (size_t i = 0; i < blocks_num; i++) {
        writer.flush();
        auto beg = i * max_num_of_a_block;
        auto end = std::min((i + 1) * max_num_of_a_block, rows);
        num_in_a_blk[i] = end - beg;
        mins[i] = sorted_kv[beg].Key;
        maxs[i] = sorted_kv[end].Key;
        for (auto j = 0; j < num_in_a_blk[i]; j++) {
            block_key_buf[j] = sorted_kv[beg + j].Key;
            block_val_buf[j] = sorted_kv[beg + j].Value;
        }
        writer.write((const char*)block_key_buf.get(), num_in_a_blk[i] * sizeof(KeyType));
        writer.write((const char*)block_val_buf.get(), num_in_a_blk[i] * sizeof(ValueType));
    }
    writer.flush();
    auto index_meta_pos = writer.tellg();
    writeBinaryPOD(writer, blocks_num);
    writeBinaryPOD(writer, block_size);
    writeBinaryPOD(writer, data_pos);
    writer.write((const char*)mins.data(), mins.size() * sizeof(KeyType));
    writer.write((const char*)maxs.data(), maxs.size() * sizeof(KeyType));
    writer.write((const char*)num_in_a_blk.data(), num_in_a_blk.size() * sizeof(size_t));
    writer.flush();
    return index_meta_pos;
}

Status
MinHashBandIndex::Load(FileReader& reader, size_t rows, bool mmap_enable, BloomFilter<KeyType>& bloom_filter) {
    size_t data_pos;
    readBinaryPOD(reader, this->block_num_);
    readBinaryPOD(reader, this->block_size_);
    readBinaryPOD(reader, data_pos);
    mins_.resize(block_num_);
    maxs_.resize(block_num_);
    num_in_a_blk_.resize(block_num_);
    reader.read((char*)mins_.data(), mins_.size() * sizeof(KeyType));
    reader.read((char*)maxs_.data(), maxs_.size() * sizeof(KeyType));
    reader.read((char*)num_in_a_blk_.data(), num_in_a_blk_.size() * sizeof(size_t));
    if (mmap_enable) {
        mmap_enable_ = mmap_enable;
        data_ = static_cast<char*>(
            mmap(nullptr, block_size_ * block_num_, PROT_READ, MMAP_IO_FLAGS, reader.descriptor(), data_pos));
    } else {
        owned_data_ = std::make_unique<char[]>(block_size_ * block_num_);
        reader.seek(data_pos);
        reader.read(owned_data_.get(), block_size_ * block_num_);
        data_ = owned_data_.get();
    }
    for (auto i = 0; i < block_num_; i++) {
        KeyType* blk_i = reinterpret_cast<KeyType*>(data_ + block_size_ * i);
        for (auto j = 0; j < num_in_a_blk_[i]; j++) {
            bloom_filter.add(blk_i[j]);
        }
    }
}
ValueType
MinHashBandIndex::Search(KeyType key) {
    auto block_id = find_hash_key(mins_.data(), mins_.size(), key);
    if (block_id == -1 || key > maxs_[block_id]) {
        return -1;
    }
    auto rows = num_in_a_blk_[block_id];
    KeyType* blk_k = reinterpret_cast<KeyType*>(data_ + block_size_ * block_id);
    ValueType* blk_v = reinterpret_cast<ValueType*>(data_ + block_size_ * block_id + rows * sizeof(KeyType));
    auto inner_id = find_hash_key(blk_k, rows, key);
    if (inner_id == -1) {
        return -1;
    } else {
        return blk_v[inner_id];
    }
}

Status
MinHashIndex::BuildAndSave(MinHashIndexBuildParams* params) {
    if (params == nullptr) {
        LOG_KNOWHERE_ERROR_ << "build parameters is null.";
        return Status::invalid_args;
    }
    std::unique_ptr<float[]> data = nullptr;
    size_t ntotal, dim;
    diskann::load_bin(params->data_path, data, ntotal, dim);
    if (dim % params->band != 0) {
        LOG_KNOWHERE_ERROR_ << "dim % params.band != 0";
        return Status::invalid_args;
    }
    size_t band_index_n = params->band;
    size_t block_size = params->block_size;
    std::shared_ptr<KVPair[]> total_kv_pair = gen_transposed_hash_kv(data.get(), ntotal, dim, band_index_n);
    sort_kv(total_kv_pair, ntotal, band_index_n);

    faiss::BlockFileIOWriter writer(params->index_file_path.c_str(), params->block_size);
    // size_t data_pos = writer.tellg();
    // writer.flush_and_write(data, rows * dim * sizeof(float));

    std::vector<size_t> band_index_ofs(band_index_n);
    for (size_t index_i = 0; index_i < band_index_n; index_i++) {
        // std::cout <<"saving band i index"<<index_i<<std::endl;
        band_index_ofs[index_i] = MinHashBandIndex::FormatAndSave(writer, total_kv_pair, block_size, ntotal);
    }

    // write file header
    {
        MemoryIOWriter header_writer;
        writeBinaryPOD(header_writer, ntotal);
        writeBinaryPOD(header_writer, dim);
        writeBinaryPOD(header_writer, block_size);
        writeBinaryPOD(header_writer, band_index_n);
        // writeBinaryPOD(header_writer, data_pos);
        header_writer.write((char*)band_index_ofs.data(), band_index_ofs.size() * sizeof(size_t));
        writer.write_header((char*)header_writer.data_, header_writer.rp_);
        if (header_writer.data_) {
            delete[] header_writer.data_;
        }
    }
    return Status::success;
}

Status
MinHashIndex::Load(MinHashIndexLoadParams* params) {
    if (params == nullptr) {
        LOG_KNOWHERE_ERROR_ << "load parameters is null.";
        return Status::invalid_args;
    }
    auto reader = FileReader(params->index_file_path);
    readBinaryPOD(reader, ntotal_);
    readBinaryPOD(reader, dim_);
    readBinaryPOD(reader, block_size_);
    readBinaryPOD(reader, band_);
    //  size_t raw_data_pos;
    //  readBinaryPOD(reader, raw_data_pos);
    //  raw_data_ = data_ = static_cast<char*>(
    //    mmap(nullptr, dim_ * ntotal_ * sizeof(float), PROT_READ, MMAP_IO_FLAGS, reader.descriptor(), raw_data_pos));
    band_index_ = std::make_unique<MinHashBandIndex[]>(band_);
    std::vector<size_t> band_index_ofs(band_);
    reader.read((char*)band_index_ofs.data(), band_index_ofs.size() * sizeof(size_t));
    if (params->global_bloom_filter) {
        bloom_ = std::vector<BloomFilter<KeyType>>(1, BloomFilter<KeyType>(ntotal_, params->false_positive_prob));
    } else {
        bloom_ = std::vector<BloomFilter<KeyType>>(band_, BloomFilter<KeyType>(ntotal_, params->false_positive_prob));
    }
    for (auto i = 0; i < band_; i++) {
        reader.seek(band_index_ofs[i]);
        band_index_[i].Load(reader, ntotal_, params->enable_mmap, bloom_[i % bloom_.size()]);
    }
    is_loaded_ = true;
    return Status::success;
}

void
MinHashIndex::Search(const float* query, float* distances, Idx* labels) {
    *distances = 0;
    *labels = -1;
    for (auto i = 0; i < band_; i++) {
        const auto hash = caculate_hash(query, dim_, band_);
        auto& band = band_index_[i];
        auto& bloom = bloom_[i % bloom_.size()];
        if (bloom.contains(hash)) {
            auto id = band.Search(hash);
            if (id != -1) {
                *distances = 1;
                *labels = id;
                return;
            }
        }
    }
    return;
}
}  // namespace knowhere
#endif
