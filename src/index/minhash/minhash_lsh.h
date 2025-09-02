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

#ifndef MINHASH_LSH_H
#define MINHASH_LSH_H
#include "faiss/impl/io.h"
#include "index/minhash/minhash_util.h"
#include "io/file_io.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/bloomfilter.h"
#include "knowhere/comp/task.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "simd/hook.h"
#include "sys/stat.h"
namespace knowhere::minhash {
struct MinHashLSHBuildParams {
    std::string data_path;
    std::string index_file_path;
    size_t band = 1;
    size_t block_size = 8192;
    bool with_raw_data = false;
    size_t mh_vec_element_size = 8;
    size_t mh_vec_length = 0;
};

struct MinHashLSHLoadParams {
    std::string index_file_path;
    bool hash_code_in_memory = false;
    bool global_bloom_filter = false;
    float false_positive_prob = 0.01;
};

struct MinHashLSHSearchParams {
    size_t k = 1;
    size_t refine_k = 1;
    bool search_with_jaccard = false;
    faiss::IDSelector* id_selector = nullptr;
};

// index of each band
class MinHashBandIndex {
 public:
    static size_t
    FormatAndSave(faiss::BlockFileIOWriter& writer, const KVPair* sorted_kv, const size_t block_size,
                  const size_t rows);

    Status
    Load(FileReader& reader, size_t rows, char* mmap_data, BloomFilter<KeyType>& bloom_filter);

    void
    Search(KeyType key, MinHashLSHResultHandler* res, faiss::IDSelector* id_selector) const;

    Status
    WarmUp() const {
        if (mmap_enable_) {
            if (madvise(data_, block_size_ * blocks_num_, MADV_WILLNEED) == -1) {
                LOG_KNOWHERE_WARNING_ << "Failed to warmup band data : " << strerror(errno);
                return Status::disk_file_error;
            }
        }
        return Status::success;
    }

    Status
    CoolDown() const {
        if (mmap_enable_) {
            if (madvise(data_, block_size_ * blocks_num_, MADV_DONTNEED) == -1) {
                LOG_KNOWHERE_WARNING_ << "Failed to cooldown band data : " << strerror(errno);
                return Status::disk_file_error;
            }
        }
        return Status::success;
    }

 private:
    std::vector<KeyType> mins_;
    std::vector<KeyType> maxs_;
    std::vector<size_t> num_in_a_blk_;
    size_t block_size_ = 8192;
    size_t blocks_num_ = 0;
    bool mmap_enable_ = false;
    char* data_ = nullptr;
    std::unique_ptr<char[]> owned_data_ = nullptr;
};

/* all index meta and codes will maintain as blocks in file*/
class MinHashLSH {
 public:
    MinHashLSH(){};
    static Status
    BuildAndSave(MinHashLSHBuildParams* params);
    Status
    Load(MinHashLSHLoadParams* params);
    Status
    Search(const char* query, float* distances, idx_t* labels, MinHashLSHSearchParams* params) const;
    Status
    BatchSearch(const char* query, size_t nq, float* distances, idx_t* labels, std::shared_ptr<ThreadPool> pool,
                MinHashLSHSearchParams* params) const;
    Status
    GetDataByIds(const idx_t* ids, size_t n, char* data) const;
    bool
    HasRawData() const {
        return this->with_raw_data_;
    }
    size_t
    Size() const {
        return this->ntotal_ * band_ * sizeof(KeyType);
    }
    size_t
    Count() const {
        return ntotal_;
    };
    size_t
    GetVectorSize() const {
        return this->mh_vec_length_ * this->mh_vec_elememt_size_;
    }

    ~MinHashLSH() {
        if (mmap_data_) {
            munmap(mmap_data_, file_size_);
        }
    }

 private:
    std::unique_ptr<MinHashBandIndex[]> band_index_;
    bool is_loaded_ = false;
    size_t block_size_ = 0;
    size_t band_ = 1;
    char* mmap_data_ = nullptr;
    size_t file_size_ = 0;
    bool with_raw_data_ = false;
    char* raw_data_ = nullptr;  // mmap mode, use IO object later
    std::vector<BloomFilter<KeyType>> bloom_;
    size_t mh_vec_elememt_size_ = 0;
    size_t mh_vec_length_ = 0;
    size_t ntotal_ = 0;
};

size_t
MinHashBandIndex::FormatAndSave(faiss::BlockFileIOWriter& writer, const KVPair* sorted_kv, const size_t block_size,
                                const size_t rows) {
    size_t max_num_of_a_block = block_size / sizeof(KVPair);
    size_t blocks_num = (rows + max_num_of_a_block - 1) / max_num_of_a_block;
    std::vector<KeyType> mins;
    std::vector<KeyType> maxs;
    std::vector<size_t> num_in_a_blk;
    mins.resize(blocks_num);
    maxs.resize(blocks_num);
    num_in_a_blk.resize(blocks_num);
    std::unique_ptr<KeyType[]> block_key_buf = std::make_unique<KeyType[]>(max_num_of_a_block);
    std::unique_ptr<ValueType[]> block_val_buf = std::make_unique<ValueType[]>(max_num_of_a_block);
    writer.flush();
    size_t data_pos = writer.tellg();
    for (size_t i = 0; i < blocks_num; i++) {
        writer.flush();
        auto beg = i * max_num_of_a_block;
        auto end = std::min((i + 1) * max_num_of_a_block, rows);
        num_in_a_blk[i] = end - beg;
        mins[i] = sorted_kv[beg].Key;
        maxs[i] = sorted_kv[end - 1].Key;
        for (size_t j = 0; j < num_in_a_blk[i]; j++) {
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
MinHashBandIndex::Load(FileReader& reader, size_t rows, char* mmap_data, BloomFilter<KeyType>& bloom_filter) {
    size_t data_pos;
    readBinaryPOD(reader, this->blocks_num_);
    readBinaryPOD(reader, this->block_size_);
    readBinaryPOD(reader, data_pos);
    this->mins_.resize(blocks_num_);
    this->maxs_.resize(blocks_num_);
    num_in_a_blk_.resize(blocks_num_);
    reader.read((char*)mins_.data(), mins_.size() * sizeof(KeyType));
    reader.read((char*)maxs_.data(), maxs_.size() * sizeof(KeyType));
    reader.read((char*)num_in_a_blk_.data(), num_in_a_blk_.size() * sizeof(size_t));
    if (mmap_data) {
        data_ = mmap_data + data_pos;
        mmap_enable_ = true;
    } else {
        owned_data_ = std::make_unique<char[]>(block_size_ * blocks_num_);
        reader.seek(data_pos);
        reader.read(owned_data_.get(), block_size_ * blocks_num_);
        data_ = owned_data_.get();
        mmap_enable_ = false;
    }

    auto build_pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    for (size_t i = 0; i < blocks_num_; i++) {
        futures.emplace_back(build_pool->push([&, idx = i]() {
            KeyType* blk_i = reinterpret_cast<KeyType*>(data_ + block_size_ * idx);
            for (size_t j = 0; j < num_in_a_blk_[idx]; j++) {
                bloom_filter.add(blk_i[j]);
            }
        }));
    }
    WaitAllSuccess(futures);
    return Status::success;
}

void
MinHashBandIndex::Search(KeyType key, MinHashLSHResultHandler* res, faiss::IDSelector* id_selector) const {
    auto block_id = faiss::u64_binary_search_ge(maxs_.data(), maxs_.size(), key);

    if (block_id == -1 || key < mins_[block_id]) {
        return;
    }
    while ((size_t)block_id < mins_.size() && key >= mins_[block_id]) {
        size_t rows = num_in_a_blk_[block_id];
        KeyType* blk_k = reinterpret_cast<KeyType*>(data_ + block_size_ * block_id);
        ValueType* blk_v = reinterpret_cast<ValueType*>(data_ + block_size_ * block_id + rows * sizeof(KeyType));
        int inner_id = faiss::u64_binary_search_eq(blk_k, rows, key);
        if (inner_id != -1) {
            for (; key == blk_k[inner_id] && (size_t)inner_id < rows; inner_id++) {
                if (id_selector == nullptr || id_selector->is_member(blk_v[inner_id])) {
                    res->push(blk_v[inner_id], 1.0);
                }
                if (res->full())
                    break;
            }
        }
        block_id++;
    }
    return;
}

Status
MinHashLSH::BuildAndSave(MinHashLSHBuildParams* params) {
    if (params == nullptr) {
        LOG_KNOWHERE_ERROR_ << "build parameters is null.";
        return Status::invalid_args;
    }
    size_t band_index_n = params->band;
    size_t block_size = params->block_size;
    size_t mh_vec_element_size = params->mh_vec_element_size;
    size_t mh_vec_length = params->mh_vec_length;
    size_t data_size = mh_vec_element_size * mh_vec_length;
    size_t ntotal, bin_vec_dim;
    int64_t data_pos = -1;
    if (params->with_raw_data) {
        block_size = ROUND_UP(mh_vec_element_size * mh_vec_length, block_size);
    }
    std::shared_ptr<KVPair[]> total_kv_pair;

    size_t header_size = DIV_ROUND_UP(sizeof(MinHashLSH) + band_index_n * sizeof(size_t), block_size);
    faiss::BlockFileIOWriter writer(params->index_file_path.c_str(), block_size, header_size);
    // load raw data, generate hash kv for each band and save raw data
    {
        std::unique_ptr<char[]> data = nullptr;
        // raw data save like binary vector format
        load_vec_data<bin1>(params->data_path, data, ntotal, bin_vec_dim);
        if (bin_vec_dim != mh_vec_element_size * mh_vec_length * 8) {
            LOG_KNOWHERE_ERROR_ << "fail to load binary file, dim in file(" << bin_vec_dim
                                << ") not equal to mh_vec_element_size * mh_vec_length * 8:"
                                << params->mh_vec_element_size * params->mh_vec_length * 8;
            return Status::disk_file_error;
        }
        if (mh_vec_length % band_index_n != 0) {
            LOG_KNOWHERE_ERROR_ << "params->mh_vec_length % params.band != 0";
            return Status::invalid_args;
        }

        total_kv_pair = GenTransposedHashKV(data.get(), ntotal, data_size, band_index_n);
        if (params->with_raw_data) {
            data_pos = writer.tellg();
            // todo: @cqy123456 format raw data if use disk index
            auto vec_num_a_blk = block_size / data_size;

            for (size_t i = 0; i < ntotal; i += vec_num_a_blk) {
                auto num = std::min(ntotal - i, vec_num_a_blk);
                writer.flush_and_write((char*)(data.get() + i * data_size), num * data_size);
            }
        }
    }

    // save hash kv as MinHashBandIndex format
    std::vector<size_t> band_index_ofs(band_index_n);
    {
        SortHashKV(total_kv_pair, ntotal, band_index_n);

        for (size_t index_i = 0; index_i < band_index_n; index_i++) {
            band_index_ofs[index_i] =
                MinHashBandIndex::FormatAndSave(writer, total_kv_pair.get() + index_i * ntotal, block_size, ntotal);
        }
    }

    // write file header
    {
        MemoryIOWriter header_writer;
        writeBinaryPOD(header_writer, ntotal);
        writeBinaryPOD(header_writer, mh_vec_length);
        writeBinaryPOD(header_writer, mh_vec_element_size);
        writeBinaryPOD(header_writer, block_size);
        writeBinaryPOD(header_writer, band_index_n);
        writeBinaryPOD(header_writer, data_pos);
        header_writer.write((char*)band_index_ofs.data(), band_index_ofs.size() * sizeof(size_t));
        writer.write_header((char*)header_writer.data_, header_writer.rp_);
        if (header_writer.data_) {
            delete[] header_writer.data_;
        }
    }
    return Status::success;
}

Status
MinHashLSH::Load(MinHashLSHLoadParams* params) {
    if (params == nullptr) {
        LOG_KNOWHERE_ERROR_ << "load parameters is null.";
        return Status::invalid_args;
    }
    auto reader = FileReader(params->index_file_path);
    readBinaryPOD(reader, this->ntotal_);
    readBinaryPOD(reader, this->mh_vec_length_);
    readBinaryPOD(reader, this->mh_vec_elememt_size_);
    readBinaryPOD(reader, this->block_size_);
    readBinaryPOD(reader, this->band_);

    int64_t data_pos;
    readBinaryPOD(reader, data_pos);
    if (data_pos == -1) {
        this->with_raw_data_ = false;
    } else {
        this->with_raw_data_ = true;
    }

    if (!params->hash_code_in_memory || this->with_raw_data_) {
        auto f = std::unique_ptr<FILE, decltype(&fclose)>(fopen(params->index_file_path.c_str(), "r"), &fclose);
        if (!f) {
            LOG_KNOWHERE_ERROR_ << "Failed to open file: " << params->index_file_path << " Error: " << strerror(errno);
            return Status::disk_file_error;
        }
        struct stat s;
        if (fstat(fileno(f.get()), &s) != 0) {
            LOG_KNOWHERE_ERROR_ << "Failed to stat file: " << strerror(errno);
            return Status::disk_file_error;
        }

        this->file_size_ = s.st_size;
        if (this->file_size_ == 0) {
            LOG_KNOWHERE_ERROR_ << "empty index file";
            return Status::disk_file_error;
        }
        this->mmap_data_ = static_cast<char*>(mmap(NULL, file_size_, PROT_READ, MAP_SHARED, fileno(f.get()), 0));
        if (mmap_data_ == MAP_FAILED) {
            mmap_data_ = nullptr;
            LOG_KNOWHERE_ERROR_ << "fail to mmap data ." << errno << " " << strerror(errno);
            return Status::disk_file_error;
        }

    } else {
        this->mmap_data_ = nullptr;
    }
    if (this->with_raw_data_) {
        raw_data_ = mmap_data_ + data_pos;
    } else {
        raw_data_ = nullptr;
    }
    band_index_ = std::make_unique<MinHashBandIndex[]>(band_);
    std::vector<size_t> band_index_ofs(band_);
    reader.read((char*)band_index_ofs.data(), band_index_ofs.size() * sizeof(size_t));
    size_t bloom_filter_num = params->global_bloom_filter ? 1 : this->band_;
    bloom_.reserve(bloom_filter_num);
    for (size_t i = 0; i < bloom_filter_num; i++) {
        bloom_.emplace_back(this->ntotal_, params->false_positive_prob);
    }
    auto band_mmap_addr = params->hash_code_in_memory ? nullptr : this->mmap_data_;
    for (size_t i = 0; i < band_; i++) {
        reader.seek(band_index_ofs[i]);
        band_index_[i].Load(reader, this->ntotal_, band_mmap_addr, bloom_[i % bloom_.size()]);
    }
    is_loaded_ = true;
    return Status::success;
}

Status
MinHashLSH::Search(const char* query, float* distances, idx_t* labels, MinHashLSHSearchParams* params) const {
    if (params == nullptr) {
        LOG_KNOWHERE_ERROR_ << "search parameters is null.";
        return Status::invalid_args;
    }
    auto search_with_jaccard = params->search_with_jaccard;
    if (search_with_jaccard && !this->with_raw_data_) {
        LOG_KNOWHERE_ERROR_ << "fail to search with jaccard distance without raw data.";
        return Status::invalid_args;
    }
    auto topk = params->k;
    auto id_selector = params->id_selector;

    std::shared_ptr<idx_t[]> reorder_ids = nullptr;
    std::shared_ptr<float[]> reorder_dis = nullptr;
    std::shared_ptr<MinHashLSHResultHandler> res = nullptr;
    if (search_with_jaccard) {
        auto refine_k = std::max(params->refine_k, topk);
        reorder_ids = std::shared_ptr<idx_t[]>(new idx_t[refine_k]);
        reorder_dis = std::shared_ptr<float[]>(new float[refine_k]);
        res = std::shared_ptr<MinHashLSHResultHandler>(
            new MinHashLSHResultHandler(reorder_ids.get(), reorder_dis.get(), refine_k));
    } else {
        res = std::shared_ptr<MinHashLSHResultHandler>(new MinHashLSHResultHandler(labels, distances, topk));
    }
    for (size_t i = 0; i < band_; i++) {
        const auto hash = GetHashKey(query, this->mh_vec_elememt_size_ * this->mh_vec_length_, band_, i);
        auto& band = band_index_[i];
        auto& bloom = bloom_[i % bloom_.size()];
        if (bloom.contains(hash)) {
            band.Search(hash, res.get(), id_selector);
        }
        if (res->full())
            break;
    }
    if (search_with_jaccard) {
        MinHashJaccardKNNSearchByIDs(query, this->raw_data_, reorder_ids.get(), this->mh_vec_length_,
                                     this->mh_vec_elememt_size_, res->topk_, topk, distances, labels);
    }
    return Status::success;
}
Status
MinHashLSH::BatchSearch(const char* query, size_t nq, float* distances, idx_t* labels, std::shared_ptr<ThreadPool> pool,
                        MinHashLSHSearchParams* params) const {
    if (params == nullptr) {
        LOG_KNOWHERE_ERROR_ << "search parameters is null.";
        return Status::invalid_args;
    }
    auto search_with_jaccard = params->search_with_jaccard;
    if (search_with_jaccard && !this->with_raw_data_) {
        LOG_KNOWHERE_ERROR_ << "fail to search with jaccard distance without raw data.";
        return Status::invalid_args;
    }
    auto topk = params->k;
    auto id_selector = params->id_selector;
    // init
    std::shared_ptr<idx_t[]> reorder_ids_list = nullptr;
    std::shared_ptr<float[]> reorder_dis_list = nullptr;
    std::vector<MinHashLSHResultHandler> all_res;
    if (search_with_jaccard) {
        auto refine_k = std::max(params->refine_k, topk);
        reorder_ids_list = std::shared_ptr<idx_t[]>(new idx_t[nq * refine_k]);
        reorder_dis_list = std::shared_ptr<float[]>(new float[nq * refine_k]);
        all_res.reserve(nq);
        for (size_t i = 0; i < nq; i++) {
            all_res.emplace_back(reorder_ids_list.get() + i * refine_k, reorder_dis_list.get() + i * refine_k,
                                 refine_k);
        }
    } else {
        all_res.reserve(nq);
        for (size_t i = 0; i < nq; i++) {
            all_res.emplace_back(labels + i * topk, distances + i * topk, topk);
        }
    }
    // prepare query key
    std::vector<minhash::KeyType> query_hash_v;
    query_hash_v.reserve(nq * band_);
    {
        auto query_kv = minhash::GenTransposedHashKV((const char*)query, nq,
                                                     this->mh_vec_elememt_size_ * this->mh_vec_length_, this->band_);
        for (auto i = 0; i < nq * band_; i++) {
            query_hash_v.emplace_back(query_kv[i].Key);
        }
    }
    if (query_hash_v.size() != nq * band_) {
        return Status::internal_error;
    }
    // search bands
    std::vector<folly::Future<folly::Unit>> futures;
    std::vector<size_t> access_list(nq);
    for (size_t i = 0; i < nq; i++) {
        access_list[i] = i;
    }
    size_t band_ofs = 0;
    while (access_list.size() && band_ofs < this->band_) {
        size_t band_beg = band_ofs;
        size_t band_end = std::min(band_beg + kQueryBandBatch, band_);
        for (size_t i = band_beg; i < band_end; i++) {
            band_index_[i].WarmUp();
        }
        size_t access_num = access_list.size();
        size_t run_times = (access_num + kQueryBatch - 1) / kQueryBatch;
        futures.reserve(run_times);
        // avoid lots of page miss in mmap mode, all thread only access band from band_beg to band_end
        for (size_t row = 0; row < run_times; ++row) {
            futures.emplace_back(pool->push([&, query_id_beg = row * kQueryBatch,
                                             query_id_end = std::min((size_t)((row + 1) * kQueryBatch), access_num)]() {
                for (auto i = band_beg; i < band_end; i++) {
                    auto& band = band_index_[i];
                    auto& bloom = bloom_[i % bloom_.size()];
                    const minhash::KeyType* band_i_q_hash = query_hash_v.data() + nq * i;
                    for (auto j = query_id_beg; j < query_id_end; j++) {
                        auto index = access_list[j];
                        const auto hash = band_i_q_hash[index];
                        auto& res = all_res[index];
                        if (res.full()) {
                            continue;
                        }
                        if (bloom.contains(hash)) {
                            band.Search(hash, &res, id_selector);
                        }
                    }
                }
            }));
        }
        WaitAllSuccess(futures);
        futures.clear();
        std::vector<size_t> new_access_list;
        for (auto q_i : access_list) {
            if (!all_res[q_i].full()) {
                new_access_list.emplace_back(q_i);
            }
        }
        band_ofs += kQueryBandBatch;
        access_list = new_access_list;
    }
    // reorder by jaccard distance
    if (search_with_jaccard) {
        futures.reserve(nq);
        for (size_t i = 0; i < nq; i++) {
            futures.emplace_back(pool->push([&, id = i]() {
                const char* q = query + i * mh_vec_elememt_size_ * mh_vec_length_;
                auto reorder_ids = all_res[i].ids_list_;
                auto refine_k = all_res[i].topk_;
                auto res_ids = labels + i * topk;
                auto res_dis = distances + i * topk;
                MinHashJaccardKNNSearchByIDs(q, this->raw_data_, reorder_ids, this->mh_vec_length_,
                                             this->mh_vec_elememt_size_, refine_k, topk, res_dis, res_ids);
                return;
            }));
        }
        WaitAllSuccess(futures);
    }
    return Status::success;
}

Status
MinHashLSH::GetDataByIds(const idx_t* ids, size_t n, char* data) const {
    if (this->with_raw_data_) {
        auto mh_vec_size = this->mh_vec_elememt_size_ * this->mh_vec_length_;
        for (size_t i = 0; i < n; i++) {
            char* des = data + i * mh_vec_size;
            char* src = this->raw_data_ + ids[i] * mh_vec_size;
            std::memcpy(des, src, mh_vec_size);
        }
        return Status::success;
    } else {
        return Status::not_implemented;
    }
}
}  // namespace knowhere::minhash
#endif
