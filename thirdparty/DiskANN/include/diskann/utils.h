// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <fcntl.h>
#include <algorithm>
#include <errno.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <future>
#include <fstream>
#include <iostream>
#include <limits.h>

#include <string>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string.h>
#if !defined(__ARM_NEON) || !defined(__aarch64__)
#include <xmmintrin.h>
#endif
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#include <unistd.h>
typedef int FileHandle;

#include "logger.h"
#include "cached_io.h"
#include "ann_exception.h"
#include "common_includes.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t) (X) % (uint64_t) (Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)
#define IS_4096_ALIGNED(X) IS_ALIGNED(X, 4096)
#define METADATA_SIZE \
  4096  // all metadata of individual sub-component files is written in first
        // 4KB for unified files

#define BUFFER_SIZE_FOR_CACHED_IO (_u64) 1024 * (_u64) 1048576
#define COMPLETION_PERCENT 10

inline bool file_exists(const std::string& name, bool dirCheck = false) {
  int         val;
  struct stat buffer;
  val = stat(name.c_str(), &buffer);

  if (val != 0) {
    switch (errno) {
      case EINVAL:
        diskann::cout << "Invalid argument passed to stat()" << std::endl;
        break;
      case ENOENT:
        // file is not existing, not an issue, so we won't cout anything.
        break;
      default:
        diskann::cout << "Unexpected error in stat():" << errno << std::endl;
        break;
    }
    return false;
  } else {
    // the file entry exists. If reqd, check if this is a directory.
    return dirCheck ? buffer.st_mode & S_IFDIR : true;
  }
}

typedef uint64_t _u64;
typedef int64_t  _s64;
typedef uint32_t _u32;
typedef int32_t  _s32;
typedef uint16_t _u16;
typedef int16_t  _s16;
typedef uint8_t  _u8;
typedef int8_t   _s8;
inline void      open_file_to_write(std::ofstream&     writer,
                                    const std::string& filename) {
       writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
       if (!file_exists(filename))
    writer.open(filename, std::ios::binary | std::ios::out);
  else
    writer.open(filename, std::ios::binary | std::ios::in | std::ios::out);

  if (writer.fail()) {
         char buff[1024];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    strerror_r(errno, buff, 1024);
#pragma GCC diagnostic pop
    diskann::cerr << std::string("Failed to open file") + filename +
                         " for write because " + buff
                  << std::endl;
    throw diskann::ANNException(std::string("Failed to open file ") + filename +
                                    " for write because: " + buff,
                                -1);
  }
}

inline _u64 get_file_size(const std::string& fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    _u64 end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    diskann::cerr << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

inline int delete_file(const std::string& fileName) {
  if (file_exists(fileName)) {
    auto rc = ::remove(fileName.c_str());
    if (rc != 0) {
      diskann::cerr
          << "Could not delete file: " << fileName
          << " even though it exists. This might indicate a permissions issue. "
             "If you see this message, please contact the diskann team."
          << std::endl;
    }
    return rc;
  } else {
    return 0;
  }
}

namespace diskann {
  static const size_t MAX_SIZE_OF_STREAMBUF = 2LL * 1024 * 1024 * 1024;

  enum Metric {
    L2 = 0,
    INNER_PRODUCT = 1,
    COSINE = 2,
  };

  inline void alloc_aligned(void** ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
    *ptr = ::aligned_alloc(align, size);
    assert(*ptr != nullptr);
  }

  inline void realloc_aligned(void** ptr, size_t size, size_t align) {
    assert(IS_ALIGNED(size, align));
    diskann::cerr << "No aligned realloc on GCC. Must malloc and mem_align, "
                     "left it out for now."
                  << std::endl;
    assert(*ptr != nullptr);
  }

  inline void check_stop(std::string arnd) {
    int brnd;
    diskann::cout << arnd << std::endl;
    std::cin >> brnd;
  }

  inline void aligned_free(void* ptr) {
    // Gopal. Must have a check here if the pointer was actually allocated by
    // _alloc_aligned
    if (ptr == nullptr) {
      return;
    }
    free(ptr);
    // Note: malloc_trim() combines with jemalloc/tcmalloc may cause crash
    // if (malloc_trim(0) == 0) {
    //   LOG_KNOWHERE_DEBUG_ << "Failed to release free memory from the heap.";
    // }
  }

  inline void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size,
                        unsigned N) {
    for (unsigned i = 0; i < size; ++i) {
      addr[i] = rng() % (N - size);
    }

    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
      if (addr[i] <= addr[i - 1]) {
        addr[i] = addr[i - 1] + 1;
      }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
      addr[i] = (addr[i] + off) % N;
    }
  }

  // get_bin_metadata functions START
  inline void get_bin_metadata_impl(std::basic_istream<char>& reader,
                                    size_t& nrows, size_t& ncols) {
    int nrows_32, ncols_32;
    reader.read((char*) &nrows_32, sizeof(int));
    reader.read((char*) &ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
  }

  inline void get_bin_metadata(const std::string& bin_file, size_t& nrows,
                               size_t& ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    get_bin_metadata_impl(reader, nrows, ncols);
  }
  // get_bin_metadata functions END

  template<typename T>
  inline std::string getValues(T* data, size_t num) {
    std::stringstream stream;
    stream << "[";
    for (size_t i = 0; i < num; i++) {
      stream << std::to_string(data[i]) << ",";
    }
    stream << "]" << std::endl;

    return stream.str();
  }

  // load_bin functions START
  template<typename T>
  inline void load_bin_impl(std::basic_istream<char>& reader,
                            size_t actual_file_size, std::unique_ptr<T[]>& data,
                            size_t& npts, size_t& dim) {
    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    LOG_KNOWHERE_DEBUG_ << "Metadata: #pts = " << npts << ", #dims = " << dim
                        << "...";

    size_t expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch. Actual size is " << actual_file_size
             << " while expected size is  " << expected_actual_file_size
             << " npts = " << npts << " dim = " << dim
             << " size of <T>= " << sizeof(T) << std::endl;
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    data = std::make_unique<T[]>(npts * dim);
    reader.read(reinterpret_cast<char*>(data.get()), npts * dim * sizeof(T));
  }

  inline void wait_for_keystroke() {
    int a;
    std::cout << "Press any number to continue.." << std::endl;
    std::cin >> a;
  }

  template<typename T>
  inline void load_bin(const std::string& bin_file, std::unique_ptr<T[]>& data, size_t& npts,
                       size_t& dim) {
    // OLS
    //_u64            read_blk_size = 64 * 1024 * 1024;
    // cached_ifstream reader(bin_file, read_blk_size);
    // size_t actual_file_size = reader.get_file_size();
    // END OLS

    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
      LOG_KNOWHERE_DEBUG_ << "Opening bin file " << bin_file.c_str() << "... ";
      reader.open(bin_file, std::ios::binary | std::ios::ate);
      uint64_t fsize = reader.tellg();
      reader.seekg(0);
      load_bin_impl<T>(reader, fsize, data, npts, dim);
    } catch (std::system_error& e) {
      throw FileException(bin_file, e, __FUNCSIG__, __FILE__, __LINE__);
    }
    LOG_KNOWHERE_DEBUG_ << "done.";
  }
  // load_bin functions END

  inline void load_truthset(const std::string&           bin_file,
                            std::unique_ptr<uint32_t[]>& ids,
                            std::unique_ptr<float[]>& dists, size_t& npts,
                            size_t& dim) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    LOG_KNOWHERE_DEBUG_ << "Reading truthset file " << bin_file.c_str()
                        << " ...";
    size_t actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    LOG_KNOWHERE_DEBUG_ << "Metadata: #pts = " << npts << ", #dims = " << dim
                        << "... ";

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    size_t expected_file_size_with_dists =
        2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    size_t expected_file_size_just_ids =
        npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
      truthset_type = 2;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size
             << ", expected: " << expected_file_size_with_dists << " or "
             << expected_file_size_just_ids;
      diskann::cout << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    ids = std::make_unique<uint32_t[]>(npts * dim);
    reader.read(reinterpret_cast<char*>(ids.get()), npts * dim * sizeof(uint32_t));

    if (truthset_type == 1) {
      dists = std::make_unique<float[]>(npts * dim);
      reader.read(reinterpret_cast<char*>(dists.get()), npts * dim * sizeof(float));
    }
  }

  inline void prune_truthset_for_range(
      const std::string& bin_file, float range,
      std::vector<std::vector<_u32>>& groundtruth, size_t& npts) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    LOG_KNOWHERE_DEBUG_ << "Reading truthset file " << bin_file.c_str()
                        << "... ";
    size_t actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    _u64   dim = (unsigned) dim_i32;

    LOG_KNOWHERE_DEBUG_ << "Metadata: #pts = " << npts << ", #dims = " << dim
                        << "... ";

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    size_t expected_file_size_with_dists =
        2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size
             << ", expected: " << expected_file_size_with_dists;
      diskann::cout << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    auto ids = std::make_unique<_u32[]>(npts * dim);
    reader.read(reinterpret_cast<char*>(ids.get()), npts * dim * sizeof(uint32_t));
    std::unique_ptr<float[]> dists = nullptr;

    if (truthset_type == 1) {
      dists = std::make_unique<float[]>(npts * dim);
      reader.read(reinterpret_cast<char*>(dists.get()), npts * dim * sizeof(float));
    }
    float min_dist = std::numeric_limits<float>::max();
    float max_dist = 0;
    groundtruth.resize(npts);
    for (_u32 i = 0; i < npts; i++) {
      groundtruth[i].clear();
      for (_u32 j = 0; j < dim; j++) {
        if (dists[i * dim + j] <= range) {
          groundtruth[i].emplace_back(ids[i * dim + j]);
        }
        min_dist =
            min_dist > dists[i * dim + j] ? dists[i * dim + j] : min_dist;
        max_dist =
            max_dist < dists[i * dim + j] ? dists[i * dim + j] : max_dist;
      }
      // std::cout<<groundtruth[i].size() << " " ;
    }
    std::cout << "Min dist: " << min_dist << ", Max dist: " << max_dist
              << std::endl;
  }

  inline void load_range_truthset(const std::string&              bin_file,
                                  std::vector<std::vector<_u32>>& groundtruth,
                                  _u64&                           gt_num) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    LOG_KNOWHERE_DEBUG_ << "Reading truthset file " << bin_file.c_str()
                        << "... ";
    size_t actual_file_size = reader.get_file_size();

    int npts_u32, total_u32;
    reader.read((char*) &npts_u32, sizeof(int));
    reader.read((char*) &total_u32, sizeof(int));

    gt_num = (_u64) npts_u32;
    _u64 total_res = (_u64) total_u32;

    LOG_KNOWHERE_DEBUG_ << "Metadata: #pts = " << gt_num
                        << ", #total_results = " << total_res << "...";

    size_t expected_file_size =
        2 * sizeof(_u32) + gt_num * sizeof(_u32) + total_res * sizeof(_u32);

    if (actual_file_size != expected_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch in range truthset. actual size: "
             << actual_file_size << ", expected: " << expected_file_size;
      diskann::cout << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    groundtruth.clear();
    groundtruth.resize(gt_num);
    std::vector<_u32> gt_count(gt_num);

    reader.read((char*) gt_count.data(), sizeof(_u32) * gt_num);

    std::vector<_u32> gt_stats(gt_count);
    std::sort(gt_stats.begin(), gt_stats.end());

    std::cout << "GT count percentiles:" << std::endl;
    for (_u32 p = 0; p < 100; p += 5)
      std::cout
          << "percentile " << p << ": "
          << gt_stats[static_cast<size_t>(std::floor((p / 100.0) * gt_num))]
          << std::endl;
    std::cout << "percentile 100"
              << ": " << gt_stats[gt_num - 1] << std::endl;

    for (_u32 i = 0; i < gt_num; i++) {
      groundtruth[i].clear();
      groundtruth[i].resize(gt_count[i]);
      if (gt_count[i] != 0)
        reader.read((char*) groundtruth[i].data(), sizeof(_u32) * gt_count[i]);
    }
  }

  template<typename T>
  inline uint64_t save_bin(const std::string& filename, T* data, size_t npts,
                           size_t ndims) {
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    LOG_KNOWHERE_DEBUG_ << "Writing bin: " << filename.c_str();
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    writer.write((char*) &npts_i32, sizeof(int));
    writer.write((char*) &ndims_i32, sizeof(int));
    LOG_KNOWHERE_DEBUG_ << "bin: #pts = " << npts << ", #dims = " << ndims
                        << ", size = "
                        << npts * ndims * sizeof(T) + 2 * sizeof(int) << "B";

    //    data = new T[npts_u64 * ndims_u64];
    writer.write((char*) data, npts * ndims * sizeof(T));
    writer.close();
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    LOG_KNOWHERE_DEBUG_ << "Finished writing bin.";
    return bytes_written;
  }

  // load_aligned_bin functions START

  template<typename T>
  inline void load_aligned_bin_impl(std::basic_istream<char>& reader,
                                    size_t actual_file_size, T*& data,
                                    size_t& npts, size_t& dim,
                                    size_t& rounded_dim) {
    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    size_t expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch. Actual size is " << actual_file_size
             << " while expected size is  " << expected_actual_file_size
             << " npts = " << npts << " dim = " << dim
             << " size of <T>= " << sizeof(T) << std::endl;
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    rounded_dim = ROUND_UP(dim, 8);
    std::stringstream stream;
    stream << "Metadata: #pts = " << npts << ", #dims = " << dim
           << ", aligned_dim = " << rounded_dim << "... ";
    size_t allocSize = npts * rounded_dim * sizeof(T);
    stream << "allocating aligned memory of " << allocSize << " bytes... ";
    alloc_aligned(((void**) &data), allocSize, 8 * sizeof(T));
    stream << "done. Copying data to mem_aligned buffer...";

    for (size_t i = 0; i < npts; i++) {
      reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
      memset((void*) (data + i * rounded_dim + dim), 0,
             (rounded_dim - dim) * sizeof(T));
    }
    stream << " done." << std::endl;
    LOG_KNOWHERE_DEBUG_ << stream.str();
  }

  template<typename T>
  inline void load_aligned_bin(const std::string& bin_file, T*& data,
                               size_t& npts, size_t& dim, size_t& rounded_dim) {
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
      LOG_KNOWHERE_DEBUG_ << "Reading (with alignment) bin file " << bin_file
                          << " ...";
      reader.open(bin_file, std::ios::binary | std::ios::ate);

      uint64_t fsize = reader.tellg();
      reader.seekg(0);
      load_aligned_bin_impl(reader, fsize, data, npts, dim, rounded_dim);
    } catch (std::system_error& e) {
      throw FileException(bin_file, e, __FUNCSIG__, __FILE__, __LINE__);
    }
  }

  template<typename InType, typename OutType>
  void convert_types(const InType* srcmat, OutType* destmat, size_t npts,
                     size_t dim) {
    for (int64_t i = 0; i < (_s64) npts; i++) {
      for (uint64_t j = 0; j < dim; j++) {
        destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
      }
    }
  }

  // this function will take in_file of n*d dimensions and save the output as a
  // floating point matrix
  // with n*(d+1) dimensions. All vectors are scaled by a large value M so that
  // the norms are <=1 and the final coordinate is set so that the resulting
  // norm (in d+1 coordinates) is equal to 1 this is a classical transformation
  // from MIPS to L2 search from "On Symmetric and Asymmetric LSHs for Inner
  // Product Search" by Neyshabur and Srebro

  template<typename T>
  float prepare_base_for_inner_products(const std::string in_file,
                                        const std::string out_file) {
    static_assert(
        knowhere::KnowhereFloatTypeCheck<T>::value,
        "prepare_base_for_inner_products only support fp16, bf16, fp32.");
    LOG_KNOWHERE_DEBUG_
        << "Pre-processing base file by adding extra coordinate";
    std::ifstream in_reader(in_file.c_str(), std::ios::binary);
    std::ofstream out_writer(out_file.c_str(), std::ios::binary);
    _u64          npts, in_dims, out_dims;
    float         max_norm = 0;

    _u32 npts32, dims32;
    in_reader.read((char*) &npts32, sizeof(uint32_t));
    in_reader.read((char*) &dims32, sizeof(uint32_t));

    npts = npts32;
    in_dims = dims32;
    out_dims = in_dims + 1;
    _u32 outdims32 = (_u32) out_dims;

    out_writer.write((char*) &npts32, sizeof(uint32_t));
    out_writer.write((char*) &outdims32, sizeof(uint32_t));

    size_t               BLOCK_SIZE = 100000;
    size_t               block_size = npts <= BLOCK_SIZE ? npts : BLOCK_SIZE;
    std::unique_ptr<T[]> in_block_data =
        std::make_unique<T[]>(block_size * in_dims);
    std::unique_ptr<T[]> out_block_data =
        std::make_unique<T[]>(block_size * out_dims);

    std::memset((void*) out_block_data.get(), 0,
                sizeof(T) * block_size * out_dims);
    _u64 num_blocks = DIV_ROUND_UP(npts, block_size);

    std::vector<float> norms(npts, 0);

    for (_u64 b = 0; b < num_blocks; b++) {
      _u64 start_id = b * block_size;
      _u64 end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
      _u64 block_pts = end_id - start_id;
      in_reader.read((char*) in_block_data.get(),
                     block_pts * in_dims * sizeof(T));
      for (_u64 p = 0; p < block_pts; p++) {
        for (_u64 j = 0; j < in_dims; j++) {
          norms[start_id + p] +=
              in_block_data[p * in_dims + j] * in_block_data[p * in_dims + j];
        }
        max_norm =
            max_norm > norms[start_id + p] ? max_norm : norms[start_id + p];
      }
    }

    max_norm = std::sqrt(max_norm);

    in_reader.seekg(2 * sizeof(_u32), std::ios::beg);
    for (_u64 b = 0; b < num_blocks; b++) {
      _u64 start_id = b * block_size;
      _u64 end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
      _u64 block_pts = end_id - start_id;
      in_reader.read((char*) in_block_data.get(),
                     block_pts * in_dims * sizeof(T));
      for (_u64 p = 0; p < block_pts; p++) {
        for (_u64 j = 0; j < in_dims; j++) {
          out_block_data[p * out_dims + j] =
              (T) (((float) in_block_data[p * in_dims + j]) / max_norm);
        }
        float res = 1 - (norms[start_id + p] / (max_norm * max_norm));
        res = res <= 0 ? 0 : std::sqrt(res);
        out_block_data[p * out_dims + out_dims - 1] = (T) res;
      }
      out_writer.write((char*) out_block_data.get(),
                       block_pts * out_dims * sizeof(T));
    }
    out_writer.close();
    return max_norm;
  }

  template<typename T>
  std::vector<float> prepare_base_for_cosine(const std::string in_file,
                                             const std::string out_file) {
    static_assert(knowhere::KnowhereFloatTypeCheck<T>::value,
                  "prepare_base_for_cosine only support fp16, bf16, fp32.");
    LOG_KNOWHERE_DEBUG_ << "Pre-processing base file by normalizing";
    std::ifstream in_reader(in_file.c_str(), std::ios::binary);
    std::ofstream out_writer(out_file.c_str(), std::ios::binary);
    _u64          npts, in_dims, out_dims;

    _u32 npts32, dims32;
    in_reader.read((char*) &npts32, sizeof(uint32_t));
    in_reader.read((char*) &dims32, sizeof(uint32_t));

    npts = npts32;
    in_dims = dims32;
    out_dims = in_dims;
    _u32 outdims32 = (_u32) out_dims;

    out_writer.write((char*) &npts32, sizeof(uint32_t));
    out_writer.write((char*) &outdims32, sizeof(uint32_t));

    size_t               BLOCK_SIZE = 100000;
    size_t               block_size = npts <= BLOCK_SIZE ? npts : BLOCK_SIZE;
    std::unique_ptr<T[]> in_block_data =
        std::make_unique<T[]>(block_size * in_dims);
    std::unique_ptr<T[]> out_block_data =
        std::make_unique<T[]>(block_size * out_dims);

    std::memset((void*) out_block_data.get(), 0,
                sizeof(T) * block_size * out_dims);
    _u64 num_blocks = DIV_ROUND_UP(npts, block_size);

    std::vector<float> norms(npts, 0);

    for (_u64 b = 0; b < num_blocks; b++) {
      _u64 start_id = b * block_size;
      _u64 end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
      _u64 block_pts = end_id - start_id;
      in_reader.read((char*) in_block_data.get(),
                     block_pts * in_dims * sizeof(T));
      for (_u64 p = 0; p < block_pts; p++) {
        for (_u64 j = 0; j < in_dims; j++) {
          norms[start_id + p] +=
              in_block_data[p * in_dims + j] * in_block_data[p * in_dims + j];
        }
      }
    }

    for (auto& norm : norms) {
      norm = std::sqrt(norm);
    }

    in_reader.seekg(2 * sizeof(_u32), std::ios::beg);
    for (_u64 b = 0; b < num_blocks; b++) {
      _u64 start_id = b * block_size;
      _u64 end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
      _u64 block_pts = end_id - start_id;
      in_reader.read((char*) in_block_data.get(),
                     block_pts * in_dims * sizeof(T));
      for (_u64 p = 0; p < block_pts; p++) {
        for (_u64 j = 0; j < in_dims; j++) {
          out_block_data[p * out_dims + j] =
              (T) (((float) in_block_data[p * in_dims + j]) /
                   norms[start_id + p]);
        }
      }
      out_writer.write((char*) out_block_data.get(),
                       block_pts * out_dims * sizeof(T));
    }
    out_writer.close();

    return norms;
  }

  // plain saves data as npts X ndims array into filename
  template<typename T>
  void save_Tvecs(const char* filename, T* data, size_t npts, size_t ndims) {
    std::string fname(filename);

    // create cached ofstream with 64MB cache
    cached_ofstream writer(fname, 64 * 1048576);

    unsigned dims_u32 = (unsigned) ndims;

    // start writing
    for (uint64_t i = 0; i < npts; i++) {
      // write dims in u32
      writer.write((char*) &dims_u32, sizeof(unsigned));

      // get cur point in data
      T* cur_pt = data + i * ndims;
      writer.write((char*) cur_pt, ndims * sizeof(T));
    }
  }
  template<typename T>
  inline uint64_t save_data_in_base_dimensions(const std::string& filename,
                                               T* data, size_t npts,
                                               size_t ndims, size_t aligned_dim,
                                               size_t offset = 0) {
    std::ofstream writer;  //(filename, std::ios::binary | std::ios::out);
    open_file_to_write(writer, filename);
    int  npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    _u64 bytes_written = 2 * sizeof(uint32_t) + npts * ndims * sizeof(T);
    writer.seekp(offset, writer.beg);
    writer.write((char*) &npts_i32, sizeof(int));
    writer.write((char*) &ndims_i32, sizeof(int));
    for (size_t i = 0; i < npts; i++) {
      writer.write((char*) (data + i * aligned_dim), ndims * sizeof(T));
    }
    writer.close();
    return bytes_written;
  }

  template<typename T>
  inline void copy_aligned_data_from_file(const std::string bin_file, T*& data,
                                          size_t& npts, size_t& dim,
                                          const size_t& rounded_dim,
                                          size_t        offset = 0) {
    if (data == nullptr) {
      diskann::cerr << "Memory was not allocated for " << data
                    << " before calling the load function. Exiting..."
                    << std::endl;
      throw diskann::ANNException(
          "Null pointer passed to copy_aligned_data_from_file function", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    }
    std::ifstream reader(bin_file, std::ios::binary);
    reader.seekg(offset, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    /*
    size_t expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch. Actual size is " << actual_file_size
             << " while expected size is  " << expected_actual_file_size
             << " npts = " << npts << " dim = " << dim
             << " size of <T>= " << sizeof(T) << std::endl;
      diskann::cout << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    */

    for (size_t i = 0; i < npts; i++) {
      reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
      memset((void*) (data + i * rounded_dim + dim), 0,
             (rounded_dim - dim) * sizeof(T));
    }
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector(const char* vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64) {
#if defined(__ARM_NEON) && defined(__aarch64__)
      __builtin_prefetch((const char*) vec + d, 0, 3);
#else
      _mm_prefetch((const char*) vec + d, _MM_HINT_T0);
#endif
    }
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector_l2(const char* vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
#if defined(__ARM_NEON) && defined(__aarch64__)
      __builtin_prefetch((const char*) vec + d, 0, 2);
#else
      _mm_prefetch((const char*) vec + d, _MM_HINT_T1);
#endif
  }

  // NOTE: Implementation in utils.cpp.
  void block_convert(std::ofstream& writr, std::ifstream& readr,
                     float* read_buf, _u64 npts, _u64 ndims);

  void normalize_data_file(const std::string& inFileName,
                           const std::string& outFileName);

  inline std::string get_pq_pivots_filename(const std::string& prefix) {
    return prefix + "_pq_pivots.bin";
  }

  inline std::string get_pq_rearrangement_perm_filename(
      const std::string& pq_pivots_filename) {
    return pq_pivots_filename + "_rearrangement_perm.bin";
  }

  inline std::string get_pq_chunk_offsets_filename(
      const std::string& pq_pivots_filename) {
    return pq_pivots_filename + "_chunk_offsets.bin";
  }

  inline std::string get_pq_centroid_filename(
      const std::string& pq_pivots_filename) {
    return pq_pivots_filename + "_centroid.bin";
  }

  inline std::string get_pq_compressed_filename(const std::string& prefix) {
    return prefix + "_pq_compressed.bin";
  }

  inline std::string get_disk_index_filename(const std::string& prefix) {
    return prefix + "_disk.index";
  }

  inline std::string get_disk_index_medoids_filename(
      const std::string& disk_index_filename) {
    return disk_index_filename + "_medoids.bin";
  }

  inline std::string get_disk_index_centroids_filename(
      const std::string& disk_index_filename) {
    return disk_index_filename + "_centroids.bin";
  }

  inline std::string get_sample_data_filename(const std::string& prefix) {
    return prefix + "_sample_data.bin";
  }

  inline std::string get_disk_index_max_base_norm_file(
      const std::string& disk_index_filename) {
    return disk_index_filename + "_max_base_norm.bin";
  }

  inline std::string get_cached_nodes_file(
      const std::string& disk_index_filename) {
    return disk_index_filename + "_cached_nodes.bin";
  }
};  // namespace diskann

struct PivotContainer {
  PivotContainer() = default;

  PivotContainer(size_t pivo_id, float pivo_dist)
      : piv_id{pivo_id}, piv_dist{pivo_dist} {
  }

  bool operator<(const PivotContainer& p) const {
    return p.piv_dist < piv_dist;
  }

  bool operator>(const PivotContainer& p) const {
    return p.piv_dist > piv_dist;
  }

  size_t piv_id;
  float  piv_dist;
};

/*
inline bool file_exists(const std::string& name) {
  struct stat buffer;
  auto        val = stat(name.c_str(), &buffer);
  diskann::cout << " Stat(" << name.c_str() << ") returned: " << val
                << std::endl;
  return (val == 0);
}
*/

inline bool validate_index_file_size(std::ifstream& in) {
  if (!in.is_open())
    throw diskann::ANNException(
        "Index file size check called on unopened file stream", -1, __FUNCSIG__,
        __FILE__, __LINE__);
  in.seekg(0, in.end);
  size_t actual_file_size = in.tellg();
  in.seekg(0, in.beg);
  size_t expected_file_size;
  in.read((char*) &expected_file_size, sizeof(uint64_t));
  in.seekg(0, in.beg);
  if (actual_file_size != expected_file_size) {
    diskann::cerr << "Index file size error. Expected size (metadata): "
                  << expected_file_size
                  << ", actual file size : " << actual_file_size << "."
                  << std::endl;
    return false;
  }
  return true;
}

// This function is valid only for float data type.
template<typename T>
inline void normalize(T* arr, size_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; i++) {
    sum += arr[i] * arr[i];
  }
  sum = sqrt(sum);
  for (uint32_t i = 0; i < dim; i++) {
    arr[i] = (T) (arr[i] / sum);
  }
}

// need to check and change this
inline bool avx2Supported() {
  return true;
}
inline void printProcessMemory(const char*) {
}

extern bool AvxSupportedCPU;
extern bool Avx2SupportedCPU;
