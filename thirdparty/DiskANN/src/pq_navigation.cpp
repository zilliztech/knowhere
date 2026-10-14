// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "diskann/pq_navigation.h"
#include "diskann/defaults.h"
#include "diskann/exceptions.h"
#include "diskann/logger.h"
#include "diskann/partition_and_pq.h"
#include "diskann/utils.h"

namespace diskann {
  static int aisaq_read_pq_header(const char* prefix, size_t& npts_u64,
                                  size_t& nchunks_u64) {
    std::ifstream freader;
    std::string   pq_compressed_vectors = get_pq_compressed_filename(prefix);
    std::string   pq_compressed_rearranged_vectors =
        get_pq_compressed_rearranged_filename(prefix);
    std::string pq_source_file;
    freader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    if (file_exists(pq_compressed_vectors)) {
      pq_source_file = pq_compressed_vectors;
    } else {
      if (file_exists(pq_compressed_rearranged_vectors)) {
        pq_source_file = pq_compressed_rearranged_vectors;
      } else {
        LOG_KNOWHERE_ERROR_ << "could not find any pq compressed file";
        return -1;
      }
    }
    try {
      freader.open(pq_source_file, std::ios::binary);
      uint32_t val;
      freader.read((char*) &val, sizeof(uint32_t));
      npts_u64 = (size_t) val;
      freader.read((char*) &val, sizeof(uint32_t));
      nchunks_u64 = (size_t) val;
      freader.close();
    } catch (std::system_error& e) {
      throw FileException(pq_source_file, e, __FUNCSIG__, __FILE__, __LINE__);
    }

    return 0;
  }

  PQNavigationStore::PQNavigationStore(bool resident_codes, bool rearranged,
                                       std::string pivots_path)
      : resident_codes_(resident_codes), rearranged_(rearranged),
        pivots_path_(std::move(pivots_path)) {
  }

  void PQNavigationStore::GetNecessaryFilenames(
      const std::string& prefix, std::vector<std::string>& filenames,
      NavigationFilePurpose purpose) const {
    const auto pivots = get_pq_pivots_filename(prefix);
    filenames.insert(filenames.end(),
                     {pivots, get_pq_rearrangement_perm_filename(pivots),
                      get_pq_chunk_offsets_filename(pivots),
                      get_pq_centroid_filename(pivots)});
    // AiSAQ's ordinary codes stay with its reader. Registration still includes
    // them; rearranged codes retain their original localization behavior.
    if (purpose == NavigationFilePurpose::Build || resident_codes_ ||
        rearranged_)
      filenames.push_back(rearranged_
                              ? get_pq_compressed_rearranged_filename(prefix)
                              : get_pq_compressed_filename(prefix));
  }

  int PQNavigationStore::Load(const std::string& prefix) {
    const auto pivots =
        pivots_path_.empty() ? get_pq_pivots_filename(prefix) : pivots_path_;
    size_t centroids, dimension;
    get_bin_metadata(pivots, centroids, dimension);
    if (centroids != 256) {
      LOG(ERROR) << "Number of PQ centroids is not 256. Exiting.";
      return -1;
    }
    dimension_ = dimension;
    size_t count, chunks;
    if (resident_codes_) {
      load_bin<uint8_t>(get_pq_compressed_filename(prefix), codes_, count,
                        chunks);
    } else if (aisaq_read_pq_header(prefix.c_str(), count, chunks) != 0) {
      LOG_KNOWHERE_ERROR_ << "failed reading pq header";
      return -1;
    }
    count_ = count;
    chunks_ = chunks;
    pq_table_.load_pq_centroid_bin(pivots.c_str(), chunks);
    if (!resident_codes_ && chunks_ > defaults::MAX_PQ_CHUNKS) {
      std::stringstream stream;
      stream << "Error: Loading index. Ensure that max PQ bytes for in-memory "
                "PQ data does not exceed "
             << defaults::MAX_PQ_CHUNKS;
      throw ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    LOG(INFO) << "Loaded PQ navigation. #points: " << count_
              << " #dim: " << dimension_ << " #chunks: " << chunks_;
    return 0;
  }

  size_t PQNavigationStore::MemorySize() const {
    size_t size = sizeof(*this);
    if (resident_codes_)
      size += count_ * chunks_ * sizeof(uint8_t);
    size += pq_table_.get_total_dims() * 256 * sizeof(float) * 2;
    size += pq_table_.get_total_dims() * (sizeof(uint32_t) + sizeof(float));
    size += (pq_table_.get_num_chunks() + 1) * sizeof(uint32_t);
    return size;
  }

  size_t PQNavigationStore::QueryMemorySize(uint64_t aligned_dimension) const {
    return sizeof(PQNavigationDistanceComputer) +
           ROUND_UP(
               defaults::MAX_GRAPH_DEGREE * aligned_dimension * sizeof(uint8_t),
               256) +
           ROUND_UP(256 * aligned_dimension * sizeof(float), 256);
  }

  std::unique_ptr<NavigationDistanceComputer>
  PQNavigationStore::CreateDistanceComputer(uint64_t aligned_dimension) const {
    return std::make_unique<PQNavigationDistanceComputer>(*this,
                                                          aligned_dimension);
  }

  PQNavigationDistanceComputer::PQNavigationDistanceComputer(
      const PQNavigationStore& model, uint64_t aligned_dimension)
      : model_(model) {
    alloc_aligned(
        reinterpret_cast<void**>(&gathered_codes_),
        defaults::MAX_GRAPH_DEGREE * aligned_dimension * sizeof(uint8_t), 256);
    try {
      alloc_aligned(reinterpret_cast<void**>(&lookup_table_),
                    256 * aligned_dimension * sizeof(float), 256);
    } catch (...) {
      aligned_free(gathered_codes_);
      throw;
    }
  }

  PQNavigationDistanceComputer::~PQNavigationDistanceComputer() {
    aligned_free(gathered_codes_);
    aligned_free(lookup_table_);
  }

  void PQNavigationDistanceComputer::set_query(const float* query) {
    model_.pq_table_.populate_chunk_distances(query, lookup_table_);
  }

  void PQNavigationDistanceComputer::compute_distances(const unsigned* ids,
                                                       uint64_t        count,
                                                       float* distances) {
    aggregate_coords(ids, count, model_.codes_.get(), model_.chunks_,
                     gathered_codes_);
    EvaluateCodes(gathered_codes_, count, distances);
  }

  void PQNavigationDistanceComputer::compute_distances(const unsigned* ids,
                                                       uint64_t        count,
                                                       float*        distances,
                                                       PQDataGetter& source) {
    source.aggregate_pq_coords(ids, count, model_.chunks_, gathered_codes_);
    EvaluateCodes(gathered_codes_, count, distances);
  }

  void PQNavigationDistanceComputer::EvaluateCodes(const uint8_t* codes,
                                                   uint64_t       count,
                                                   float*         distances) {
    pq_dist_lookup(codes, count, model_.chunks_, lookup_table_, distances);
  }
}  // namespace diskann
