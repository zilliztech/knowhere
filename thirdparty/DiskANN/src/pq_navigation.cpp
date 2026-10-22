// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "diskann/pq_navigation.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "diskann/defaults.h"
#include "diskann/parameters.h"
#include "diskann/pq_table.h"
#include "diskann/utils.h"

namespace diskann {
  namespace {
    // Keep the kernel independent of the model method's register allocation.
    // Inlining this loop can spill the code indices inside the hot chunk loop.
    [[gnu::noinline]] void LookupPQCodes(const uint8_t* codes, uint64_t count,
                                         uint64_t chunks, const float* lookup,
                                         float* distances) {
      if (count == 0)
        return;
#if defined(__SSE2__)
      // A small chunk window retains partial sums in registers while keeping
      // its lookup tables local. Each lane still adds chunks in the native
      // order; there is no horizontal reduction, reassociation, or changed
      // precision.
      constexpr uint64_t kChunkWindow = 16;
      std::memset(distances, 0, count * sizeof(float));
      uint64_t chunk = 0;
      for (; chunk + kChunkWindow <= chunks; chunk += kChunkWindow) {
        uint64_t i = 0;
        for (; i + 4 <= count; i += 4) {
          auto        sums = _mm_loadu_ps(distances + i);
          const auto* block = codes + i * chunks + chunk;
          for (uint64_t offset = 0; offset < kChunkWindow; ++offset) {
            const auto* table = lookup + 256 * (chunk + offset);
            const auto  values =
                _mm_setr_ps(table[block[offset]], table[block[chunks + offset]],
                            table[block[2 * chunks + offset]],
                            table[block[3 * chunks + offset]]);
            sums = _mm_add_ps(sums, values);
          }
          _mm_storeu_ps(distances + i, sums);
        }
        if (i + 2 <= count) {
          auto sums = _mm_loadl_pi(
              _mm_setzero_ps(), reinterpret_cast<const __m64*>(distances + i));
          const auto* block = codes + i * chunks + chunk;
          for (uint64_t offset = 0; offset < kChunkWindow; ++offset) {
            const auto* table = lookup + 256 * (chunk + offset);
            const auto  values = _mm_setr_ps(
                table[block[offset]], table[block[chunks + offset]], 0, 0);
            sums = _mm_add_ps(sums, values);
          }
          _mm_storel_pi(reinterpret_cast<__m64*>(distances + i), sums);
          i += 2;
        }
        for (; i < count; ++i) {
          float sum = distances[i];
          for (uint64_t offset = 0; offset < kChunkWindow; ++offset)
            sum += lookup[256 * (chunk + offset) +
                          codes[i * chunks + chunk + offset]];
          distances[i] = sum;
        }
      }
      for (; chunk < chunks; ++chunk)
        for (uint64_t i = 0; i < count; ++i)
          distances[i] += lookup[256 * chunk + codes[i * chunks + chunk]];
#else
      pq_dist_lookup(codes, count, chunks, lookup, distances);
#endif
    }
  }  // namespace

  struct PQNavigationQueryTable {
    float* lookup = nullptr;
    explicit PQNavigationQueryTable(uint64_t chunks) {
      alloc_aligned(reinterpret_cast<void**>(&lookup),
                    ROUND_UP(256 * chunks * sizeof(float), 256), 256);
    }
    ~PQNavigationQueryTable() {
      aligned_free(lookup);
    }
  };

  struct PQNavigationGatherBuffer {
    uint8_t* codes = nullptr;
    explicit PQNavigationGatherBuffer(uint64_t chunks) {
      alloc_aligned(reinterpret_cast<void**>(&codes),
                    ROUND_UP(defaults::MAX_GRAPH_DEGREE * chunks, 256), 256);
    }
    ~PQNavigationGatherBuffer() {
      aligned_free(codes);
    }
  };

  struct PQNavigationState {
    FixedChunkPQTable          table;
    std::unique_ptr<uint8_t[]> codes;
    uint64_t                   rows = 0, dimension = 0, chunks = 0;
    mutable std::mutex         pool_mutex;
    mutable std::vector<std::unique_ptr<PQNavigationQueryTable>>   tables;
    mutable std::vector<std::unique_ptr<PQNavigationGatherBuffer>> gathers;
    mutable size_t allocated_tables = 0, allocated_gathers = 0;
    mutable size_t pool_capacity = 1;

    template<typename Buffer>
    std::unique_ptr<Buffer> acquire(std::vector<std::unique_ptr<Buffer>>& pool,
                                    size_t& allocated) const {
      std::lock_guard<std::mutex> lock(pool_mutex);
      if (!pool.empty()) {
        auto buffer = std::move(pool.back());
        pool.pop_back();
        return buffer;
      }
      auto buffer = std::make_unique<Buffer>(chunks);
      ++allocated;
      return buffer;
    }

    template<typename Buffer>
    void release(std::unique_ptr<Buffer>               buffer,
                 std::vector<std::unique_ptr<Buffer>>& pool,
                 size_t& allocated) const noexcept {
      std::lock_guard<std::mutex> lock(pool_mutex);
      if (pool.size() < pool_capacity) {
        try {
          pool.push_back(std::move(buffer));
          return;
        } catch (...) {
          // Returning a workspace must not throw from a scorer destructor.
        }
      }
      --allocated;
    }

    void prepare(size_t count) const {
      std::lock_guard<std::mutex> lock(pool_mutex);
      pool_capacity = std::max<size_t>(1, count);
      while (tables.size() > pool_capacity) {
        tables.pop_back();
        --allocated_tables;
      }
      while (gathers.size() > pool_capacity) {
        gathers.pop_back();
        --allocated_gathers;
      }
      tables.reserve(pool_capacity);
      gathers.reserve(pool_capacity);
      while (tables.size() < pool_capacity) {
        tables.push_back(std::make_unique<PQNavigationQueryTable>(chunks));
        ++allocated_tables;
      }
      while (gathers.size() < pool_capacity) {
        gathers.push_back(std::make_unique<PQNavigationGatherBuffer>(chunks));
        ++allocated_gathers;
      }
    }
  };

  PQNavigationStore::PQNavigationStore(const std::string& prefix)
      : PQNavigationStore(get_pq_pivots_filename(prefix),
                          get_pq_compressed_filename(prefix), true) {
  }

  PQNavigationStore::PQNavigationStore(const std::string& pivots,
                                       const std::string& codes, bool resident)
      : state_(std::make_shared<PQNavigationState>()) {
    size_t centroids, dimension, rows, chunks;
    get_bin_metadata(pivots, centroids, dimension);
    get_bin_metadata(codes, rows, chunks);
    if (centroids != 256 || dimension == 0 || rows == 0 || chunks == 0 ||
        chunks > dimension || dimension > std::numeric_limits<int32_t>::max() ||
        rows > (std::numeric_limits<size_t>::max() - 2 * sizeof(uint32_t)) /
                   chunks ||
        (resident &&
         get_file_size(codes) != 2 * sizeof(uint32_t) + rows * chunks)) {
      throw std::runtime_error(
          "Invalid DiskANN navigation PQ metadata or code file size");
    }
    state_->rows = rows;
    state_->dimension = dimension;
    state_->chunks = chunks;
    // Reader-backed codes can have a dedicated layout (AiSAQ's page-aligned
    // rearranged file). The source reader validates that body, not this model.
    if (resident)
      load_bin<uint8_t>(codes, state_->codes, rows, chunks);
    state_->table.load_pq_centroid_bin(pivots.c_str(), chunks);
  }
  PQNavigationStore::~PQNavigationStore() = default;
  int64_t PQNavigationStore::Count() const {
    return state_->rows;
  }
  int64_t PQNavigationStore::Dimension() const {
    return state_->dimension;
  }
  uint64_t PQNavigationStore::ChunkCount() const {
    return state_->chunks;
  }
  size_t PQNavigationStore::MemorySize() const {
    std::lock_guard<std::mutex> lock(state_->pool_mutex);
    const auto                  dimension = state_->dimension;
    return (state_->codes ? state_->rows * state_->chunks : 0) +
           dimension *
               (256 * sizeof(float) * 2 + sizeof(uint32_t) + sizeof(float)) +
           (state_->chunks + 1) * sizeof(uint32_t) +
           state_->allocated_tables *
               ROUND_UP(256 * state_->chunks * sizeof(float), 256) +
           state_->allocated_gathers *
               ROUND_UP(defaults::MAX_GRAPH_DEGREE * state_->chunks, 256);
  }
  void PQNavigationStore::PrepareWorkspaces(size_t count) const {
    state_->prepare(count);
  }
  std::unique_ptr<NavigationDistanceComputer>
  PQNavigationStore::CreateDistanceComputer() const {
    return CreatePQDistanceComputer();
  }
  std::unique_ptr<PQNavigationDistanceComputer>
  PQNavigationStore::CreatePQDistanceComputer(PQCodeSource* source) const {
    return std::make_unique<PQNavigationDistanceComputer>(state_, source);
  }

  PQNavigationDistanceComputer::PQNavigationDistanceComputer(
      std::shared_ptr<PQNavigationState> state, PQCodeSource* source)
      : state_(std::move(state)),
        workspace_(state_->acquire(state_->tables, state_->allocated_tables)),
        source_(source) {
  }
  PQNavigationDistanceComputer::~PQNavigationDistanceComputer() {
    state_->release(std::move(workspace_), state_->tables,
                    state_->allocated_tables);
  }
  void PQNavigationDistanceComputer::set_query(const float* query) {
    state_->table.populate_chunk_distances(query, workspace_->lookup);
  }
  const float* PQNavigationDistanceComputer::lookup_table() const {
    return workspace_->lookup;
  }
  void PQNavigationDistanceComputer::evaluate_encoded(const uint8_t* codes,
                                                      uint64_t       count,
                                                      float* distances) {
    LookupPQCodes(codes, count, state_->chunks, workspace_->lookup, distances);
  }
  void PQNavigationDistanceComputer::compute_distances(const unsigned* ids,
                                                       uint64_t        count,
                                                       float* distances, float,
                                                       bool, QueryStats*) {
    if (count == 0)
      return;
    if (count > defaults::MAX_GRAPH_DEGREE)
      throw std::invalid_argument(
          "Navigation batch exceeds graph scratch capacity");
    // Iterators retain only their chunk-sized query table. The gather buffer
    // belongs to this scoring batch and is returned even if the code source
    // fails.
    auto release_gather = [this](PQNavigationGatherBuffer* buffer) {
      state_->release(std::unique_ptr<PQNavigationGatherBuffer>(buffer),
                      state_->gathers, state_->allocated_gathers);
    };
    std::unique_ptr<PQNavigationGatherBuffer, decltype(release_gather)>
        gathered(state_->acquire(state_->gathers, state_->allocated_gathers)
                     .release(),
                 release_gather);
    if (source_) {
      source_->gather(ids, count, state_->chunks, gathered->codes);
    } else if (state_->codes) {
      aggregate_coords(ids, count, state_->codes.get(), state_->chunks,
                       gathered->codes);
    } else {
      throw std::logic_error(
          "PQ navigation requires a resident or reader-backed code source");
    }
    evaluate_encoded(gathered->codes, count, distances);
  }
}  // namespace diskann
