// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "diskann/pq_navigation.h"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "diskann/parameters.h"
#include "diskann/defaults.h"
#include "diskann/pq_table.h"
#include "diskann/utils.h"

namespace diskann {
struct PQNavigationWorkspace {
  float* lookup = nullptr;
  uint8_t* gathered = nullptr;
  PQNavigationWorkspace(uint64_t dimension, uint64_t) {
    // Preserve the native reserve sizes and allocation order. Only the first
    // chunks columns are used; this refactor does not shrink the native buffers.
    const auto aligned_dimension = ROUND_UP(dimension, 8);
    alloc_aligned(reinterpret_cast<void**>(&gathered),
                  ROUND_UP(defaults::MAX_GRAPH_DEGREE * aligned_dimension, 256), 256);
    try {
      alloc_aligned(reinterpret_cast<void**>(&lookup),
                    ROUND_UP(256 * aligned_dimension * sizeof(float), 256), 256);
    } catch (...) {
      aligned_free(gathered);
      throw;
    }
  }
  ~PQNavigationWorkspace() {
    aligned_free(lookup);
    aligned_free(gathered);
  }
};

struct PQNavigationState {
  FixedChunkPQTable table;
  std::unique_ptr<uint8_t[]> codes;
  uint64_t rows = 0, dimension = 0, chunks = 0;
  mutable std::mutex pool_mutex;
  mutable std::vector<std::unique_ptr<PQNavigationWorkspace>> pool;
  mutable size_t allocated_workspaces = 0;

  std::unique_ptr<PQNavigationWorkspace> acquire() const {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!pool.empty()) {
      auto workspace = std::move(pool.back());
      pool.pop_back();
      return workspace;
    }
    auto workspace = std::make_unique<PQNavigationWorkspace>(dimension, chunks);
    ++allocated_workspaces;
    return workspace;
  }
  void release(std::unique_ptr<PQNavigationWorkspace> workspace) const noexcept {
    std::lock_guard<std::mutex> lock(pool_mutex);
    try {
      pool.push_back(std::move(workspace));
    } catch (...) {
      --allocated_workspaces;
    }
  }
};

PQNavigationStore::PQNavigationStore(const std::string& prefix)
    : PQNavigationStore(get_pq_pivots_filename(prefix),
                        get_pq_compressed_filename(prefix), true) {}

PQNavigationStore::PQNavigationStore(const std::string& pivots,
                                    const std::string& codes, bool resident)
    : state_(std::make_shared<PQNavigationState>()) {
  size_t centroids, dimension, rows, chunks;
  get_bin_metadata(pivots, centroids, dimension);
  get_bin_metadata(codes, rows, chunks);
  if (centroids != 256 || dimension == 0 || rows == 0 || chunks == 0 ||
      chunks > dimension || dimension > std::numeric_limits<int32_t>::max() ||
      rows > (std::numeric_limits<size_t>::max() - 2 * sizeof(uint32_t)) / chunks ||
      (resident && get_file_size(codes) != 2 * sizeof(uint32_t) + rows * chunks)) {
    throw std::runtime_error("Invalid DiskANN navigation PQ metadata or code file size");
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
int64_t PQNavigationStore::Count() const { return state_->rows; }
int64_t PQNavigationStore::Dimension() const { return state_->dimension; }
uint64_t PQNavigationStore::ChunkCount() const { return state_->chunks; }
size_t PQNavigationStore::MemorySize() const {
  std::lock_guard<std::mutex> lock(state_->pool_mutex);
  const auto dimension = state_->dimension;
  const auto aligned_dimension = ROUND_UP(dimension, 8);
  return (state_->codes ? state_->rows * state_->chunks : 0) +
         dimension * (256 * sizeof(float) * 2 + sizeof(uint32_t) + sizeof(float)) +
         (state_->chunks + 1) * sizeof(uint32_t) +
         state_->allocated_workspaces *
             (ROUND_UP(256 * aligned_dimension * sizeof(float), 256) +
              ROUND_UP(defaults::MAX_GRAPH_DEGREE * aligned_dimension, 256));
}
void PQNavigationStore::PrepareWorkspaces(size_t count) const {
  std::vector<std::unique_ptr<PQNavigationWorkspace>> workspaces;
  workspaces.reserve(count);
  for (size_t i = 0; i < count; ++i)
    workspaces.push_back(state_->acquire());
  for (auto& workspace : workspaces)
    state_->release(std::move(workspace));
}
std::unique_ptr<NavigationDistanceComputer>
PQNavigationStore::CreateDistanceComputer(const NavigationQueryOptions&) const {
  return CreatePQDistanceComputer();
}
std::unique_ptr<PQNavigationDistanceComputer>
PQNavigationStore::CreatePQDistanceComputer(PQCodeSource* source) const {
  return std::make_unique<PQNavigationDistanceComputer>(state_, source);
}

PQNavigationDistanceComputer::PQNavigationDistanceComputer(
    std::shared_ptr<PQNavigationState> state, PQCodeSource* source)
    : state_(std::move(state)), workspace_(state_->acquire()), source_(source) {}
PQNavigationDistanceComputer::~PQNavigationDistanceComputer() {
  state_->release(std::move(workspace_));
}
void PQNavigationDistanceComputer::set_query(const float* query) {
  state_->table.populate_chunk_distances(query, workspace_->lookup);
}
const float* PQNavigationDistanceComputer::lookup_table() const {
  return workspace_->lookup;
}
void PQNavigationDistanceComputer::evaluate_encoded(const uint8_t* codes,
                                                     uint64_t count,
                                                     float* distances) {
  pq_dist_lookup(codes, count, state_->chunks, workspace_->lookup, distances);
}
void PQNavigationDistanceComputer::compute_distances(
    const unsigned* ids, uint64_t count, float* distances, float, bool, QueryStats*) {
  if (count == 0)
    return;
  if (count > defaults::MAX_GRAPH_DEGREE)
    throw std::invalid_argument("Navigation batch exceeds graph scratch capacity");
  if (source_) {
    source_->gather(ids, count, state_->chunks, workspace_->gathered);
  } else if (state_->codes) {
    aggregate_coords(ids, count, state_->codes.get(), state_->chunks,
                     workspace_->gathered);
  } else {
    throw std::logic_error("PQ navigation requires a resident or reader-backed code source");
  }
  evaluate_encoded(workspace_->gathered, count, distances);
}
}  // namespace diskann
