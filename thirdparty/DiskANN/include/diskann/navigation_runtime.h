// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "diskann/defaults.h"
#include "diskann/utils.h"
#include "knowhere/operands.h"

namespace diskann {
  struct NavigationMetadataContext {
    std::string prefix;
    // Empty callbacks select local files. Otherwise existence and localization
    // must refer to the same source; a local cache cannot override that source.
    std::function<std::optional<bool>(const std::string&)> exists;
    std::function<bool(const std::string&)>                load_file;
  };

  class NavigationTypeError : public std::invalid_argument {
   public:
    using std::invalid_argument::invalid_argument;
  };

  enum class NavigationFilePurpose { Build, Load };

  std::string GetNavigationMetadataFilename(const std::string& prefix);
  void        WriteNavigationMetadata(const std::string& prefix,
                                      const std::string& codec);
  // Only an absent metadata file selects legacy PQ. All read/format errors
  // propagate; unknown codec names are rejected by the creation factory.
  std::string ReadNavigationCodec(const NavigationMetadataContext& context);

  bool IsSupportedNavigationCodec(const std::string& codec);

  class NavigationDistanceComputer {
   public:
    virtual ~NavigationDistanceComputer() = default;
    // Reset codec-owned request parameters when reusing a thread slot.
    virtual void ResetQueryParameters() {
    }
    // The graph supplies the same metric-prepared query as before.
    virtual void set_query(const float* query) = 0;
    virtual void compute_distances(const unsigned* ids, uint64_t count,
                                   float* distances) = 0;
    // A full candidate pool supplies its worst navigation distance (smaller
    // is better), fixed for this batch. Codecs may skip unlikely candidates
    // by returning +infinity. Without a finite threshold, score every ID.
    virtual void compute_distances(const unsigned* ids, uint64_t count,
                                   float* distances, float threshold) {
      compute_distances(ids, count, distances);
    }
  };

  using NavigationQuerySetup = std::function<void(NavigationDistanceComputer&)>;

  struct NavigationBuildContext {
    std::string              prefix;
    std::string              data_path;
    knowhere::DataFormatEnum data_type;
    Metric                   metric;
    size_t                   count, dimension;
    // Shared with SSD-PQ, valid only for the synchronous Build() call.
    const float* training_data;
    size_t       training_count;
  };

  class NavigationStore {
   public:
    virtual ~NavigationStore() = default;
    virtual const char* CodecName() const = 0;
    virtual bool        IsCompatible(knowhere::DataFormatEnum data_type,
                                     bool embedding_list) const = 0;
    virtual int         ValidateBuild() const = 0;
    virtual bool        NeedsTrainingSample() const = 0;
    virtual int         Build(const NavigationBuildContext& context) = 0;
    // Original navigation contribution to the build-cache memory gate.
    // Missing assets mean that this optional cache cannot be generated.
    virtual std::optional<size_t> BuildCacheMemorySize(
        const std::string& prefix) const = 0;
    // Deserialize files already prepared by the caller.
    virtual int  Load(const std::string& prefix) = 0;
    virtual void GetNecessaryFilenames(
        const std::string& prefix, std::vector<std::string>& filenames,
        NavigationFilePurpose purpose = NavigationFilePurpose::Build) const = 0;
    virtual uint64_t Count() const = 0;
    virtual uint64_t Dimension() const = 0;
    virtual size_t   MemorySize() const = 0;
    virtual size_t   QueryMemorySize(uint64_t aligned_dimension) const = 0;
    // The existing thread slot owns this computer and borrows it per batch.
    virtual std::unique_ptr<NavigationDistanceComputer> CreateDistanceComputer(
        uint64_t aligned_dimension,
        uint64_t batch_capacity = defaults::MAX_GRAPH_DEGREE) const = 0;
  };
  // Select a navigation implementation; unsupported codec names are rejected.
  std::unique_ptr<NavigationStore> CreateNavigationStore(
      const std::string& codec);
}  // namespace diskann
