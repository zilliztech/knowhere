// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "index/diskann/navigation_store.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "diskann/aux_utils.h"
#include "index/diskann/diskann_config.h"
#include "index/diskann/rabitq_store.h"

namespace knowhere {
namespace {
const DiskANNNavigationConfig&
NavigationConfig(const DiskANNConfig& config) {
    return static_cast<const DiskANNNavigationConfig&>(config);
}

class RaBitQNavigationBuilder final : public diskann::NavigationBuilder {
 public:
    explicit RaBitQNavigationBuilder(uint8_t bits) : bits_(bits) {
    }
    void
    build(const diskann::BuildConfig&, const diskann::PreparedBuildContext& context,
          const diskann::NavigationTrainingData&) const override {
        RaBitQStore::BuildFromFloatBin(context.prepared_source, RaBitQStore::SidecarFilename(context.prefix), bits_);
    }

 private:
    uint8_t bits_;
};

struct NavigationCodec {
    bool external;
    Status (*validate)(const DiskANNNavigationConfig&, std::string*);
    std::optional<uint64_t> (*estimate)(const DiskANNConfig&, int64_t, int64_t);
    NavigationFileSet (*files)(const std::string&);
    std::unique_ptr<diskann::NavigationBuilder> (*builder)(const DiskANNConfig&);
    std::unique_ptr<NavigationStore> (*load)(const std::string&);
};

const std::unordered_map<std::string, NavigationCodec>&
Codecs() {
    static const std::unordered_map<std::string, NavigationCodec> codecs = {
        {"PQ",
         {false, [](const DiskANNNavigationConfig&, std::string*) { return Status::success; },
          [](const DiskANNConfig&, int64_t, int64_t) -> std::optional<uint64_t> { return 0; },
          [](const std::string& prefix) {
              return NavigationFileSet{diskann::pq_navigation_files(prefix), {}};
          },
          nullptr, [](const std::string&) -> std::unique_ptr<NavigationStore> { return nullptr; }}},
        {"RABITQ",
         {true,
          [](const DiskANNNavigationConfig& config, std::string* error) {
              const auto metric = config.metric_type.value_or(metric::L2);
              if (metric != metric::L2 && metric != metric::IP && metric != metric::COSINE) {
                  if (error)
                      *error = "DISKANN_RABITQ supports L2, IP and COSINE";
                  return Status::invalid_metric_type;
              }
              const auto bits = config.rbq_bits.value_or(1);
              if (bits < 1 || bits > 9) {
                  if (error)
                      *error = "DISKANN_RABITQ supports rbq_bits in [1, 9]";
                  return Status::invalid_args;
              }
              return Status::success;
          },
          [](const DiskANNConfig& config, int64_t rows, int64_t dim) -> std::optional<uint64_t> {
              const auto bits = NavigationConfig(config).rbq_bits;
              if (!bits.has_value())
                  return std::nullopt;
              if (dim <= 0 || dim >= std::numeric_limits<int>::max()) {
                  throw std::invalid_argument("invalid DiskANN navigation dimension");
              }
              const auto prepared_dim = dim + (config.metric_type.value_or(metric::L2) == metric::IP ? 1 : 0);
              return RaBitQStore::EstimateMemorySize(rows, prepared_dim, static_cast<uint8_t>(bits.value()));
          },
          [](const std::string& prefix) {
              return NavigationFileSet{{RaBitQStore::SidecarFilename(prefix)}, {}};
          },
          [](const DiskANNConfig& config) -> std::unique_ptr<diskann::NavigationBuilder> {
              return std::make_unique<RaBitQNavigationBuilder>(
                  static_cast<uint8_t>(NavigationConfig(config).rbq_bits.value_or(1)));
          },
          [](const std::string& prefix) -> std::unique_ptr<NavigationStore> {
              return std::make_unique<RaBitQStore>(RaBitQStore::SidecarFilename(prefix));
          }}}};
    return codecs;
}

const NavigationCodec&
Codec(const DiskANNConfig& config) {
    // Older native/AiSAQ callers have a plain DiskANNConfig and use PQ. This
    // cast only reads the selector; algorithm dispatch is by registered name.
    const auto* navigation = dynamic_cast<const DiskANNNavigationConfig*>(&config);
    const auto name = navigation ? navigation->navigation_codec.value_or("PQ") : "PQ";
    const auto found = Codecs().find(name);
    if (found == Codecs().end())
        throw std::invalid_argument("unsupported DiskANN navigation codec: " + name);
    return found->second;
}
}  // namespace

expected<std::string>
DetectNavigationCodec(const DiskANNNavigationConfig& config, const std::string& prefix, milvus::FileManager& manager) {
    std::vector<std::string> found;
    for (const auto& [name, codec] : Codecs()) {
        bool present = false;
        // Any exclusive required artifact identifies a possible codec, even
        // when its primary file is missing. Never disguise an incomplete
        // model as absence and silently select another codec.
        for (const auto& path : codec.files(prefix).required) {
            const auto exists = manager.IsExisted(path);
            std::error_code error;
            const auto local_exists = std::filesystem::exists(path, error);
            if (!exists.has_value() || error) {
                return expected<std::string>::Err(Status::disk_file_error, "Cannot query navigation file: " + path);
            }
            // A fresh LocalFileManager has not registered already-localized
            // files. Remote query failures still remain errors above.
            present = present || exists.value() || local_exists;
        }
        if (present)
            found.push_back(name);
    }
    if (found.empty()) {
        return expected<std::string>::Err(Status::disk_file_error, "No stored DiskANN navigation model found");
    }
    if (config.navigation_codec.has_value()) {
        const auto& requested = config.navigation_codec.value();
        if (std::find(found.begin(), found.end(), requested) == found.end()) {
            return expected<std::string>::Err(Status::invalid_serialized_index_type,
                                              "Requested navigation codec does not match stored model: " + requested);
        }
        return requested;
    }
    if (found.size() != 1) {
        return expected<std::string>::Err(
            Status::invalid_serialized_index_type,
            "Ambiguous DiskANN navigation files; specify navigation_codec to disambiguate");
    }
    return found.front();
}

Status
ValidateNavigationConfig(const DiskANNNavigationConfig& config, std::string* error) {
    const auto found = Codecs().find(config.navigation_codec.value_or("PQ"));
    if (found == Codecs().end()) {
        if (error)
            *error = "unsupported DiskANN navigation codec";
        return Status::invalid_args;
    }
    return found->second.validate(config, error);
}

bool
UsesExternalNavigation(const DiskANNConfig& config) {
    return Codec(config).external;
}

std::optional<uint64_t>
EstimateNavigationMemory(const DiskANNConfig& config, int64_t rows, int64_t dim) {
    const auto* navigation = dynamic_cast<const DiskANNNavigationConfig*>(&config);
    if (navigation && !navigation->navigation_codec.has_value())
        return std::nullopt;
    return Codec(config).estimate(config, rows, dim);
}

NavigationFileSet
NavigationFiles(const DiskANNConfig& config, const std::string& prefix) {
    return Codec(config).files(prefix);
}

std::vector<std::string>
AllNavigationFiles(const std::string& prefix) {
    std::vector<std::string> files;
    for (const auto& [name, codec] : Codecs()) {
        const auto declared = codec.files(prefix);
        files.insert(files.end(), declared.required.begin(), declared.required.end());
        files.insert(files.end(), declared.optional.begin(), declared.optional.end());
    }
    return files;
}

std::unique_ptr<diskann::NavigationBuilder>
CreateNavigationBuilder(const DiskANNConfig& config, std::unique_ptr<diskann::NavigationBuilder> native_pq) {
    const auto& codec = Codec(config);
    return codec.builder ? codec.builder(config) : std::move(native_pq);
}

std::unique_ptr<NavigationStore>
LoadNavigationStore(const DiskANNConfig& config, const std::string& prefix) {
    return Codec(config).load(prefix);
}
}  // namespace knowhere
