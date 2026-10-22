// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "index/diskann/navigation_store.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "diskann/aux_utils.h"
#include "diskann/pq_navigation.h"
#include "index/diskann/diskann_config.h"
#include "index/diskann/pq_navigation_config.h"
#include "index/diskann/rabitq_navigation_config.h"
#include "index/diskann/rabitq_store.h"

namespace knowhere {
namespace {
template <typename Parameters>
Status
ParseParameters(Parameters& parameters, const DiskANNConfig& config, PARAM_TYPE phase, std::string* error) {
    const auto* navigation = dynamic_cast<const DiskANNNavigationConfig*>(&config);
    Json json = navigation ? navigation->CodecParameters() : Json::object();
    const auto status = Config::FormatAndCheck(parameters, json, error);
    if (status != Status::success)
        return status;
    return Config::Load(parameters, json, phase, error);
}

double
GetPQNavigationBudget(const DiskANNConfig& config) {
    if (const auto* pq = dynamic_cast<const PQDiskANNConfig*>(&config))
        return pq->pq_code_budget_gb.value_or(0);
    PQDiskANNConfig parameters;
    std::string error;
    if (ParseParameters(parameters, config, PARAM_TYPE::TRAIN, &error) != Status::success)
        throw std::invalid_argument(error);
    return parameters.pq_code_budget_gb.value_or(0);
}

expected<NavigationQueryFactory>
PQQueryFactory(std::shared_ptr<NavigationStore> model, const DiskANNConfig&) {
    auto pq = std::dynamic_pointer_cast<diskann::PQNavigationStore>(model);
    if (!pq)
        return expected<NavigationQueryFactory>::Err(Status::invalid_index_error, "PQ model type mismatch");
    return NavigationQueryFactory([pq = std::move(pq)]() { return pq->CreateDistanceComputer(); });
}

expected<NavigationQueryFactory>
RaBitQQueryFactory(std::shared_ptr<NavigationStore> model, const DiskANNConfig& config) {
    RaBitQQueryConfig parameters;
    std::string error;
    const auto status = ParseParameters(parameters, config, PARAM_TYPE::SEARCH, &error);
    if (status != Status::success)
        return expected<NavigationQueryFactory>::Err(status, error);
    auto rabitq = std::dynamic_pointer_cast<RaBitQStore>(model);
    if (!rabitq)
        return expected<NavigationQueryFactory>::Err(Status::invalid_index_error, "RaBitQ model type mismatch");
    const auto bits = static_cast<uint8_t>(parameters.rbq_bits_query.value());
    return NavigationQueryFactory(
        [rabitq = std::move(rabitq), bits]() { return rabitq->CreateDistanceComputer(bits); });
}

class RaBitQNavigationBuilder final : public diskann::NavigationBuilder {
 public:
    explicit RaBitQNavigationBuilder(uint8_t bits) : bits_(bits) {
    }
    std::vector<std::string>
    files(const std::string& prefix) const override {
        return {RaBitQStore::SidecarFilename(prefix)};
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
    NavigationFeatures features;
    Status (*validate)(const DiskANNNavigationConfig&, PARAM_TYPE, std::string*);
    std::optional<uint64_t> (*estimate)(const DiskANNConfig&, int64_t, int64_t);
    NavigationFileSet (*files)(const std::string&);
    std::unique_ptr<diskann::NavigationBuilder> (*builder)(const DiskANNConfig&, NavigationInputType);
    std::unique_ptr<NavigationStore> (*load)(const std::string&, size_t);
    expected<NavigationQueryFactory> (*query)(std::shared_ptr<NavigationStore>, const DiskANNConfig&);
    expected<NavigationIndexMetadata> (*metadata)(const NavigationStore&, const DiskANNConfig&);
};

const std::unordered_map<std::string, NavigationCodec>&
Codecs() {
    static const std::unordered_map<std::string, NavigationCodec> codecs = {
        {"PQ",
         {{false, true, true, true},
          [](const DiskANNNavigationConfig& config, PARAM_TYPE phase, std::string* error) {
              if (phase != PARAM_TYPE::TRAIN && phase != STATIC)
                  return Status::success;
              PQDiskANNConfig parameters;
              return ParseParameters(parameters, config, phase, error);
          },
          [](const DiskANNConfig&, int64_t, int64_t) -> std::optional<uint64_t> { return 0; },
          [](const std::string& prefix) {
              return NavigationFileSet{diskann::pq_navigation_files(prefix), {}};
          },
          [](const DiskANNConfig& config, NavigationInputType input) -> std::unique_ptr<diskann::NavigationBuilder> {
              switch (input) {
                  case NavigationInputType::FP32:
                      return diskann::make_pq_navigation_builder<float>(GetPQNavigationBudget(config));
                  case NavigationInputType::FP16:
                      return diskann::make_pq_navigation_builder<knowhere::fp16>(GetPQNavigationBudget(config));
                  case NavigationInputType::BF16:
                      return diskann::make_pq_navigation_builder<knowhere::bf16>(GetPQNavigationBudget(config));
              }
              throw std::invalid_argument("unsupported navigation input type");
          },
          [](const std::string& prefix, size_t workspaces) -> std::unique_ptr<NavigationStore> {
              auto model = std::make_unique<diskann::PQNavigationStore>(prefix);
              model->PrepareWorkspaces(workspaces);
              return model;
          },
          PQQueryFactory,
          [](const NavigationStore& model, const DiskANNConfig& config) -> expected<NavigationIndexMetadata> {
              const auto* pq = dynamic_cast<const diskann::PQNavigationStore*>(&model);
              if (!pq)
                  return expected<NavigationIndexMetadata>::Err(Status::invalid_index_error, "PQ model type mismatch");
              PQNavigationMetadataConfig parameters;
              std::string error;
              const auto status = ParseParameters(parameters, config, FEDER, &error);
              if (status != Status::success)
                  return expected<NavigationIndexMetadata>::Err(status, error);
              Json navigation = {{"codec", "PQ"}, {"chunks", pq->ChunkCount()}, {"pq_code_budget_gb", nullptr}};
              Json legacy = Json::object();
              const auto budget = parameters.Budget();
              if (budget.has_value()) {
                  navigation["pq_code_budget_gb"] = *budget;
                  legacy["pq_code_budget_gb"] = *budget;
              }
              return NavigationIndexMetadata{std::move(navigation), std::move(legacy)};
          }}},
        {"RABITQ",
         {{true, false, false, false},
          [](const DiskANNNavigationConfig& config, PARAM_TYPE phase, std::string* error) {
              const auto metric = config.metric_type.value_or(metric::L2);
              if (metric != metric::L2 && metric != metric::IP && metric != metric::COSINE) {
                  if (error)
                      *error = "DISKANN_RABITQ supports L2, IP and COSINE";
                  return Status::invalid_metric_type;
              }
              if (phase == PARAM_TYPE::SEARCH || phase == PARAM_TYPE::ITERATOR) {
                  RaBitQQueryConfig parameters;
                  return ParseParameters(parameters, config, phase, error);
              }
              RaBitQBuildConfig parameters;
              return ParseParameters(parameters, config, phase, error);
          },
          [](const DiskANNConfig& config, int64_t rows, int64_t dim) -> std::optional<uint64_t> {
              RaBitQBuildConfig parameters;
              std::string error;
              if (ParseParameters(parameters, config, STATIC, &error) != Status::success)
                  throw std::invalid_argument(error);
              const auto bits = parameters.rbq_bits;
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
          [](const DiskANNConfig& config, NavigationInputType) -> std::unique_ptr<diskann::NavigationBuilder> {
              RaBitQBuildConfig parameters;
              std::string error;
              if (ParseParameters(parameters, config, PARAM_TYPE::TRAIN, &error) != Status::success)
                  throw std::invalid_argument(error);
              return std::make_unique<RaBitQNavigationBuilder>(static_cast<uint8_t>(parameters.rbq_bits.value_or(1)));
          },
          [](const std::string& prefix, size_t) -> std::unique_ptr<NavigationStore> {
              return std::make_unique<RaBitQStore>(RaBitQStore::SidecarFilename(prefix));
          },
          RaBitQQueryFactory,
          [](const NavigationStore& model, const DiskANNConfig&) -> expected<NavigationIndexMetadata> {
              const auto* rabitq = dynamic_cast<const RaBitQStore*>(&model);
              if (!rabitq)
                  return expected<NavigationIndexMetadata>::Err(Status::invalid_index_error,
                                                                "RaBitQ model type mismatch");
              return NavigationIndexMetadata{
                  {{"codec", "RABITQ"}, {"rbq_bits", rabitq->Bits()}, {"code_size", rabitq->CodeSize()}},
                  Json::object()};
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
ValidateNavigationConfig(const DiskANNNavigationConfig& config, PARAM_TYPE phase, std::string* error) {
    const auto found = Codecs().find(config.navigation_codec.value_or("PQ"));
    if (found == Codecs().end()) {
        if (error)
            *error = "unsupported DiskANN navigation codec";
        return Status::invalid_args;
    }
    return found->second.validate(config, phase, error);
}

NavigationFeatures
GetNavigationFeatures(const DiskANNConfig& config) {
    return Codec(config).features;
}

expected<NavigationQueryFactory>
CreateNavigationQueryFactory(const std::string& loaded_codec, std::shared_ptr<NavigationStore> model,
                             const DiskANNConfig& config) {
    const auto found = Codecs().find(loaded_codec);
    if (!model || found == Codecs().end())
        return expected<NavigationQueryFactory>::Err(Status::invalid_index_error, "Missing loaded navigation model");
    return found->second.query(std::move(model), config);
}

expected<NavigationIndexMetadata>
GetNavigationIndexMetadata(const std::string& loaded_codec, const NavigationStore& model, const DiskANNConfig& config) {
    const auto found = Codecs().find(loaded_codec);
    if (found == Codecs().end())
        return expected<NavigationIndexMetadata>::Err(Status::invalid_index_error, "Unknown loaded navigation codec");
    return found->second.metadata(model, config);
}

std::optional<uint64_t>
EstimateAdditionalNavigationMemory(const DiskANNConfig& config, int64_t rows, int64_t dim) {
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
CreateNavigationBuilder(const DiskANNConfig& config, NavigationInputType input_type) {
    const auto& codec = Codec(config);
    return codec.builder(config, input_type);
}

std::unique_ptr<NavigationStore>
LoadNavigationStore(const DiskANNConfig& config, const std::string& prefix, size_t workspaces) {
    return Codec(config).load(prefix, workspaces);
}
}  // namespace knowhere
