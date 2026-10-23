// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "diskann/navigation_runtime.h"
namespace faiss {
  struct Index;
  struct IndexPreTransform;
  struct IndexRaBitQ;
  struct RandomRotationMatrix;
}  // namespace faiss
namespace diskann {
  struct RaBitQNavigationBuildOptions {
    uint8_t bits = 1;
  };
  class RaBitQNavigationStore final : public NavigationStore {
   public:
    explicit RaBitQNavigationStore(RaBitQNavigationBuildOptions options = {});
    ~RaBitQNavigationStore() override;
    const char* CodecName() const override {
      return "RABITQ";
    }
    bool IsCompatible(knowhere::DataFormatEnum data_type,
                      bool                     embedding_list) const override {
      return data_type == knowhere::DataFormatEnum::fp32 && !embedding_list;
    }
    int  ValidateBuild() const override;
    bool NeedsTrainingSample() const override {
      return false;
    }
    int                   Build(const NavigationBuildContext& context) override;
    int                   Load(const std::string& prefix) override;
    void                  GetNecessaryFilenames(const std::string&        prefix,
                                                std::vector<std::string>& files,
                                                NavigationFilePurpose     purpose =
                                                    NavigationFilePurpose::Build) const override;
    std::optional<size_t> BuildCacheMemorySize(
        const std::string& prefix) const override;
    uint64_t Count() const override;
    uint64_t Dimension() const override;
    size_t   MemorySize() const override;
    size_t   QueryMemorySize(uint64_t aligned_dimension) const override;
    std::unique_ptr<NavigationDistanceComputer> CreateDistanceComputer(
        uint64_t aligned_dimension,
        uint64_t batch_capacity = defaults::MAX_GRAPH_DEGREE) const override;
    static void        ConfigureQuery(NavigationDistanceComputer& computer,
                                      uint8_t                     query_bits);
    static std::string SidecarFilename(const std::string& prefix);

   private:
    void                            Validate();
    RaBitQNavigationBuildOptions    options_;
    std::shared_ptr<faiss::Index>   index_;
    const faiss::IndexPreTransform* pretransform_ = nullptr;
    const faiss::RandomRotationMatrix* rotation_ = nullptr;
    const faiss::IndexRaBitQ*          rabitq_ = nullptr;
  };
}  // namespace diskann
