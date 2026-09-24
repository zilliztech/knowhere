// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "index/diskann/navigation_store.h"

namespace faiss {
struct Index;
struct IndexPreTransform;
struct IndexRaBitQ;
struct RandomRotationMatrix;
}  // namespace faiss

namespace knowhere {

class RaBitQStore final : public NavigationStore {
 public:
    static std::string
    SidecarFilename(const std::string& index_prefix);

    static void
    BuildFromFloatBin(const std::string& data_path, const std::string& sidecar_path, uint8_t rbq_bits);

    // Persistent codes and model only; excludes graph-engine scratch and node cache.
    static uint64_t
    EstimateMemorySize(int64_t rows, int64_t prepared_dim, uint8_t rbq_bits);

    explicit RaBitQStore(const std::string& sidecar_path);
    ~RaBitQStore();

    RaBitQStore(const RaBitQStore&) = delete;
    RaBitQStore&
    operator=(const RaBitQStore&) = delete;

    std::unique_ptr<diskann::NavigationDistanceComputer>
    CreateDistanceComputer(uint8_t query_bits = 4) const;

    std::unique_ptr<diskann::NavigationDistanceComputer>
    CreateDistanceComputer(const DiskANNConfig& config) const override;

    int64_t
    Count() const override;

    int64_t
    Dimension() const override;

    uint8_t
    Bits() const;

    size_t
    CodeSize() const;

    size_t
    MemorySize() const override;

 private:
    void
    Validate();

    std::unique_ptr<faiss::Index> index_;
    const faiss::IndexPreTransform* pretransform_ = nullptr;
    const faiss::RandomRotationMatrix* rotation_ = nullptr;
    const faiss::IndexRaBitQ* rabitq_ = nullptr;
};

}  // namespace knowhere
