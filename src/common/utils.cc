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

#include "knowhere/utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "faiss/IndexIVFFlat.h"
#include "faiss/impl/FaissException.h"
#include "faiss/index_io.h"
#include "io/memory_io.h"
#include "knowhere/log.h"
#include "simd/hook.h"

namespace knowhere {

const float FloatAccuracy = 0.00001;

// normalize one vector and return its norm
float
NormalizeVec(float* x, int32_t d) {
    float norm_l2_sqr = faiss::fvec_norm_L2sqr(x, d);
    if (norm_l2_sqr > 0 && std::abs(1.0f - norm_l2_sqr) > FloatAccuracy) {
        float norm_l2 = std::sqrt(norm_l2_sqr);
        for (int32_t i = 0; i < d; i++) {
            x[i] = x[i] / norm_l2;
        }
        return norm_l2;
    }
    return 1.0f;
}

// normalize all vectors and return their norms
std::vector<float>
NormalizeVecs(float* x, size_t rows, int32_t dim) {
    std::vector<float> norms(rows);
    for (size_t i = 0; i < rows; i++) {
        norms[i] = NormalizeVec(x + i * dim, dim);
    }
    return norms;
}

void
Normalize(const DataSet& dataset) {
    auto rows = dataset.GetRows();
    auto dim = dataset.GetDim();
    float* data = (float*)dataset.GetTensor();

    LOG_KNOWHERE_DEBUG_ << "vector normalize, rows " << rows << ", dim " << dim;

    for (int32_t i = 0; i < rows; i++) {
        NormalizeVec(data + i * dim, dim);
    }
}

// copy and return normalized vectors
std::unique_ptr<float[]>
CopyAndNormalizeVecs(const float* x, size_t rows, int32_t dim) {
    auto x_normalized = std::make_unique<float[]>(rows * dim);
    std::copy_n(x, rows * dim, x_normalized.get());
    NormalizeVecs(x_normalized.get(), rows, dim);
    return x_normalized;
}

void
ConvertIVFFlatIfNeeded(const BinarySet& binset, const MetricType metric_type, const uint8_t* raw_data,
                       const size_t raw_size) {
    std::vector<std::string> names = {"IVF",  // compatible with knowhere-1.x
                                      knowhere::IndexEnum::INDEX_FAISS_IVFFLAT};
    auto binary = binset.GetByNames(names);
    if (binary == nullptr) {
        return;
    }

    MemoryIOReader reader(binary->data.get(), binary->size);

    try {
        uint32_t h;
        reader.read(&h, sizeof(h), 1);

        // only read IVF_FLAT index header
        std::unique_ptr<faiss::IndexIVFFlat> ivfl = std::make_unique<faiss::IndexIVFFlat>(faiss::IndexIVFFlat());
        faiss::read_ivf_header(ivfl.get(), &reader);
        ivfl->code_size = ivfl->d * sizeof(float);

        // is_cosine is not defined in IVF_FLAT_NM, so mark it from config
        ivfl->is_cosine = IsMetricType(metric_type, knowhere::metric::COSINE);

        auto remains = binary->size - reader.tellg() - sizeof(uint32_t) - sizeof(ivfl->invlists->nlist) -
                       sizeof(ivfl->invlists->code_size);
        auto invlist_size = sizeof(uint32_t) + sizeof(size_t) + ivfl->nlist * sizeof(size_t);
        auto ids_size = ivfl->ntotal * sizeof(faiss::Index::idx_t);
        // auto codes_size = ivfl->d * ivfl->ntotal * sizeof(float);

        // IVF_FLAT_NM format, need convert to new format
        if (remains == invlist_size + ids_size) {
            faiss::read_InvertedLists_nm(ivfl.get(), &reader);
            ivfl->restore_codes(raw_data, raw_size);

            // over-write IVF_FLAT_NM binary with native IVF_FLAT binary
            MemoryIOWriter writer;
            faiss::write_index(ivfl.get(), &writer);
            std::shared_ptr<uint8_t[]> data(writer.data());
            binary->data = data;
            binary->size = writer.tellg();

            LOG_KNOWHERE_INFO_ << "Convert IVF_FLAT_NM to native IVF_FLAT, rows " << ivfl->ntotal << ", dim "
                               << ivfl->d;
        }
    } catch (...) {
        // not IVF_FLAT_NM format, do nothing
        return;
    }
}

bool
UseDiskLoad(const std::string& index_type, const std::string& /*version*/) {
    return !index_type.compare(IndexEnum::INDEX_DISKANN);
}

}  // namespace knowhere
