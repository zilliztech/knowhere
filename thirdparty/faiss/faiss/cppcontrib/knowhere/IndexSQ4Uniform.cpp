// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <faiss/cppcontrib/knowhere/IndexSQ4Uniform.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/cppcontrib/knowhere/FaissHook.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/prefetch.h>
#include <knowhere/utils.h>



namespace faiss::cppcontrib::knowhere {

//////////////////////////////////////////////////////////////////////////////////
// SQ4UniformCosineDistanceComputer implementation
//////////////////////////////////////////////////////////////////////////////////

SQ4UniformCosineDistanceComputer::SQ4UniformCosineDistanceComputer(
        const int d_,
        std::unique_ptr<DistanceComputer>&& basedis_)
        : basedis(std::move(basedis_)), d(d_) {}

void SQ4UniformCosineDistanceComputer::set_query(const float* x) {
    // For COSINE metric, normalize query vector before computing distances
    // At this point, data has already been converted to float by knowhere layer
    std::vector<float> normalized_query(d);
    std::copy_n(x, d, normalized_query.begin());

    // Normalize using knowhere's function
    ::knowhere::NormalizeVec<float>(normalized_query.data(), d);

    // Set the normalized query to base distance computer
    basedis->set_query(normalized_query.data());

    // Store normalized query for later use
    query_storage = std::move(normalized_query);
}

float SQ4UniformCosineDistanceComputer::operator()(idx_t i) {
    float l2_sqr_dis = (*basedis)(i);
    return 1.0f - 0.5f * l2_sqr_dis;
}

void SQ4UniformCosineDistanceComputer::distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    basedis->distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);

    dis0 = 1.0f - 0.5f * dis0;
    dis1 = 1.0f - 0.5f * dis1;
    dis2 = 1.0f - 0.5f * dis2;
    dis3 = 1.0f - 0.5f * dis3;
}

float SQ4UniformCosineDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    float l2_sqr_dis = basedis->symmetric_dis(i, j);
    return 1.0f - 0.5f * l2_sqr_dis;
}

//////////////////////////////////////////////////////////////////////////////////
// WithSQ4UniformNormIPDistanceComputer implementation
//////////////////////////////////////////////////////////////////////////////////

WithSQ4UniformNormIPDistanceComputer::WithSQ4UniformNormIPDistanceComputer(
        const float* l2_norms_sqr_,
        const int d_,
        std::unique_ptr<DistanceComputer>&& basedis_)
        : basedis(std::move(basedis_)), l2_norms_sqr(l2_norms_sqr_), d(d_) {}

void WithSQ4UniformNormIPDistanceComputer::set_query(const float* x) {
    if (x != nullptr) {
        // For IP: compute query norm squared for distance conversion
        query_norm_sqr = faiss::cppcontrib::knowhere::fvec_norm_L2sqr(x, d);
        if (query_norm_sqr <= 0) {
            query_norm_sqr = 1.0f;
        }
        basedis->set_query(x);
    } else {
        query_norm_sqr = 0;
        basedis->set_query(nullptr);
    }
}

float WithSQ4UniformNormIPDistanceComputer::operator()(idx_t i) {
    float l2_sqr_dis = (*basedis)(i);
    prefetch_L2(l2_norms_sqr + i);
    const float base_norm_sqr = l2_norms_sqr[i];
    return 0.5f * (query_norm_sqr + base_norm_sqr - l2_sqr_dis);
}

void WithSQ4UniformNormIPDistanceComputer::distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    basedis->distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);

    prefetch_L2(l2_norms_sqr + idx0);
    prefetch_L2(l2_norms_sqr + idx1);
    prefetch_L2(l2_norms_sqr + idx2);
    prefetch_L2(l2_norms_sqr + idx3);

    const float base_norm_sqr0 = l2_norms_sqr[idx0];
    const float base_norm_sqr1 = l2_norms_sqr[idx1];
    const float base_norm_sqr2 = l2_norms_sqr[idx2];
    const float base_norm_sqr3 = l2_norms_sqr[idx3];

    dis0 = 0.5f * (query_norm_sqr + base_norm_sqr0 - dis0);
    dis1 = 0.5f * (query_norm_sqr + base_norm_sqr1 - dis1);
    dis2 = 0.5f * (query_norm_sqr + base_norm_sqr2 - dis2);
    dis3 = 0.5f * (query_norm_sqr + base_norm_sqr3 - dis3);
}

float WithSQ4UniformNormIPDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    float l2_sqr_dis = basedis->symmetric_dis(i, j);

    prefetch_L2(l2_norms_sqr + i);
    prefetch_L2(l2_norms_sqr + j);

    const float norm_i_sqr = l2_norms_sqr[i];
    const float norm_j_sqr = l2_norms_sqr[j];

    return 0.5f * (norm_i_sqr + norm_j_sqr - l2_sqr_dis);
}

//////////////////////////////////////////////////////////////////////////////////
// IndexScalarQuantizer4bitUniformCosine implementation
//////////////////////////////////////////////////////////////////////////////////

IndexScalarQuantizer4bitUniformCosine::IndexScalarQuantizer4bitUniformCosine(
        int d)
        : IndexScalarQuantizer(
                  d,
                  ScalarQuantizer::QT_4bit_uniform,
                  METRIC_INNER_PRODUCT) {
    is_cosine = true;

    sq.rangestat = ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.01;
}

IndexScalarQuantizer4bitUniformCosine::IndexScalarQuantizer4bitUniformCosine()
        : IndexScalarQuantizer() {
    metric_type = METRIC_INNER_PRODUCT;
    is_cosine = true;

    sq.rangestat = ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.01;
}

void IndexScalarQuantizer4bitUniformCosine::train(idx_t n, const float* x) {
    // For COSINE metric, normalize vectors before training
    // Use knowhere's CopyAndNormalizeVecs to avoid modifying input data
    auto normalized_data = ::knowhere::CopyAndNormalizeVecs<float>(x, n, d);

    // Train on normalized data
    sq.train(n, normalized_data.get());
    is_trained = true;
}

void IndexScalarQuantizer4bitUniformCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    // For COSINE metric, normalize vectors before adding
    // Use knowhere's CopyAndNormalizeVecs to avoid modifying input data
    auto normalized_data = ::knowhere::CopyAndNormalizeVecs<float>(x, n, d);

    // Add normalized data
    IndexScalarQuantizer::add(n, normalized_data.get());

    // Calculate and store inverse L2 norms from ORIGINAL vectors (not
    // normalized) This is needed for refine to work correctly with COSINE
    // metric
    const size_t current_size = inverse_l2_norms.size();
    inverse_l2_norms.resize(current_size + n);
    for (idx_t i = 0; i < n; i++) {
        const float l2sqr_norm = fvec_norm_L2sqr(x + i * d, d);
        const float inverse_l2_norm =
                (l2sqr_norm == 0.0f) ? 1.0f : (1.0f / sqrtf(l2sqr_norm));
        inverse_l2_norms[i + current_size] = inverse_l2_norm;
    }
}

DistanceComputer* IndexScalarQuantizer4bitUniformCosine::get_distance_computer()
        const {
    std::unique_ptr<DistanceComputer> base_dc(
            IndexScalarQuantizer::get_distance_computer());

    return new SQ4UniformCosineDistanceComputer(d, std::move(base_dc));
}

const float* IndexScalarQuantizer4bitUniformCosine::get_inverse_l2_norms()
        const {
    // Ensure cache is sized correctly
    if (inverse_l2_norms.size() != static_cast<size_t>(ntotal)) {
        inverse_l2_norms.resize(ntotal, 1.0f);
    }
    return inverse_l2_norms.data();
}

void IndexScalarQuantizer4bitUniformCosine::reset() {
    IndexScalarQuantizer::reset();
    inverse_l2_norms.clear();
}

//////////////////////////////////////////////////////////////////////////////////
// IndexScalarQuantizer4bitUniformIP implementation
//////////////////////////////////////////////////////////////////////////////////

IndexScalarQuantizer4bitUniformIP::IndexScalarQuantizer4bitUniformIP(int d)
        : IndexScalarQuantizer(
                  d,
                  ScalarQuantizer::QT_4bit_uniform,
                  METRIC_INNER_PRODUCT) {
    is_cosine = false;
}

IndexScalarQuantizer4bitUniformIP::IndexScalarQuantizer4bitUniformIP()
        : IndexScalarQuantizer() {
    metric_type = METRIC_INNER_PRODUCT;
    is_cosine = false;
}

void IndexScalarQuantizer4bitUniformIP::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    IndexScalarQuantizer::add(n, x);

    // Compute and store norms squared for IP distance computation
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        float norm_sqr = faiss::cppcontrib::knowhere::fvec_norm_L2sqr(vec, d);
        l2_norms_sqr.push_back(norm_sqr > 0 ? norm_sqr : 1.0f);
    }
}

void IndexScalarQuantizer4bitUniformIP::reset() {
    IndexScalarQuantizer::reset();
    l2_norms_sqr.clear();
}

DistanceComputer* IndexScalarQuantizer4bitUniformIP::get_distance_computer()
        const {
    std::unique_ptr<DistanceComputer> base_dc(
            IndexScalarQuantizer::get_distance_computer());

    return new WithSQ4UniformNormIPDistanceComputer(
            get_l2_norms_sqr(), d, std::move(base_dc));
}

const float* IndexScalarQuantizer4bitUniformIP::get_l2_norms_sqr() const {
    return l2_norms_sqr.data();
}

//////////////////////////////////////////////////////////////////////////////////
// IndexHNSWSQ4UniformCosine implementation
//////////////////////////////////////////////////////////////////////////////////

IndexHNSWSQ4UniformCosine::IndexHNSWSQ4UniformCosine() : IndexHNSW() {
    is_cosine = true;
}

IndexHNSWSQ4UniformCosine::IndexHNSWSQ4UniformCosine(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M)
        : IndexHNSW(new IndexScalarQuantizer4bitUniformCosine(d), M) {
    FAISS_THROW_IF_NOT_MSG(
            qtype == ScalarQuantizer::QT_4bit_uniform,
            "IndexHNSWSQ4UniformCosine only supports QT_4bit_uniform");

    is_trained = this->storage->is_trained;
    own_fields = true;
    is_cosine = true;
}

//////////////////////////////////////////////////////////////////////////////////
// IndexHNSWSQ4UniformIP implementation
//////////////////////////////////////////////////////////////////////////////////

IndexHNSWSQ4UniformIP::IndexHNSWSQ4UniformIP() : IndexHNSW() {
    is_cosine = false;
}

IndexHNSWSQ4UniformIP::IndexHNSWSQ4UniformIP(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M)
        : IndexHNSW(new IndexScalarQuantizer4bitUniformIP(d), M) {
    FAISS_THROW_IF_NOT_MSG(
            qtype == ScalarQuantizer::QT_4bit_uniform,
            "IndexHNSWSQ4UniformIP only supports QT_4bit_uniform");

    is_trained = this->storage->is_trained;
    own_fields = true;
    is_cosine = false;
}

}


