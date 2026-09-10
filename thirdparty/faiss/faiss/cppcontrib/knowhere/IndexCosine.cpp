// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <faiss/cppcontrib/knowhere/IndexCosine.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>

#include <cstring>

#include "simd/hook.h"

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/cppcontrib/knowhere/utils/distances.h>
#include <faiss/utils/prefetch.h>

#include "knowhere/utils.h"



namespace faiss::cppcontrib::knowhere {

//////////////////////////////////////////////////////////////////////////////////

//
struct FlatCosineDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    const float* inverse_l2_norms;
    float inverse_query_norm = 0;

    const int8_t* routing_sq8_codes = nullptr;
    const float* routing_sq8_scales = nullptr;
    std::vector<int8_t> routing_query;
    float routing_query_scale = 1.0f;

    void set_routing_query(const float* x) {
        if (x == nullptr || routing_sq8_codes == nullptr) {
            routing_query.clear();
            routing_query_scale = 1.0f;
            return;
        }
        routing_query.resize(d);
        float max_abs = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            max_abs = std::max(max_abs, std::abs(x[j]));
        }
        routing_query_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inverse_scale = 1.0f / routing_query_scale;
        for (size_t j = 0; j < d; ++j) {
            const long quantized = std::lrintf(x[j] * inverse_scale);
            routing_query[j] = static_cast<int8_t>(
                    std::clamp(quantized, -127L, 127L));
        }
    }

    void ensure_routing_query() {
        if (routing_query.empty() && q != nullptr) {
            set_routing_query(q);
        }
    }

    float routing_scale(const idx_t idx) const {
        return routing_sq8_scales[idx] * routing_query_scale *
                inverse_l2_norms[idx] * inverse_query_norm;
    }

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        const float norm = fvec_norm_L2sqr((const float*)code, d);
        return (norm == 0) ? 0 : (fvec_inner_product(q, (const float*)code, d) / sqrtf(norm) * inverse_query_norm);
    }

    float operator()(const idx_t i) final override {
        const float* __restrict y_i =
                reinterpret_cast<const float*>(codes + i * code_size);

        prefetch_L2(inverse_l2_norms + i);

        const float dp0 = fvec_inner_product(q, y_i, d);

        const float inverse_code_norm_i = inverse_l2_norms[i];
        const float distance = dp0 * inverse_code_norm_i * inverse_query_norm;
        return distance;
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        const float* __restrict y_i =
                reinterpret_cast<const float*>(codes + i * code_size);
        const float* __restrict y_j =
                reinterpret_cast<const float*>(codes + j * code_size);

        prefetch_L2(inverse_l2_norms + i);
        prefetch_L2(inverse_l2_norms + j);

        const float dp0 = fvec_inner_product(y_i, y_j, d);

        const float inverse_code_norm_i = inverse_l2_norms[i];
        const float inverse_code_norm_j = inverse_l2_norms[j];

        return dp0 * inverse_code_norm_i * inverse_code_norm_j;
    }

    explicit FlatCosineDis(const IndexFlatCosine& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {
        // it is the caller's responsibility to ensure that everything is all right.
        inverse_l2_norms = storage.get_inverse_l2_norms();
        if (!storage.routing_sq8_codes.empty() &&
            storage.routing_sq8_scales.size() == size_t(storage.ntotal)) {
            routing_sq8_codes = storage.routing_sq8_codes.data();
            routing_sq8_scales = storage.routing_sq8_scales.data();
        }

        if (q != nullptr) {
            const float query_l2norm = fvec_norm_L2sqr(q, d);
            inverse_query_norm = (query_l2norm <= 0) ? 1.0f : (1.0f / sqrtf(query_l2norm));
        } else {
            inverse_query_norm = 0;
        }
    }

    void set_query(const float* x) final override {
        q = x;

        if (q != nullptr) {
            const float query_l2norm = fvec_norm_L2sqr(q, d);
            inverse_query_norm = (query_l2norm <= 0) ? 1.0f : (1.0f / sqrtf(query_l2norm));
        } else {
            inverse_query_norm = 0;
        }
        routing_query.clear();
    }

    bool supports_approximate_routing_distance() const final override {
        return d >= 896 && routing_sq8_codes != nullptr;
    }

    float routing_distance(const idx_t idx) final override {
        ndis += 1;
        ensure_routing_query();
        const float dot = int8_vec_inner_product(
                routing_query.data(), routing_sq8_codes + idx * d, d);
        return dot * routing_scale(idx);
    }

    void routing_distances_batch_2(
            const idx_t idx0,
            const idx_t idx1,
            float& dis0,
            float& dis1) final override {
        float ignored2 = 0.0f;
        float ignored3 = 0.0f;
        routing_distances_batch_4(
                idx0, idx1, idx1, idx1, dis0, dis1, ignored2, ignored3);
        ndis -= 2;
    }

    void routing_distances_batch_3(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            float& dis0,
            float& dis1,
            float& dis2) final override {
        float ignored3 = 0.0f;
        routing_distances_batch_4(
                idx0, idx1, idx2, idx2, dis0, dis1, dis2, ignored3);
        ndis -= 1;
    }

    void routing_distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;
        ensure_routing_query();
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        int8_vec_inner_product_batch_4(
                routing_query.data(),
                routing_sq8_codes + idx0 * d,
                routing_sq8_codes + idx1 * d,
                routing_sq8_codes + idx2 * d,
                routing_sq8_codes + idx3 * d,
                d,
                dot0,
                dot1,
                dot2,
                dot3);
        dis0 = dot0 * routing_scale(idx0);
        dis1 = dot1 * routing_scale(idx1);
        dis2 = dot2 * routing_scale(idx2);
        dis3 = dot3 * routing_scale(idx3);
    }

    void distances_batch_2(
            const idx_t idx0,
            const idx_t idx1,
            float& dis0,
            float& dis1) final override {
        ndis += 2;
        const float* y0 = reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* y1 = reinterpret_cast<const float*>(codes + idx1 * code_size);
        prefetch_L2(inverse_l2_norms + idx0);
        prefetch_L2(inverse_l2_norms + idx1);
        fvec_inner_product_batch_2(q, y0, y1, d, dis0, dis1);
        dis0 *= inverse_l2_norms[idx0] * inverse_query_norm;
        dis1 *= inverse_l2_norms[idx1] * inverse_query_norm;
    }

    void distances_batch_3(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            float& dis0,
            float& dis1,
            float& dis2) final override {
        ndis += 3;
        const float* y0 = reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* y1 = reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* y2 = reinterpret_cast<const float*>(codes + idx2 * code_size);
        prefetch_L2(inverse_l2_norms + idx0);
        prefetch_L2(inverse_l2_norms + idx1);
        prefetch_L2(inverse_l2_norms + idx2);
        fvec_inner_product_batch_3(q, y0, y1, y2, d, dis0, dis1, dis2);
        dis0 *= inverse_l2_norms[idx0] * inverse_query_norm;
        dis1 *= inverse_l2_norms[idx1] * inverse_query_norm;
        dis2 *= inverse_l2_norms[idx2] * inverse_query_norm;
    }

    bool supports_tail_distance_batches() const final override {
        return true;
    }

    bool prefers_compact_tail_distance_batches() const final override {
        return true;
    }

    bool supports_distance_batch_8() const final override {
        return d >= 128;
    }

    bool should_prefetch_graph_offsets() const final override {
        return d < 896;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        prefetch_L2(inverse_l2_norms + idx0);
        prefetch_L2(inverse_l2_norms + idx1);
        prefetch_L2(inverse_l2_norms + idx2);
        prefetch_L2(inverse_l2_norms + idx3);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        
        const float inverse_code_norm0 = inverse_l2_norms[idx0];
        const float inverse_code_norm1 = inverse_l2_norms[idx1];
        const float inverse_code_norm2 = inverse_l2_norms[idx2];
        const float inverse_code_norm3 = inverse_l2_norms[idx3];
        
        dis0 = dp0 * inverse_code_norm0 * inverse_query_norm;
        dis1 = dp1 * inverse_code_norm1 * inverse_query_norm;
        dis2 = dp2 * inverse_code_norm2 * inverse_query_norm;
        dis3 = dp3 * inverse_code_norm3 * inverse_query_norm;
    }

    void distances_batch_8(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            const idx_t idx4,
            const idx_t idx5,
            const idx_t idx6,
            const idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) final override {
        ndis += 8;
        const idx_t ids[8] = {
                idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7};
        const float* ys[8];
        for (size_t lane = 0; lane < 8; ++lane) {
            ys[lane] = reinterpret_cast<const float*>(
                    codes + ids[lane] * code_size);
            prefetch_L2(inverse_l2_norms + ids[lane]);
        }
        float dp[8] = {};
        fvec_inner_product_batch_8(
                q,
                ys[0],
                ys[1],
                ys[2],
                ys[3],
                ys[4],
                ys[5],
                ys[6],
                ys[7],
                d,
                dp);
        float* outputs[8] = {
                &dis0, &dis1, &dis2, &dis3, &dis4, &dis5, &dis6, &dis7};
        for (size_t lane = 0; lane < 8; ++lane) {
            *outputs[lane] = dp[lane] * inverse_l2_norms[ids[lane]] *
                    inverse_query_norm;
        }
    }
};


//////////////////////////////////////////////////////////////////////////////////

// initialize in a custom way
WithCosineNormDistanceComputer::WithCosineNormDistanceComputer(
    const float* inverse_l2_norms_, 
    const int d_,
    std::unique_ptr<DistanceComputer>&& basedis_) :
basedis(std::move(basedis_)), inverse_l2_norms{inverse_l2_norms_}, d{d_} {} 

// the query remains untouched. It is a caller's responsibility
//   to normalize it.
void WithCosineNormDistanceComputer::set_query(const float* x) {
    basedis->set_query(x);

    if (x != nullptr) {
        const float query_l2norm = faiss::cppcontrib::knowhere::fvec_norm_L2sqr(x, d);
        inverse_query_norm = (query_l2norm <= 0) ? 1.0f : (1.0f / sqrtf(query_l2norm));
    } else {
        inverse_query_norm = 0;
    }
}

/// compute distance of vector i to current query
float WithCosineNormDistanceComputer::operator()(idx_t i) {
    prefetch_L2(inverse_l2_norms + i);

    float dis = (*basedis)(i);
    dis *= inverse_l2_norms[i] * inverse_query_norm;

    return dis;
}

void WithCosineNormDistanceComputer::distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    prefetch_L2(inverse_l2_norms + idx0);
    prefetch_L2(inverse_l2_norms + idx1);
    prefetch_L2(inverse_l2_norms + idx2);
    prefetch_L2(inverse_l2_norms + idx3);

    basedis->distances_batch_4(
            idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);

    dis0 = dis0 * inverse_l2_norms[idx0] * inverse_query_norm;
    dis1 = dis1 * inverse_l2_norms[idx1] * inverse_query_norm;
    dis2 = dis2 * inverse_l2_norms[idx2] * inverse_query_norm;
    dis3 = dis3 * inverse_l2_norms[idx3] * inverse_query_norm;
}

/// compute distance between two stored vectors
float WithCosineNormDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    prefetch_L2(inverse_l2_norms + i);
    prefetch_L2(inverse_l2_norms + j);

    float v = basedis->symmetric_dis(i, j);
    v *= inverse_l2_norms[i];
    v *= inverse_l2_norms[j];
    return v;
}


//////////////////////////////////////////////////////////////////////////////////

L2NormsStorage L2NormsStorage::from_l2_norms(const std::vector<float>& l2_norms) {
    L2NormsStorage result;
    result.add_l2_norms(l2_norms.data(), l2_norms.size());
    return result;
}

void L2NormsStorage::add(const float* x, const idx_t n, const idx_t d) {
    const size_t current_size = inverse_l2_norms.size();
    inverse_l2_norms.resize(current_size + n);

    for (idx_t i = 0; i < n; i++) {
        const float l2sqr_norm = fvec_norm_L2sqr(x + i * d, d);
        const float inverse_l2_norm = (l2sqr_norm == 0.0f) ? 1.0f : (1.0f / sqrtf(l2sqr_norm)); 
        inverse_l2_norms[i + current_size] = inverse_l2_norm;
    }
}

void L2NormsStorage::add_l2_norms(const float* l2_norms, const idx_t n) {
    const size_t current_size = inverse_l2_norms.size();
    inverse_l2_norms.resize(current_size + n);
    for (idx_t i = 0; i < n; i++) {
        const float l2sqr_norm = l2_norms[i];
        const float inverse_l2_norm = (l2sqr_norm == 0.0f) ? 1.0f : (1.0f / l2sqr_norm); 
        inverse_l2_norms[i + current_size] = inverse_l2_norm;
    }
}

void L2NormsStorage::reset() {
    inverse_l2_norms.clear();
}

std::vector<float> L2NormsStorage::as_l2_norms() const {
    std::vector<float> result(inverse_l2_norms.size());
    for (size_t i = 0; i < inverse_l2_norms.size(); i++) {
        result[i] = 1.0f / inverse_l2_norms[i];
    }

    return result;
}


//////////////////////////////////////////////////////////////////////////////////

//
IndexFlatCosine::IndexFlatCosine() : IndexFlat() {
    metric_type = MetricType::METRIC_INNER_PRODUCT;
}

//
IndexFlatCosine::IndexFlatCosine(idx_t d) : IndexFlat(d, MetricType::METRIC_INNER_PRODUCT) {
}

void IndexFlatCosine::add_routing_sq8(idx_t n, const float* x) {
    if (d < 896 || n <= 0) {
        return;
    }
    const size_t old_count = routing_sq8_scales.size();
    routing_sq8_scales.resize(old_count + n);
    routing_sq8_codes.resize((old_count + n) * d);
    for (idx_t row = 0; row < n; ++row) {
        const float* vector = x + row * d;
        float max_abs = 0.0f;
        for (idx_t j = 0; j < d; ++j) {
            max_abs = std::max(max_abs, std::abs(vector[j]));
        }
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        routing_sq8_scales[old_count + row] = scale;
        const float inverse_scale = 1.0f / scale;
        int8_t* output = routing_sq8_codes.data() + (old_count + row) * d;
        for (idx_t j = 0; j < d; ++j) {
            const long quantized = std::lrintf(vector[j] * inverse_scale);
            output[j] = static_cast<int8_t>(
                    std::clamp(quantized, -127L, 127L));
        }
    }
}

void IndexFlatCosine::rebuild_routing_sq8() {
    routing_sq8_codes.clear();
    routing_sq8_scales.clear();
    if (d >= 896 && ntotal > 0) {
        add_routing_sq8(ntotal, get_xb());
    }
}

//
void IndexFlatCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    // Store inverse L2 norms (for distance computation)
    inverse_norms_storage.add(x, n, d);
    add_routing_sq8(n, x);

    // Add original vectors to the base index
    IndexFlat::add(n, x);
}

void IndexFlatCosine::reset() {
    IndexFlat::reset();
    inverse_norms_storage.reset();
    routing_sq8_codes.clear();
    routing_sq8_scales.clear();
}

void IndexFlatCosine::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT(k > 0);
    float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
    knn_cosine(x, get_xb(), inverse_norms_storage.inverse_l2_norms.data(), d, n, ntotal, &res, sel);
}

void IndexFlatCosine::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;
    range_search_cosine(x, get_xb(), inverse_norms_storage.inverse_l2_norms.data(), d, n, ntotal, radius, result, sel);
}

const float* IndexFlatCosine::get_inverse_l2_norms() const {
    return inverse_norms_storage.inverse_l2_norms.data();
}

//
FlatCodesDistanceComputer* IndexFlatCosine::get_FlatCodesDistanceComputer() const {
    return new FlatCosineDis(*this);
}


//////////////////////////////////////////////////////////////////////////////////

IndexScalarQuantizerCosine::IndexScalarQuantizerCosine(
        int d,
        ::faiss::ScalarQuantizer::QuantizerType qtype)
        : ::faiss::IndexScalarQuantizer(d, qtype, MetricType::METRIC_INNER_PRODUCT) {
}

IndexScalarQuantizerCosine::IndexScalarQuantizerCosine()
        : ::faiss::IndexScalarQuantizer() {
    metric_type = MetricType::METRIC_INNER_PRODUCT;
}

void IndexScalarQuantizerCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    ::faiss::IndexScalarQuantizer::add(n, x);
    inverse_norms_storage.add(x, n, d);
}

void IndexScalarQuantizerCosine::reset() {
    ::faiss::IndexScalarQuantizer::reset();
    inverse_norms_storage.reset();
}

const float* IndexScalarQuantizerCosine::get_inverse_l2_norms() const {
    return inverse_norms_storage.inverse_l2_norms.data();
}

DistanceComputer* IndexScalarQuantizerCosine::get_distance_computer() const {
    return new WithCosineNormDistanceComputer(
        this->get_inverse_l2_norms(),
        this->d,
        std::unique_ptr<faiss::DistanceComputer>(
            ::faiss::IndexScalarQuantizer::get_FlatCodesDistanceComputer())
    );
}


//////////////////////////////////////////////////////////////////////////////////

//
IndexPQCosine::IndexPQCosine(int d, size_t M, size_t nbits) : 
    IndexPQ(d, M, nbits, MetricType::METRIC_INNER_PRODUCT) {
}

IndexPQCosine::IndexPQCosine() : IndexPQ() {
    metric_type = MetricType::METRIC_INNER_PRODUCT;
} 

void IndexPQCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    IndexPQ::add(n, x);
    inverse_norms_storage.add(x, n, d);
}

void IndexPQCosine::reset() {
    IndexPQ::reset();
    inverse_norms_storage.reset();
}

const float* IndexPQCosine::get_inverse_l2_norms() const {
    return inverse_norms_storage.inverse_l2_norms.data();
}

DistanceComputer* IndexPQCosine::get_distance_computer() const {
    return new WithCosineNormDistanceComputer(
        this->get_inverse_l2_norms(),
        this->d,
        std::unique_ptr<faiss::DistanceComputer>(IndexPQ::get_FlatCodesDistanceComputer())
    );
}


//////////////////////////////////////////////////////////////////////////////////

IndexProductResidualQuantizerCosine::IndexProductResidualQuantizerCosine(
        int d,
        size_t nsplits,
        size_t Msub,
        size_t nbits,
        AdditiveQuantizer::Search_type_t search_type) :
    IndexProductResidualQuantizer(d, nsplits, Msub, nbits, MetricType::METRIC_INNER_PRODUCT, search_type) {
}        


IndexProductResidualQuantizerCosine::IndexProductResidualQuantizerCosine() :
    IndexProductResidualQuantizer() {
    metric_type = MetricType::METRIC_INNER_PRODUCT;
}

void IndexProductResidualQuantizerCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    IndexProductResidualQuantizer::add(n, x);
    inverse_norms_storage.add(x, n, d);
}

void IndexProductResidualQuantizerCosine::reset() {
    IndexProductResidualQuantizer::reset();
    inverse_norms_storage.reset();
}

const float* IndexProductResidualQuantizerCosine::get_inverse_l2_norms() const {
    return inverse_norms_storage.inverse_l2_norms.data();
}

DistanceComputer* IndexProductResidualQuantizerCosine::get_distance_computer() const {
    return new WithCosineNormDistanceComputer(
        this->get_inverse_l2_norms(),
        this->d,
        std::unique_ptr<faiss::DistanceComputer>(IndexProductResidualQuantizer::get_FlatCodesDistanceComputer())
    );
}


//////////////////////////////////////////////////////////////////////////////////

//
IndexHNSWFlatCosine::IndexHNSWFlatCosine() {
    is_trained = true;
}

IndexHNSWFlatCosine::IndexHNSWFlatCosine(int d, int M) :
    IndexHNSW(new IndexFlatCosine(d), M)
{
    own_fields = true;
    is_trained = true;
}

const float* IndexHNSWFlatCosine::get_inverse_l2_norms() const {
    auto* s = dynamic_cast<const HasInverseL2Norms*>(storage);
    return s ? s->get_inverse_l2_norms() : nullptr;
}


//////////////////////////////////////////////////////////////////////////////////

//
IndexHNSWSQCosine::IndexHNSWSQCosine() {
}

IndexHNSWSQCosine::IndexHNSWSQCosine(
        int d,
        ::faiss::ScalarQuantizer::QuantizerType qtype,
        int M) :
    IndexHNSW(new IndexScalarQuantizerCosine(d, qtype), M)
{
    is_trained = this->storage->is_trained;
    own_fields = true;
}

const float* IndexHNSWSQCosine::get_inverse_l2_norms() const {
    auto* s = dynamic_cast<const HasInverseL2Norms*>(storage);
    return s ? s->get_inverse_l2_norms() : nullptr;
}


//
IndexHNSWPQCosine::IndexHNSWPQCosine() {
}

IndexHNSWPQCosine::IndexHNSWPQCosine(int d, size_t pq_M, int M, size_t pq_nbits) :
    IndexHNSW(new IndexPQCosine(d, pq_M, pq_nbits), M)
{
    own_fields = true;
}

const float* IndexHNSWPQCosine::get_inverse_l2_norms() const {
    auto* s = dynamic_cast<const HasInverseL2Norms*>(storage);
    return s ? s->get_inverse_l2_norms() : nullptr;
}

void IndexHNSWPQCosine::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
    (dynamic_cast<IndexPQCosine*>(storage))->pq.compute_sdc_table();
}

//
IndexHNSWProductResidualQuantizer::IndexHNSWProductResidualQuantizer() = default;

IndexHNSWProductResidualQuantizer::IndexHNSWProductResidualQuantizer(
        int d,
        size_t prq_nsplits,
        size_t prq_Msub,
        size_t prq_nbits,
        size_t M,
        MetricType metric,
        AdditiveQuantizer::Search_type_t prq_search_type
) : IndexHNSW(new IndexProductResidualQuantizer(d, prq_nsplits, prq_Msub, prq_nbits, metric, prq_search_type), M) {}

//
IndexHNSWProductResidualQuantizerCosine::IndexHNSWProductResidualQuantizerCosine() {
}

IndexHNSWProductResidualQuantizerCosine::IndexHNSWProductResidualQuantizerCosine(
        int d,
        size_t prq_nsplits,
        size_t prq_Msub,
        size_t prq_nbits,
        size_t M,
        AdditiveQuantizer::Search_type_t prq_search_type
) : IndexHNSW(new IndexProductResidualQuantizerCosine(d, prq_nsplits, prq_Msub, prq_nbits, prq_search_type), M) {
}

const float* IndexHNSWProductResidualQuantizerCosine::get_inverse_l2_norms() const {
    auto* s = dynamic_cast<const HasInverseL2Norms*>(storage);
    return s ? s->get_inverse_l2_norms() : nullptr;
}

}
