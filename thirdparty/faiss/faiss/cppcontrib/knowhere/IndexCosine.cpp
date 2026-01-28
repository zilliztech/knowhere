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

#include <faiss/cppcontrib/knowhere/FaissHook.h>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/prefetch.h>



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
    is_cosine = true;
}

//
IndexFlatCosine::IndexFlatCosine(idx_t d) : IndexFlat(d, MetricType::METRIC_INNER_PRODUCT, true) {}

//
void IndexFlatCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    // todo aguzhva:
    // it is a tricky situation at this moment, because IndexFlatCosine
    //   contains duplicate norms (one in IndexFlatCodes and one in HasInverseL2Norms).
    //   Norms in IndexFlatCodes are going to be removed in the future.
    IndexFlat::add(n, x);
    inverse_norms_storage.add(x, n, d);
}

void IndexFlatCosine::reset() {
    IndexFlat::reset();
    inverse_norms_storage.reset();
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
        ScalarQuantizer::QuantizerType qtype) 
        : IndexScalarQuantizer(d, qtype, MetricType::METRIC_INNER_PRODUCT) {
    is_cosine = true;
}

IndexScalarQuantizerCosine::IndexScalarQuantizerCosine() : IndexScalarQuantizer() {
    metric_type = MetricType::METRIC_INNER_PRODUCT;
    is_cosine = true;
}

void IndexScalarQuantizerCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    // todo aguzhva:
    // it is a tricky situation at this moment, because IndexScalarQuantizerCosine
    //   contains duplicate norms (one in IndexFlatCodes and one in HasInverseL2Norms).
    //   Norms in IndexFlatCodes are going to be removed in the future.
    IndexScalarQuantizer::add(n, x);
    inverse_norms_storage.add(x, n, d);
}

void IndexScalarQuantizerCosine::reset() {
    IndexScalarQuantizer::reset();
    inverse_norms_storage.reset();
}

const float* IndexScalarQuantizerCosine::get_inverse_l2_norms() const {
    return inverse_norms_storage.inverse_l2_norms.data();
}

DistanceComputer* IndexScalarQuantizerCosine::get_distance_computer() const {
    return new WithCosineNormDistanceComputer(
        this->get_inverse_l2_norms(),
        this->d,
        std::unique_ptr<faiss::DistanceComputer>(IndexScalarQuantizer::get_FlatCodesDistanceComputer())
    );
}


//////////////////////////////////////////////////////////////////////////////////

//
IndexPQCosine::IndexPQCosine(int d, size_t M, size_t nbits) : 
    IndexPQ(d, M, nbits, MetricType::METRIC_INNER_PRODUCT) {
    is_cosine = true;
}

IndexPQCosine::IndexPQCosine() : IndexPQ() {
    metric_type = MetricType::METRIC_INNER_PRODUCT;
    is_cosine = true;
} 

void IndexPQCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    // todo aguzhva:
    // it is a tricky situation at this moment, because IndexPQCosine
    //   contains duplicate norms (one in IndexFlatCodes and one in HasInverseL2Norms).
    //   Norms in IndexFlatCodes are going to be removed in the future.
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
    is_cosine = true;
}        


IndexProductResidualQuantizerCosine::IndexProductResidualQuantizerCosine() :
    IndexProductResidualQuantizer() {
    metric_type = MetricType::METRIC_INNER_PRODUCT;
    is_cosine = true;
}

void IndexProductResidualQuantizerCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    // todo aguzhva:
    // it is a tricky situation at this moment, because IndexProductResidualQuantizerCosine
    //   contains duplicate norms (one in IndexFlatCodes and one in HasInverseL2Norms).
    //   Norms in IndexFlatCodes are going to be removed in the future.
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
    is_cosine = true;
}

IndexHNSWFlatCosine::IndexHNSWFlatCosine(int d, int M) :
    IndexHNSW(new IndexFlatCosine(d), M) 
{
    own_fields = true;
    is_trained = true;
    is_cosine = true;
}


//////////////////////////////////////////////////////////////////////////////////

//
IndexHNSWSQCosine::IndexHNSWSQCosine() {
    is_cosine = true;    
}

IndexHNSWSQCosine::IndexHNSWSQCosine(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M) : 
    IndexHNSW(new IndexScalarQuantizerCosine(d, qtype), M) 
{
    is_trained = this->storage->is_trained;
    own_fields = true;
    is_cosine = true;
}


//
IndexHNSWPQCosine::IndexHNSWPQCosine() {
    is_cosine = true;
}

IndexHNSWPQCosine::IndexHNSWPQCosine(int d, size_t pq_M, int M, size_t pq_nbits) :
    IndexHNSW(new IndexPQCosine(d, pq_M, pq_nbits), M) 
{
    own_fields = true;
    is_cosine = true;
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
    is_cosine = true;    
}

IndexHNSWProductResidualQuantizerCosine::IndexHNSWProductResidualQuantizerCosine(
        int d,
        size_t prq_nsplits,
        size_t prq_Msub,
        size_t prq_nbits,
        size_t M,
        AdditiveQuantizer::Search_type_t prq_search_type
) : IndexHNSW(new IndexHNSWProductResidualQuantizerCosine(d, prq_nsplits, prq_Msub, prq_nbits, prq_search_type), M) {
    is_cosine = true;
}

}



