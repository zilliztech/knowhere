/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>

namespace faiss::cppcontrib::knowhere {

// Private Knowhere serialization tag. Upstream Faiss reserves "IHNr" for
// its incompatible direct-build/staged-search IndexHNSWRaBitQ format.
inline constexpr char kHnswRaBitQFourcc[] = "IHRK";
inline constexpr char kHnswRaBitQCosineFourcc[] = "IHRC";
inline constexpr char kRaBitQPreTransformCosineFourcc[] = "IRKC";

/** Random-rotation + RaBitQ storage with Knowhere cosine semantics.
 *
 * Original vectors are quantized without permanently normalizing them. The
 * underlying RaBitQ distance computer estimates inner product; this wrapper
 * applies the stored database inverse norm and the query inverse norm.
 */
struct IndexPreTransformRaBitQCosine : faiss::IndexPreTransform,
                                       HasInverseL2Norms {
    L2NormsStorage inverse_norms_storage;

    IndexPreTransformRaBitQCosine();
    IndexPreTransformRaBitQCosine(
            faiss::VectorTransform* transform,
            faiss::Index* index);

    void add(idx_t n, const float* x) override;
    void reset() override;
    faiss::DistanceComputer* get_distance_computer() const override;
    const float* get_inverse_l2_norms() const override;

    void validate_norms() const;
};

/** HNSW graph backed by a randomly-rotated standalone RaBitQ index.
 *
 * The storage layout is deliberately strict:
 *
 *   IndexPreTransform
 *     -> RandomRotationMatrix
 *     -> faiss::IndexRaBitQ
 *
 * RaBitQ does not implement code-to-code symmetric distances, so this index
 * is immutable after its graph and storage have been assembled. Build the
 * graph with exact storage first, then attach the trained/populated RaBitQ
 * storage to this runtime type.
 */
struct IndexHNSWRaBitQ : IndexHNSW {
    IndexHNSWRaBitQ();

    void add(idx_t n, const float* x) override;

    const faiss::IndexPreTransform* pretransform_index() const;

    const faiss::IndexRaBitQ* rabitq_index() const;

    /** Validate the complete runtime/storage shape and serialized invariants.
     * Throws FaissException on malformed state. */
    void validate_storage() const;
};

/** Cosine runtime marker for HNSW backed by cosine-aware RaBitQ storage. */
struct IndexHNSWRaBitQCosine : IndexHNSWRaBitQ, HasInverseL2Norms {
    IndexHNSWRaBitQCosine();

    const float* get_inverse_l2_norms() const override;
    void validate_cosine_storage() const;
};

} // namespace faiss::cppcontrib::knowhere
