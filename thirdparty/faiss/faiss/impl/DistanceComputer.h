/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/utils/prefetch.h>

namespace faiss {

/***********************************************************
 * The distance computer maintains a current query and computes
 * distances to elements in an index that supports random access.
 *
 * The DistanceComputer is not intended to be thread-safe (eg. because
 * it maintains counters) so the distance functions are not const,
 * instantiate one from each thread if needed.
 *
 * Note that the equivalent for IVF indexes is the InvertedListScanner,
 * that has additional methods to handle the inverted list context.
 ***********************************************************/
struct DistanceComputer {
    /// called before computing distances. Pointer x should remain valid
    /// while operator () is called
    virtual void set_query(const float* x) = 0;

    /// compute distance of vector i to current query
    virtual float operator()(idx_t i) = 0;

    /// Approximate distances used only to order graph-routing nodes that
    /// cannot enter the result set. Implementations must keep the score on
    /// the same scale as operator().
    virtual float routing_distance(idx_t i) {
        return this->operator()(i);
    }

    virtual void routing_distances_batch_2(
            idx_t idx0, idx_t idx1, float& dis0, float& dis1) {
        dis0 = routing_distance(idx0);
        dis1 = routing_distance(idx1);
    }

    virtual void routing_distances_batch_3(
            idx_t idx0,
            idx_t idx1,
            idx_t idx2,
            float& dis0,
            float& dis1,
            float& dis2) {
        dis0 = routing_distance(idx0);
        dis1 = routing_distance(idx1);
        dis2 = routing_distance(idx2);
    }

    virtual void routing_distances_batch_4(
            idx_t idx0,
            idx_t idx1,
            idx_t idx2,
            idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) {
        dis0 = routing_distance(idx0);
        dis1 = routing_distance(idx1);
        dis2 = routing_distance(idx2);
        dis3 = routing_distance(idx3);
    }

    virtual bool supports_approximate_routing_distance() const {
        return false;
    }

    virtual void distances_batch_2(
            const idx_t idx0,
            const idx_t idx1,
            float& dis0,
            float& dis1) {
        dis0 = this->operator()(idx0);
        dis1 = this->operator()(idx1);
    }

    virtual void distances_batch_3(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            float& dis0,
            float& dis1,
            float& dis2) {
        dis0 = this->operator()(idx0);
        dis1 = this->operator()(idx1);
        dis2 = this->operator()(idx2);
    }

    /// compute distances of current query to 4 stored vectors.
    /// certain DistanceComputer implementations may benefit
    /// heavily from this.
    virtual void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) {
        // compute first, assign next
        const float d0 = this->operator()(idx0);
        const float d1 = this->operator()(idx1);
        const float d2 = this->operator()(idx2);
        const float d3 = this->operator()(idx3);
        dis0 = d0;
        dis1 = d1;
        dis2 = d2;
        dis3 = d3;
    }

    virtual void distances_batch_8(
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
            float& dis7) {
        distances_batch_4(
                idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
        distances_batch_4(
                idx4, idx5, idx6, idx7, dis4, dis5, dis6, dis7);
    }

    /// Start fetching four stored vectors before their distance batch is
    /// evaluated. Non-flat distance computers can keep the default no-op.
    virtual void
    prefetch_batch_4(const idx_t, const idx_t, const idx_t, const idx_t) {
    }

    virtual bool
    should_pipeline_distance_batches(const float) const {
        return false;
    }

    virtual bool
    supports_tail_distance_batches() const {
        return false;
    }

    virtual bool
    supports_distance_batch_8() const {
        return false;
    }

    /// Graph-offset prefetch only helps when distance evaluation is short
    /// enough that the random adjacency lookup remains exposed.
    virtual bool
    should_prefetch_graph_offsets() const {
        return false;
    }

    /// compute distance between two stored vectors
    virtual float symmetric_dis(idx_t i, idx_t j) = 0;

    // Append capability slots after the established distance ABI. Compact
    // graph loops should only batch short tails when the representation
    // amortizes per-vector work (for example cosine norm handling).
    virtual bool
    prefers_compact_tail_distance_batches() const {
        return false;
    }

    virtual ~DistanceComputer() {}
};

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCT search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis_)
            : basedis(basedis_) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    float routing_distance(idx_t i) override {
        return -basedis->routing_distance(i);
    }

    void routing_distances_batch_2(
            idx_t idx0, idx_t idx1, float& dis0, float& dis1) override {
        basedis->routing_distances_batch_2(idx0, idx1, dis0, dis1);
        dis0 = -dis0;
        dis1 = -dis1;
    }

    void routing_distances_batch_3(
            idx_t idx0,
            idx_t idx1,
            idx_t idx2,
            float& dis0,
            float& dis1,
            float& dis2) override {
        basedis->routing_distances_batch_3(
                idx0, idx1, idx2, dis0, dis1, dis2);
        dis0 = -dis0;
        dis1 = -dis1;
        dis2 = -dis2;
    }

    void routing_distances_batch_4(
            idx_t idx0,
            idx_t idx1,
            idx_t idx2,
            idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        basedis->routing_distances_batch_4(
                idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
        dis0 = -dis0;
        dis1 = -dis1;
        dis2 = -dis2;
        dis3 = -dis3;
    }

    bool supports_approximate_routing_distance() const override {
        return basedis->supports_approximate_routing_distance();
    }

    void distances_batch_2(
            const idx_t idx0,
            const idx_t idx1,
            float& dis0,
            float& dis1) override {
        basedis->distances_batch_2(idx0, idx1, dis0, dis1);
        dis0 = -dis0;
        dis1 = -dis1;
    }

    void distances_batch_3(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            float& dis0,
            float& dis1,
            float& dis2) override {
        basedis->distances_batch_3(idx0, idx1, idx2, dis0, dis1, dis2);
        dis0 = -dis0;
        dis1 = -dis1;
        dis2 = -dis2;
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        basedis->distances_batch_4(
                idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
        dis0 = -dis0;
        dis1 = -dis1;
        dis2 = -dis2;
        dis3 = -dis3;
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
            float& dis7) override {
        basedis->distances_batch_8(
                idx0,
                idx1,
                idx2,
                idx3,
                idx4,
                idx5,
                idx6,
                idx7,
                dis0,
                dis1,
                dis2,
                dis3,
                dis4,
                dis5,
                dis6,
                dis7);
        dis0 = -dis0;
        dis1 = -dis1;
        dis2 = -dis2;
        dis3 = -dis3;
        dis4 = -dis4;
        dis5 = -dis5;
        dis6 = -dis6;
        dis7 = -dis7;
    }

    void
    prefetch_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3) override {
        basedis->prefetch_batch_4(idx0, idx1, idx2, idx3);
    }

    bool
    should_pipeline_distance_batches(const float routing_alpha) const override {
        return basedis->should_pipeline_distance_batches(routing_alpha);
    }

    bool
    supports_tail_distance_batches() const override {
        return basedis->supports_tail_distance_batches();
    }

    bool
    prefers_compact_tail_distance_batches() const override {
        return basedis->prefers_compact_tail_distance_batches();
    }

    bool
    supports_distance_batch_8() const override {
        return basedis->supports_distance_batch_8();
    }

    bool
    should_prefetch_graph_offsets() const override {
        return basedis->should_prefetch_graph_offsets();
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() override {
        delete basedis;
    }
};

/*************************************************************
 * Specialized version of the DistanceComputer when we know that codes are
 * laid out in a flat index.
 */
struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;
    size_t code_size;

    const float* q = nullptr; // not used in all distance computers

    FlatCodesDistanceComputer(
            const uint8_t* codes_,
            size_t code_size_,
            const float* q_ = nullptr)
            : codes(codes_), code_size(code_size_), q(q_) {}

    explicit FlatCodesDistanceComputer(const float* q_)
            : codes(nullptr), code_size(0), q(q_) {}

    FlatCodesDistanceComputer() : codes(nullptr), code_size(0), q(nullptr) {}

    float operator()(idx_t i) override {
        return distance_to_code(codes + i * code_size);
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        distance_to_code_batch_4(
                codes + idx0 * code_size,
                codes + idx1 * code_size,
                codes + idx2 * code_size,
                codes + idx3 * code_size,
                dis0,
                dis1,
                dis2,
                dis3);
    }

    void
    prefetch_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3) override {
        prefetch_L2(codes + idx0 * code_size);
        prefetch_L2(codes + idx1 * code_size);
        prefetch_L2(codes + idx2 * code_size);
        prefetch_L2(codes + idx3 * code_size);
    }

    bool
    should_pipeline_distance_batches(const float routing_alpha) const override {
        // For short and medium vectors, software prefetch can hide a useful
        // fraction of the first cache-line miss. Long vectors already form
        // four hardware-prefetched streams, so extra buffering is useful only
        // while the filter still admits distance batches densely.
        return code_size <= 1024 || routing_alpha <= 0.35f;
    }

    /// Computes a partial dot product over a slice of the query vector.
    /// The slice is defined by the following parameters:
    ///   — `offset`: the starting index of the first component to include
    ///   — `num_components`: the number of consecutive components to include
    ///
    /// Components refer to raw dimensions of the flat (uncompressed) query
    /// vector.
    ///
    /// By default, this method throws an error, as it is only implemented
    /// in specific subclasses such as `FlatL2Dis`. Other flat distance
    /// computers may override this when partial dot product support is needed.
    ///
    /// Over time, this method might be changed to a pure virtual function (`=
    /// 0`) to enforce implementation in subclasses that require this
    /// functionality.
    ///
    /// This method is not part of the generic `DistanceComputer` interface
    /// because for compressed representations (e.g., product quantization),
    /// calling `partial_dot_product` repeatedly is often less efficient than
    /// computing the full distance at once.
    ///
    /// Supporting efficient partial scans generally requires a different memory
    /// layout, such as interleaved blocks that keep SIMD lanes full. This is a
    /// non-trivial change and not supported in the current flat layout.
    ///
    /// For more details on partial (or chunked) dot product computations and
    /// the performance trade-offs involved, refer to the Panorama paper:
    /// https://arxiv.org/pdf/2510.00566
    virtual float partial_dot_product(
            const idx_t /* i */,
            const uint32_t /* offset */,
            const uint32_t /* num_components */) {
        FAISS_THROW_MSG("partial_dot_product not implemented");
    }

    /// compute distance of current query to an encoded vector
    virtual float distance_to_code(const uint8_t* code) = 0;
    virtual void distance_to_code_batch_4(
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            const uint8_t* c4,
            float& d1,
            float& d2,
            float& d3,
            float& d4) {
        d1 = distance_to_code(c1);
        d2 = distance_to_code(c2);
        d3 = distance_to_code(c3);
        d4 = distance_to_code(c4);
    }

    /// Compute partial dot products of current query to 4 stored vectors.
    /// See `partial_dot_product` for more details.
    virtual void partial_dot_product_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dp0,
            float& dp1,
            float& dp2,
            float& dp3,
            const uint32_t offset,
            const uint32_t num_components) {
        // default implementation for correctness
        const float d0 =
                this->partial_dot_product(idx0, offset, num_components);
        const float d1 =
                this->partial_dot_product(idx1, offset, num_components);
        const float d2 =
                this->partial_dot_product(idx2, offset, num_components);
        const float d3 =
                this->partial_dot_product(idx3, offset, num_components);
        dp0 = d0;
        dp1 = d1;
        dp2 = d2;
        dp3 = d3;
    }

    virtual ~FlatCodesDistanceComputer() override {}
};

} // namespace faiss
