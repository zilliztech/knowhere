/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizerOp.h>

namespace faiss {

/**
 * The uniform quantizer has a range [vmin, vmax]. The range can be
 * the same for all dimensions (uniform) or specific per dimension
 * (default).
 */

struct ScalarQuantizer {
    QuantizerType qtype;

    RangeStat rangestat;
    float rangestat_arg;

    /// dimension of input vectors
    size_t d;

    /// bits per scalar code
    size_t bits;

    /// bytes per vector
    size_t code_size;

    /// trained values (including the range)
    std::vector<float> trained;

    ScalarQuantizer(size_t d, QuantizerType qtype);
    ScalarQuantizer();

    /// updates internal values based on qtype and d
    void set_derived_sizes();

    void train(size_t n, const float* x);

    /// Used by an IVF index to train based on the residuals
    void train_residual(
            size_t n,
            const float* x,
            Index* quantizer,
            bool by_residual,
            bool verbose);

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     */
    void compute_codes(const float* x, uint8_t* codes, size_t n) const;

    /** Decode a set of vectors
     *
     * @param codes  codes to decode, size n * code_size
     * @param x      output vectors, size n * d
     */
    void decode(const uint8_t* code, float* x, size_t n) const;

    /*****************************************************
     * Objects that provide methods for encoding/decoding, distance
     * computation and inverted list scanning
     *****************************************************/

    Quantizer* select_quantizer() const;

    SQDistanceComputer* get_distance_computer(
            MetricType metric = METRIC_L2) const;

    InvertedListScanner* select_InvertedListScanner(
            MetricType mt,
            const Index* quantizer,
            bool store_pairs,
            bool by_residual = false) const;

    size_t cal_size() {
        return sizeof(*this) + trained.size() * sizeof(float);
    }
};

/*******************************************************************
 * IndexScalarQuantizer/IndexIVFScalarQuantizer scanner object
 *
 * It is an InvertedListScanner, but is designed to work with
 * IndexScalarQuantizer as well.
 ********************************************************************/

template <class DCClass>
struct IVFSQScannerIP : InvertedListScanner {
    DCClass dc;
    bool by_residual;

    float accu0; /// added to all distances

    IVFSQScannerIP(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            bool store_pairs,
            bool by_residual)
            : dc(d, trained), by_residual(by_residual), accu0(0) {
        this->store_pairs = store_pairs;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        dc.set_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        accu0 = by_residual ? coarse_dis : 0;
    }

    float distance_to_code(const uint8_t* code) const final {
        return accu0 + dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k,
            const BitsetView bitset = nullptr) const override {
        size_t nup = 0;

        for (size_t j = 0; j < list_size; j++) {
            if (bitset.empty() || !bitset.test(ids[j])) {
                float accu = accu0 + dc.query_to_code(codes);
                if (accu > simi[0]) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    minheap_replace_top(k, simi, idxi, accu, id);
                    nup++;
                }
            }
            codes += code_size;
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res,
            const BitsetView bitset = nullptr) const override {
        for (size_t j = 0; j < list_size; j++) {
            if (bitset.empty() || !bitset.test(ids[j])) {
                float accu = accu0 + dc.query_to_code(codes);
                if (accu > radius) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    res.add(accu, id);
                }
            }
            codes += code_size;
        }
    }
};

template <class DCClass>
struct IVFSQScannerL2 : InvertedListScanner {
    DCClass dc;

    bool by_residual;
    const Index* quantizer;
    const float* x; /// current query

    std::vector<float> tmp;

    IVFSQScannerL2(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            const Index* quantizer,
            bool store_pairs,
            bool by_residual)
            : dc(d, trained),
              by_residual(by_residual),
              quantizer(quantizer),
              x(nullptr),
              tmp(d) {
        this->store_pairs = store_pairs;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        x = query;
        if (!quantizer) {
            dc.set_query(query);
        }
    }

    void set_list(idx_t list_no, float /*coarse_dis*/) override {
        this->list_no = list_no;
        if (by_residual) {
            // shift of x_in wrt centroid
            quantizer->compute_residual(x, tmp.data(), list_no);
            dc.set_query(tmp.data());
        } else {
            dc.set_query(x);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k,
            const BitsetView bitset = nullptr) const override {
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            if (bitset.empty() || !bitset.test(ids[j])) {
                float dis = dc.query_to_code(codes);
                if (dis < simi[0]) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    maxheap_replace_top(k, simi, idxi, dis, id);
                    nup++;
                }
            }
            codes += code_size;
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res,
            const BitsetView bitset = nullptr) const override {
        for (size_t j = 0; j < list_size; j++) {
            if (bitset.empty() || !bitset.test(ids[j])) {
                float dis = dc.query_to_code(codes);
                if (dis < radius) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    res.add(dis, id);
                }
            }
            codes += code_size;
        }
    }
};

} // namespace faiss
