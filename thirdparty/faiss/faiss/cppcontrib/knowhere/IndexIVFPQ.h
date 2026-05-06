/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/IndexIVFPQ.h>
#include <faiss/cppcontrib/knowhere/IndexIVF.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// Path-D step 11.5.1: fork IVFPQ machinery collapsed onto baseline.
//
// The following symbols are now type/value aliases of baseline:
//   - IVFPQSearchParameters
//   - IndexIVFPQStats
//   - indexIVFPQ_stats        (extern; baseline owns the global)
//   - precomputed_table_max_bytes
//   - index_ivfpq_add_core_o_bs
//   - initialize_IVFPQ_precomputed_table
//
// Fork-side declarations were byte-identical to baseline's, so the
// `using` collapses are pure simplification with no behavioral change.
using IVFPQSearchParameters = ::faiss::IVFPQSearchParameters;
using IndexIVFPQStats = ::faiss::IndexIVFPQStats;
using ::faiss::initialize_IVFPQ_precomputed_table;

/** Inverted file with Product Quantizer encoding.
 *
 * Path-D step 11.5.1: reparented to inherit from `::faiss::IndexIVFPQ`
 * directly. After audit, every fork IVFPQ method body (encode_vectors,
 * sa_decode, add_core, add_core_o, train_encoder,
 * train_encoder_num_vectors, encode, encode_multiple, decode_multiple,
 * precompute_table, find_duplicates, reconstruct_from_offset) was
 * byte-identical to baseline's same-named method, so they have been
 * deleted; baseline's inherited bodies take over. The default ctor was
 * also byte-identical and is gone.
 *
 * The 4-arg ctor stays (with a different default for `own_invlists`)
 * to preserve fork's convention of installing a fork
 * `ArrayInvertedLists` (NormInvertedLists-capable) by default.
 *
 * The fork scanner override was deleted after the knowhere-only
 * scanner hooks were split from baseline `::faiss::InvertedListScanner`.
 * IVFPQ is not currently an iterator-supported knowhere index type, and
 * non-iterator search paths can use baseline FAISS scanner dispatch
 * directly.
 */
struct IndexIVFPQ : ::faiss::IndexIVFPQ {
    IndexIVFPQ(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            MetricType metric = METRIC_L2);

    // Default ctor body was byte-identical to baseline's; inheriting
    // baseline's default ctor is functionally equivalent.
    IndexIVFPQ() = default;
};

}  // namespace knowhere
}  // namespace cppcontrib
}  // namespace faiss
