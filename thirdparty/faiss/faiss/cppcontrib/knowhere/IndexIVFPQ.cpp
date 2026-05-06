/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IndexIVFPQ.h>

namespace faiss::cppcontrib::knowhere {

/*****************************************
 * IndexIVFPQ implementation
 ******************************************/

// Fork IVFPQ derives from baseline IVFPQ. The 4-arg ctor is the only
// local body: it forwards to baseline with own_invlists=false so we can
// install the fork ArrayInvertedLists (NormInvertedLists-capable) instead
// of baseline's plain ::faiss::ArrayInvertedLists.
IndexIVFPQ::IndexIVFPQ(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        MetricType metric)
        : ::faiss::IndexIVFPQ(
                  quantizer,
                  d,
                  nlist,
                  M,
                  nbits_per_idx,
                  metric,
                  /*own_invlists=*/false) {
    replace_invlists(
            new ArrayInvertedLists(nlist, code_size, false),
            /*own=*/true);
}

// All other IVFPQ behavior, including get_InvertedListScanner(), is
// inherited from baseline FAISS.

} // namespace faiss::cppcontrib::knowhere
