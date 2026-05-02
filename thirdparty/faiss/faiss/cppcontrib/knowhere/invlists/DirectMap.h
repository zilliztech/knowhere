/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

// Path-D step 11.1: fork DirectMap collapsed to baseline alias.
//
// After step 10.9 removed the fork-only `ConcurrentArray` variant and
// step 10.14c widened the fork DirectMap API to take baseline
// ::faiss::InvertedLists*, fork and baseline DirectMap became
// structurally identical. Step 10.14c also reparented fork
// BlockInvertedLists onto baseline, making baseline DirectMap::remove_ids'
// BlockInvertedLists shortcut (which dynamic_casts to baseline
// BlockInvertedLists*) fire correctly on fork objects — so aliasing
// here loses nothing.
//
// The `using` lines expose the baseline names in the knowhere
// namespace so fork code's unqualified references (`DirectMap`,
// `DirectMapAdd`, `lo_build`, `lo_listno`, `lo_offset`) keep
// compiling without source changes.

#include <faiss/invlists/DirectMap.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

using DirectMap = ::faiss::DirectMap;
using DirectMapAdd = ::faiss::DirectMapAdd;
using ::faiss::lo_build;
using ::faiss::lo_listno;
using ::faiss::lo_offset;

}
}
} // namespace faiss
