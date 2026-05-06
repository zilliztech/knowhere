/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/index_io.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedListsIOHook.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/** Fork BlockInvertedLists — Path-D step 10.14c reparented directly
 * onto baseline ::faiss::BlockInvertedLists.
 *
 * Adds NO state and NO method overrides beyond baseline. The only
 * reason this class still exists is to be a distinct type (different
 * typeid) so the fork's BlockInvertedListsIOHook can own it through
 * the fork's InvertedListsIOHook registry without colliding with
 * baseline's own hook registration.
 *
 * Consequences:
 *   - dynamic_cast<::faiss::BlockInvertedLists*> on a fork object
 *     now succeeds, which makes baseline DirectMap::remove_ids'
 *     block-invlists shortcut fire correctly (unblocks step 11.1
 *     DirectMap alias).
 *   - baseline's remove_ids body (correctly captured orig_size +
 *     reduction clause) is now inherited, fixing a long-standing fork
 *     bug where nremove was always 0.
 *   - fork BlockInvertedLists is NO LONGER a fork::InvertedLists,
 *     so it cannot be passed to callers expecting fork::InvertedLists*.
 *     The corresponding widening of fork::IndexIVF::replace_invlists to
 *     accept baseline InvertedLists* is paired with this change.
 */
struct BlockInvertedLists : ::faiss::BlockInvertedLists {
    BlockInvertedLists(size_t nlist, size_t vec_per_block, size_t block_size);
    BlockInvertedLists(size_t nlist, const CodePacker* packer);
    BlockInvertedLists();
};

struct BlockInvertedListsIOHook : InvertedListsIOHook {
    BlockInvertedListsIOHook();
    void write(const ::faiss::InvertedLists* ils, IOWriter* f) const override;
    ::faiss::InvertedLists* read(IOReader* f, int io_flags) const override;
};

}
}
} // namespace faiss
