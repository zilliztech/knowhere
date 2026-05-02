/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/invlists/BlockInvertedLists.h>

#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

namespace faiss::cppcontrib::knowhere {

// Path-D step 10.14c: fork BlockInvertedLists reparented to
// ::faiss::BlockInvertedLists. All state (nlist, code_size, n_per_block,
// block_size, packer, codes, ids) and all methods (list_size, get_codes,
// get_ids, add_entries, update_entries, resize, remove_ids, dtor) are
// inherited from baseline unchanged. The ctor bodies are the only
// fork-local code left — they just forward to baseline.

BlockInvertedLists::BlockInvertedLists(
        size_t nlist,
        size_t n_per_block,
        size_t block_size)
        : ::faiss::BlockInvertedLists(nlist, n_per_block, block_size) {}

BlockInvertedLists::BlockInvertedLists(size_t nlist, const CodePacker* packer)
        : ::faiss::BlockInvertedLists(nlist, packer) {}

BlockInvertedLists::BlockInvertedLists()
        : ::faiss::BlockInvertedLists() {}

/**************************************************
 * IO hook implementation
 **************************************************/

BlockInvertedListsIOHook::BlockInvertedListsIOHook()
        : InvertedListsIOHook("ilbl", typeid(BlockInvertedLists).name()) {}

void BlockInvertedListsIOHook::write(
        const ::faiss::InvertedLists* ils_in, IOWriter* f) const {
    uint32_t h = fourcc("ilbl");
    WRITE1(h);
    const ::faiss::BlockInvertedLists* il =
            dynamic_cast<const ::faiss::BlockInvertedLists*>(ils_in);
    FAISS_THROW_IF_NOT(il);
    WRITE1(il->nlist);
    WRITE1(il->code_size);
    WRITE1(il->n_per_block);
    WRITE1(il->block_size);

    for (size_t i = 0; i < il->nlist; i++) {
        WRITEVECTOR(il->ids[i]);
        WRITEVECTOR(il->codes[i]);
    }
}

::faiss::InvertedLists* BlockInvertedListsIOHook::read(
        IOReader* f, int /* io_flags */) const {
    BlockInvertedLists* il = new BlockInvertedLists();
    READ1(il->nlist);
    READ1(il->code_size);
    READ1(il->n_per_block);
    READ1(il->block_size);

    il->ids.resize(il->nlist);
    il->codes.resize(il->nlist);

    for (size_t i = 0; i < il->nlist; i++) {
        READVECTOR(il->ids[i]);
        READVECTOR(il->codes[i]);
    }

    return il;
}

}
