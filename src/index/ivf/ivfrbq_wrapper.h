#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "faiss/Index.h"
#include "faiss/IndexIVFRaBitQ.h"

namespace knowhere {

// This is wrapper is needed, bcz we use faiss::IndexPreTransform
//   for wrapping faiss::IndexIVFRaBitQ. The problem is that
//   IndexPreTransform is a generic class, suitable for any other
//   use case as well, so this is wrong to reference IndexPreTransform
//   in the ivf.cc file.
struct IndexIVFRaBitQWrapper : faiss::Index {
    std::unique_ptr<faiss::Index> index;

    explicit IndexIVFRaBitQWrapper(std::unique_ptr<faiss::Index>&& index_in);

    virtual ~IndexIVFRaBitQWrapper();

    void
    train(faiss::idx_t n, const float* x) override;

    void
    add(faiss::idx_t n, const float* x) override;

    void
    search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
           const faiss::SearchParameters* params) const override;

    void
    range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                 const faiss::SearchParameters* params) const override;

    void
    reset() override;

    void
    merge_from(Index& otherIndex, faiss::idx_t add_id) override;

    faiss::DistanceComputer*
    get_distance_computer() const override;

    // point to IndexIVFRaBitQ or return nullptr
    faiss::IndexIVFRaBitQ*
    get_ivfrabitq_index();
    const faiss::IndexIVFRaBitQ*
    get_ivfrabitq_index() const;

    // return the size of the index
    size_t
    size() const;
};

}  // namespace knowhere
