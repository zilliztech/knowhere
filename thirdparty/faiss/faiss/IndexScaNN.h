#pragma once

#include <faiss/Index.h>
#include <faiss/IndexRefine.h>

namespace faiss {

struct IndexScaNNSearchParameters : SearchParameters {
    size_t reorder_k = 1;
    SearchParameters* base_index_params = nullptr;  // non-owning

    virtual ~IndexScaNNSearchParameters() = default;
};

struct IndexScaNN : IndexRefine {
    explicit IndexScaNN(Index* base_index);
    IndexScaNN(Index* base_index, const float* xb);

    IndexScaNN();

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void reset() override;

    inline bool with_raw_data() const {
        return (refine_index != nullptr);
    }

    int64_t size();

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    std::unique_ptr<IVFIteratorWorkspace> getIteratorWorkspace(
            const float* query_data,
            const IVFSearchParameters* ivfsearchParams) const;

    void getIteratorNextBatch(
            IVFIteratorWorkspace* workspace,
            size_t current_backup_count) const;
};

} // namespace faiss