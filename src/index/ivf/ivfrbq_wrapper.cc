#include "index/ivf/ivfrbq_wrapper.h"

#include <stdexcept>

#include "faiss/index_io.h"
#include "faiss/IndexPreTransform.h"

#include <faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h>

using namespace faiss;

namespace knowhere {

IndexIVFRaBitQWrapper::IndexIVFRaBitQWrapper(std::unique_ptr<faiss::Index>&& index_in) : 
    Index{index_in->d, index_in->metric_type},
    index{std::move(index_in)} 
{
    ntotal = index->ntotal;
    is_trained = index->is_trained;
    verbose = index->verbose;
    metric_arg = index->metric_arg;
}

IndexIVFRaBitQWrapper::~IndexIVFRaBitQWrapper() {}

void IndexIVFRaBitQWrapper::train(idx_t n, const float* x) {
    // index->train(n, x);
    // is_trained = index->is_trained;
    throw std::runtime_error("IndexIVFRaBitQWrapper::train(() is not supposed to be called");
}

void IndexIVFRaBitQWrapper::add(idx_t n, const float* x) {
    // index->add(n, x);
    // this->ntotal = index->ntotal;
    throw std::runtime_error("IndexIVFRaBitQWrapper::add() is not supposed to be called");
}

void IndexIVFRaBitQWrapper::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    index->search(n, x, k, distances, labels, params);
}

void IndexIVFRaBitQWrapper::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    index->range_search(n, x, radius, result, params);
}

void IndexIVFRaBitQWrapper::reset() {
    index->reset();
    this->ntotal = 0;
}

void IndexIVFRaBitQWrapper::merge_from(Index& otherIndex, idx_t add_id) {
    index->merge_from(otherIndex, add_id);
}

DistanceComputer* IndexIVFRaBitQWrapper::get_distance_computer() const {
    return index->get_distance_computer();
}

IndexIVFRaBitQ* IndexIVFRaBitQWrapper::get_ivfrabitq_index() {
    IndexPreTransform* index_pt = dynamic_cast<IndexPreTransform*>(index.get());
    if (index_pt == nullptr) {
        return nullptr;
    }

    return dynamic_cast<IndexIVFRaBitQ*>(index_pt->index);
}

const IndexIVFRaBitQ* IndexIVFRaBitQWrapper::get_ivfrabitq_index() const {
    const IndexPreTransform* index_pt = dynamic_cast<const IndexPreTransform*>(index.get());
    if (index_pt == nullptr) {
        return nullptr;
    }

    return dynamic_cast<const IndexIVFRaBitQ*>(index_pt->index);
}

size_t IndexIVFRaBitQWrapper::size() const {
    if (index == nullptr) {
        return 0;
    }

    // a temporary yet expensive workaround
    faiss::cppcontrib::knowhere::CountSizeIOWriter writer;
    faiss::write_index(index.get(), &writer);

    // todo
    return writer.total_size;
}


}