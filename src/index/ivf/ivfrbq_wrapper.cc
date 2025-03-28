// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "index/ivf/ivfrbq_wrapper.h"

#include <faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h>

#include <stdexcept>

#include "faiss/IndexPreTransform.h"
#include "faiss/index_io.h"

namespace knowhere {

IndexIVFRaBitQWrapper::IndexIVFRaBitQWrapper(std::unique_ptr<faiss::Index>&& index_in)
    : Index{index_in->d, index_in->metric_type}, index{std::move(index_in)} {
    ntotal = index->ntotal;
    is_trained = index->is_trained;
    verbose = index->verbose;
    metric_arg = index->metric_arg;
}

IndexIVFRaBitQWrapper::~IndexIVFRaBitQWrapper() {
}

void
IndexIVFRaBitQWrapper::train(faiss::idx_t n, const float* x) {
    // index->train(n, x);
    // is_trained = index->is_trained;
    throw std::runtime_error("IndexIVFRaBitQWrapper::train(() is not supposed to be called");
}

void
IndexIVFRaBitQWrapper::add(faiss::idx_t n, const float* x) {
    index->add(n, x);
    this->ntotal = index->ntotal;
    // throw std::runtime_error("IndexIVFRaBitQWrapper::add() is not supposed to be called");
}

void
IndexIVFRaBitQWrapper::search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
                              const faiss::SearchParameters* params) const {
    index->search(n, x, k, distances, labels, params);
}

void
IndexIVFRaBitQWrapper::range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                                    const faiss::SearchParameters* params) const {
    index->range_search(n, x, radius, result, params);
}

void
IndexIVFRaBitQWrapper::reset() {
    index->reset();
    this->ntotal = 0;
}

void
IndexIVFRaBitQWrapper::merge_from(Index& otherIndex, faiss::idx_t add_id) {
    index->merge_from(otherIndex, add_id);
}

faiss::DistanceComputer*
IndexIVFRaBitQWrapper::get_distance_computer() const {
    return index->get_distance_computer();
}

faiss::IndexIVFRaBitQ*
IndexIVFRaBitQWrapper::get_ivfrabitq_index() {
    faiss::IndexPreTransform* index_pt = dynamic_cast<faiss::IndexPreTransform*>(index.get());
    if (index_pt == nullptr) {
        return nullptr;
    }

    return dynamic_cast<faiss::IndexIVFRaBitQ*>(index_pt->index);
}

const faiss::IndexIVFRaBitQ*
IndexIVFRaBitQWrapper::get_ivfrabitq_index() const {
    const faiss::IndexPreTransform* index_pt = dynamic_cast<const faiss::IndexPreTransform*>(index.get());
    if (index_pt == nullptr) {
        return nullptr;
    }

    return dynamic_cast<const faiss::IndexIVFRaBitQ*>(index_pt->index);
}

size_t
IndexIVFRaBitQWrapper::size() const {
    if (index == nullptr) {
        return 0;
    }

    // a temporary yet expensive workaround
    faiss::cppcontrib::knowhere::CountSizeIOWriter writer;
    faiss::write_index(index.get(), &writer);

    // todo
    return writer.total_size;
}

}  // namespace knowhere
