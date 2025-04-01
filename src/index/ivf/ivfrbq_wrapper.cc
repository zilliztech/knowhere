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

#include <memory>

#include "faiss/IndexFlat.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/index_io.h"

namespace knowhere {

IndexIVFRaBitQWrapper::IndexIVFRaBitQWrapper(const faiss::idx_t d, const size_t nlist, const uint8_t qb,
                                             faiss::MetricType metric)
    : faiss::Index(d, metric) {
    auto idx_flat = std::make_unique<faiss::IndexFlat>(d, metric, false);
    auto idx_ivfrbq = std::make_unique<faiss::IndexIVFRaBitQ>(idx_flat.release(), d, nlist, metric);
    idx_ivfrbq->own_fields = true;
    idx_ivfrbq->qb = qb;

    auto rr = std::make_unique<faiss::RandomRotationMatrix>(d, d);
    auto idx_rr = std::make_unique<faiss::IndexPreTransform>(rr.release(), idx_ivfrbq.release());
    idx_rr->own_fields = true;

    index = std::move(idx_rr);

    this->is_trained = index->is_trained;
    this->is_cosine = index->is_cosine;
}

IndexIVFRaBitQWrapper::IndexIVFRaBitQWrapper(std::unique_ptr<faiss::Index>&& index_in)
    : Index{index_in->d, index_in->metric_type}, index{std::move(index_in)} {
    ntotal = index->ntotal;
    is_trained = index->is_trained;
    verbose = index->verbose;
    metric_arg = index->metric_arg;
}

std::unique_ptr<IndexIVFRaBitQWrapper>
IndexIVFRaBitQWrapper::from_deserialized(std::unique_ptr<faiss::Index>&& index_in) {
    auto index = std::unique_ptr<IndexIVFRaBitQWrapper>(new IndexIVFRaBitQWrapper(std::move(index_in)));

    // check a provided index type
    auto index_rabitq = index->get_ivfrabitq_index();
    if (index_rabitq == nullptr) {
        return nullptr;
    }

    // construct an index map
    index_rabitq->make_direct_map(true);

    // done
    return index;
}

void
IndexIVFRaBitQWrapper::train(faiss::idx_t n, const float* x) {
    index->train(n, x);
    is_trained = index->is_trained;
}

void
IndexIVFRaBitQWrapper::add(faiss::idx_t n, const float* x) {
    index->add(n, x);
    this->ntotal = index->ntotal;
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

std::unique_ptr<faiss::IVFIteratorWorkspace>
IndexIVFRaBitQWrapper::getIteratorWorkspace(const float* query_data,
                                            const faiss::IVFSearchParameters* ivfsearchParams) const {
    const faiss::IndexPreTransform* index_pt = dynamic_cast<const faiss::IndexPreTransform*>(index.get());
    if (index_pt == nullptr) {
        return nullptr;
    }

    const faiss::IndexIVFRaBitQ* index_rbq = dynamic_cast<const faiss::IndexIVFRaBitQ*>(index_pt->index);
    if (index_rbq == nullptr) {
        return nullptr;
    }

    // ok, transform the query
    std::unique_ptr<const float[]> transformed_query(index_pt->apply_chain(1, query_data));
    // create a workspace
    auto workspace = index_rbq->getIteratorWorkspace(transformed_query.get(), ivfsearchParams);
    // done
    return workspace;
}

void
IndexIVFRaBitQWrapper::getIteratorNextBatch(faiss::IVFIteratorWorkspace* workspace, size_t current_backup_count) const {
    const auto ivfrbq = this->get_ivfrabitq_index();
    ivfrbq->getIteratorNextBatch(workspace, current_backup_count);
}

}  // namespace knowhere
