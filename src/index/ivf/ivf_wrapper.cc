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

#include "index/ivf/ivf_wrapper.h"

#include <memory>

#include "faiss/IndexCosine.h"
#include "faiss/IndexFlat.h"
#include "faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h"
#include "faiss/index_io.h"
#include "index/refine/refine_utils.h"

namespace knowhere {

template <typename IndexIVFType>
IndexIVFWrapper<IndexIVFType>::IndexIVFWrapper(std::unique_ptr<faiss::Index>&& index_in)
    : Index{index_in->d, index_in->metric_type}, index{std::move(index_in)} {
    ntotal = index->ntotal;
    is_trained = index->is_trained;
    is_cosine = index->is_cosine;
    verbose = index->verbose;
    metric_arg = index->metric_arg;
}

template <typename IndexIVFType>
std::unique_ptr<IndexIVFWrapper<IndexIVFType>>
IndexIVFWrapper<IndexIVFType>::from_deserialized(std::unique_ptr<faiss::Index>&& index_in) {
    auto index = std::make_unique<IndexIVFWrapper<IndexIVFType>>(std::move(index_in));

    // check a provided index type
    auto index_base = index->get_base_ivf_index();
    if (index_base == nullptr) {
        return nullptr;
    }

    index_base->make_direct_map(true);

    // done
    return index;
}

template <typename IndexIVFType>
void
IndexIVFWrapper<IndexIVFType>::train(faiss::idx_t n, const float* x) {
    index->train(n, x);
    this->is_trained = index->is_trained;
}

template <typename IndexIVFType>
void
IndexIVFWrapper<IndexIVFType>::add(faiss::idx_t n, const float* x) {
    index->add(n, x);
    this->ntotal = index->ntotal;
}

template <typename IndexIVFType>
void
IndexIVFWrapper<IndexIVFType>::search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances,
                                      faiss::idx_t* labels, const faiss::SearchParameters* params) const {
    index->search(n, x, k, distances, labels, params);
}

template <typename IndexIVFType>
void
IndexIVFWrapper<IndexIVFType>::range_search(faiss::idx_t n, const float* x, float radius,
                                            faiss::RangeSearchResult* result,
                                            const faiss::SearchParameters* params) const {
    index->range_search(n, x, radius, result, params);
}

template <typename IndexIVFType>
void
IndexIVFWrapper<IndexIVFType>::reset() {
    index->reset();
    this->ntotal = 0;
}

template <typename IndexIVFType>
void
IndexIVFWrapper<IndexIVFType>::merge_from(faiss::Index& otherIndex, faiss::idx_t add_id) {
    index->merge_from(otherIndex, add_id);
}

template <typename IndexIVFType>
faiss::DistanceComputer*
IndexIVFWrapper<IndexIVFType>::get_distance_computer() const {
    return index->get_distance_computer();
}

template <typename IndexIVFType>
IndexIVFType*
IndexIVFWrapper<IndexIVFType>::get_base_ivf_index() {
    // try refine
    faiss::IndexRefine* index_refine = dynamic_cast<faiss::IndexRefine*>(index.get());
    faiss::Index* index_for_base = (index_refine != nullptr) ? index_refine->base_index : index.get();

    // Use dynamic_cast to cast to the specific IndexIVFType (e.g., IndexIVFPQ)
    return dynamic_cast<IndexIVFType*>(index_for_base);
}

template <typename IndexIVFType>
const IndexIVFType*
IndexIVFWrapper<IndexIVFType>::get_base_ivf_index() const {
    // try refine
    const faiss::IndexRefine* index_refine = dynamic_cast<const faiss::IndexRefine*>(index.get());
    const faiss::Index* index_for_base = (index_refine != nullptr) ? index_refine->base_index : index.get();
    return dynamic_cast<const IndexIVFType*>(index_for_base);
}

template <typename IndexIVFType>
faiss::IndexRefine*
IndexIVFWrapper<IndexIVFType>::get_refine_index() {
    return dynamic_cast<faiss::IndexRefine*>(index.get());
}

template <typename IndexIVFType>
const faiss::IndexRefine*
IndexIVFWrapper<IndexIVFType>::get_refine_index() const {
    return dynamic_cast<const faiss::IndexRefine*>(index.get());
}

template <typename IndexIVFType>
size_t
IndexIVFWrapper<IndexIVFType>::size() const {
    if (index == nullptr) {
        return 0;
    }

    if (size_cache_.has_value()) {
        return size_cache_.value();
    }

    // a temporary yet expensive workaround
    faiss::cppcontrib::knowhere::CountSizeIOWriter writer;
    faiss::write_index(index.get(), &writer);

    // cache the size
    size_cache_ = writer.total_size;
    return writer.total_size;
}

template <typename IndexIVFType>
template <typename U>
typename std::enable_if<std::is_same_v<U, faiss::IndexIVFScalarQuantizer>,
                        std::unique_ptr<faiss::IVFIteratorWorkspace>>::type
IndexIVFWrapper<IndexIVFType>::getIteratorWorkspace(const float* query_data,
                                                    const faiss::IVFSearchParameters* ivfsearchParams) const {
    // try refine
    const faiss::IndexRefine* index_refine = dynamic_cast<const faiss::IndexRefine*>(index.get());
    const faiss::Index* index_for_base = (index_refine != nullptr) ? index_refine->base_index : index.get();

    const IndexIVFType* index_ivf = dynamic_cast<const IndexIVFType*>(index_for_base);
    if (index_ivf == nullptr) {
        return nullptr;
    }

    // create a workspace. This will make a clone of the transformed_query.
    auto workspace = index_ivf->getIteratorWorkspace(query_data, ivfsearchParams);

    // check if refine exists
    if (index_refine != nullptr) {
        // a regular use case
        workspace->dis_refine =
            std::unique_ptr<faiss::DistanceComputer>(index_refine->refine_index->get_distance_computer());
        // this points to a previously saved clone
        workspace->dis_refine->set_query(workspace->query_data.data());
    } else {
        // don't use refine
        workspace->dis_refine = nullptr;
    }

    // done
    return workspace;
}

template <typename IndexIVFType>
template <typename U>
typename std::enable_if<std::is_same_v<U, faiss::IndexIVFScalarQuantizer>, void>::type
IndexIVFWrapper<IndexIVFType>::getIteratorNextBatch(faiss::IVFIteratorWorkspace* workspace,
                                                    size_t current_backup_count) const {
    const auto ivf = this->get_base_ivf_index();
    if (ivf != nullptr) {
        ivf->getIteratorNextBatch(workspace, current_backup_count);
    }
}

template struct IndexIVFWrapper<faiss::IndexIVFPQ>;
template struct IndexIVFWrapper<faiss::IndexIVFScalarQuantizer>;

template typename std::enable_if<std::is_same_v<faiss::IndexIVFScalarQuantizer, faiss::IndexIVFScalarQuantizer>,
                                 std::unique_ptr<faiss::IVFIteratorWorkspace>>::type
IndexIVFWrapper<faiss::IndexIVFScalarQuantizer>::getIteratorWorkspace<faiss::IndexIVFScalarQuantizer>(
    const float* query_data, const faiss::IVFSearchParameters* ivfsearchParams) const;

template
    typename std::enable_if<std::is_same_v<faiss::IndexIVFScalarQuantizer, faiss::IndexIVFScalarQuantizer>, void>::type
    IndexIVFWrapper<faiss::IndexIVFScalarQuantizer>::getIteratorNextBatch<faiss::IndexIVFScalarQuantizer>(
        faiss::IVFIteratorWorkspace* workspace, size_t current_backup_count) const;

expected<std::unique_ptr<IndexIVFPQWrapper>>
IndexIvfFactory::create_for_pq(faiss::IndexFlat* qzr_raw_ptr, const faiss::idx_t d, const size_t nlist,
                               const size_t nbits, const IvfPqConfig& ivf_pq_cfg, const DataFormatEnum raw_data_format,
                               const faiss::MetricType metric) {
    // the index factory string is either `IVFx,PQ,Refine(y)` or `IVFx,PQ`,
    //   depends on the refine parameters

    // create IndexIVFPQ
    // Index does not own qzr
    auto index = std::make_unique<faiss::IndexIVFPQ>(qzr_raw_ptr, d, nlist, ivf_pq_cfg.m.value(), nbits, metric);

    // create a refiner index, if needed
    std::unique_ptr<faiss::Index> idx_final;
    if (ivf_pq_cfg.refine.value_or(false) && ivf_pq_cfg.refine_type.has_value()) {
        // refine is needed
        const auto base_d = index->d;
        const auto base_metric_type = index->metric_type;
        auto final_index_cnd =
            pick_refine_index(raw_data_format, ivf_pq_cfg.refine_type, std::move(index), base_d, base_metric_type);
        if (!final_index_cnd.has_value()) {
            return expected<std::unique_ptr<IndexIVFPQWrapper>>::Err(Status::invalid_args, "Invalid refine parameters");
        }

        idx_final = std::move(final_index_cnd.value());
    } else {
        // refine is not needed
        idx_final = std::move(index);
    }

    auto result = std::make_unique<IndexIVFPQWrapper>(std::move(idx_final));
    return result;
}

expected<std::unique_ptr<IndexIVFSQWrapper>>
IndexIvfFactory::create_for_sq(faiss::IndexFlat* qzr_raw_ptr, const faiss::idx_t d, const size_t nlist,
                               const IvfSqConfig& ivf_sq_cfg, const DataFormatEnum raw_data_format,
                               const faiss::MetricType metric) {
    // the index factory string is either `IVFx,SQ,Refine(y)` or `IVFx,SQ`,
    //   depends on the refine parameters

    // create IndexIVFSQ
    // Index does not own qzr
    faiss::ScalarQuantizer::QuantizerType quantizer_type;
    // ivf_sq_cfg.sq_type.value() has already been guaranteed to be legal in CheckAndAdjust
    std::string quantizer_type_tolower = str_to_lower(ivf_sq_cfg.sq_type.value());
    if (quantizer_type_tolower == "sq4") {
        quantizer_type = faiss::ScalarQuantizer::QuantizerType::QT_4bit;
    } else if (quantizer_type_tolower == "sq6") {
        quantizer_type = faiss::ScalarQuantizer::QuantizerType::QT_6bit;
    } else {
        quantizer_type = faiss::ScalarQuantizer::QuantizerType::QT_8bit;
    }
    auto index = std::make_unique<faiss::IndexIVFScalarQuantizer>(qzr_raw_ptr, d, nlist, quantizer_type, metric);

    // create a refiner index, if needed
    std::unique_ptr<faiss::Index> idx_final;
    if (ivf_sq_cfg.refine.value_or(false) && ivf_sq_cfg.refine_type.has_value()) {
        // refine is needed
        const auto base_d = index->d;
        const auto base_metric_type = index->metric_type;
        auto final_index_cnd =
            pick_refine_index(raw_data_format, ivf_sq_cfg.refine_type, std::move(index), base_d, base_metric_type);
        if (!final_index_cnd.has_value()) {
            return expected<std::unique_ptr<IndexIVFSQWrapper>>::Err(Status::invalid_args, "Invalid refine parameters");
        }

        idx_final = std::move(final_index_cnd.value());
    } else {
        // refine is not needed
        idx_final = std::move(index);
    }

    auto result = std::make_unique<IndexIVFSQWrapper>(std::move(idx_final));
    return result;
}

}  // namespace knowhere
