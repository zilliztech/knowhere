#include <faiss/cppcontrib/knowhere/IndexScaNN.h>

#include <faiss/cppcontrib/knowhere/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/cppcontrib/knowhere/utils/distances.h>
#include <faiss/utils/utils.h>

#include <faiss/Index.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexIVFPQFastScan.h>



namespace faiss::cppcontrib::knowhere {

/***************************************************
 * IndexScaNN
 ***************************************************/

IndexScaNN::IndexScaNN(Index* base_index)
        : IndexRefine(
                  base_index,
                  new IndexFlat(base_index->d, base_index->metric_type)) {
    is_trained = base_index->is_trained;
    own_refine_index = true;
    FAISS_THROW_IF_NOT_MSG(
            base_index->ntotal == 0,
            "base_index should be empty in the beginning");
}

IndexScaNN::IndexScaNN(Index* base_index, const float* xb)
        : IndexRefine(base_index, nullptr) {
    is_trained = base_index->is_trained;
    if (xb) {
        refine_index = new IndexFlat(base_index->d, base_index->metric_type);
    }
    own_refine_index = true;
}

IndexScaNN::IndexScaNN() : IndexRefine() {}

void IndexScaNN::train(idx_t n, const float* x) {
    base_index->train(n, x);
    if (refine_index)
        refine_index->train(n, x);
    is_trained = true;
}

void IndexScaNN::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    base_index->add(n, x);
    if (refine_index)
        refine_index->add(n, x);
    ntotal = base_index->ntotal;
}

void IndexScaNN::reset() {
    base_index->reset();
    if (refine_index)
        refine_index->reset();
    ntotal = 0;
}

namespace {

template <class C>
static void reorder_2_heaps(
        idx_t n,
        idx_t k,
        idx_t* labels,
        float* distances,
        idx_t k_base,
        const idx_t* base_labels,
        const float* base_distances) {
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        idx_t* idxo = labels + i * k;
        float* diso = distances + i * k;
        const idx_t* idxi = base_labels + i * k_base;
        const float* disi = base_distances + i * k_base;

        heap_heapify<C>(k, diso, idxo, disi, idxi, k);
        if (k_base != k) { // add remaining elements
            heap_addn<C>(k, diso, idxo, disi + k, idxi + k, k_base - k);
        }
        heap_reorder<C>(k, diso, idxo);
    }
}

} // anonymous namespace

template <typename FastScanIndex>
static int64_t fast_scan_size(
        const FastScanIndex* index,
        const Index* refine_index) {
    auto nb = index->invlists->compute_ntotal();
    auto code_size = index->code_size;
    const auto& pq = index->pq;
    auto nlist = index->nlist;
    auto d = index->d;

    auto capacity =
            nb * code_size + nb * sizeof(int64_t) + nlist * d * sizeof(float);
    auto centroid_table = pq.M * pq.ksub * pq.dsub * sizeof(float);
    auto precomputed_table = nlist * pq.M * pq.ksub * sizeof(float);

    auto raw_data = (refine_index ? index->ntotal * d * sizeof(float) : 0);
    return (capacity + centroid_table + precomputed_table + raw_data);
}

static bool is_supported_fast_scan_base(const Index* index) {
    return dynamic_cast<const ::faiss::IndexIVFPQFastScan*>(index) ||
            dynamic_cast<const IndexIVFPQFastScan*>(index);
}

static const float* get_inverse_l2_norms_or_null(const Index* index) {
    auto norms = dynamic_cast<const HasInverseL2Norms*>(index);
    return norms ? norms->get_inverse_l2_norms() : nullptr;
}

int64_t IndexScaNN::size() {
    if (auto index_ = dynamic_cast<const ::faiss::IndexIVFPQFastScan*>(
                base_index)) {
        return fast_scan_size(index_, refine_index);
    }
    if (auto index_ = dynamic_cast<const IndexIVFPQFastScan*>(base_index)) {
        return fast_scan_size(index_, refine_index);
    }
    FAISS_THROW_MSG("IndexScaNN base index must be IndexIVFPQFastScan");
    return 0;
}

void IndexScaNN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    const IndexScaNNSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IndexScaNNSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexScaNN params have incorrect type");
    }

    idx_t k_base = (params != nullptr) ? params->reorder_k : idx_t(k * k_factor);
    SearchParameters* base_index_params = 
        (params != nullptr) ? params->base_index_params : nullptr;

    FAISS_THROW_IF_NOT(k_base >= k);

    FAISS_THROW_IF_NOT_MSG(
            is_supported_fast_scan_base(base_index),
            "IndexScaNN base index must be IndexIVFPQFastScan");

    // nothing to refine, directly return result
    if (refine_index == nullptr) {
        base_index->search(
            n,
            x,
            k,
            distances,
            labels,
            base_index_params);
        return;
    }

    idx_t* base_labels = labels;
    float* base_distances = distances;
    std::unique_ptr<idx_t[]> del1;
    std::unique_ptr<float[]> del2;

    if (k != k_base) {
        del1 = std::make_unique<idx_t[]>(n * k_base);
        base_labels = del1.get();
        del2 = std::make_unique<float[]>(n * k_base);
        base_distances = del2.get();
    }

    base_index->search(
            n,
            x,
            k_base,
            base_distances,
            base_labels,
            base_index_params);
    for (idx_t i = 0; i < n * k_base; i++) {
        FAISS_THROW_IF_NOT(base_labels[i] >= -1 && base_labels[i] < ntotal);
    }

    // compute refined distances. refine_index may be deserialized as a
    // baseline ::faiss::IndexFlat (IxFI/IxF2) or as the knowhere IndexFlat
    // subclass (IxFl); cast to the common baseline type.
    auto rf = dynamic_cast<const ::faiss::IndexFlat*>(refine_index);
    FAISS_THROW_IF_NOT(rf);

    rf->compute_distance_subset(n, x, k_base, base_distances, base_labels);

    if (auto inverse_l2_norms = get_inverse_l2_norms_or_null(base_index)) {
        for (idx_t i = 0; i < n * k_base; i++) {
            if (base_labels[i] >= 0) {
                base_distances[i] *= inverse_l2_norms[base_labels[i]];
            }
        }
    }
    // sort and store result
    if (metric_type == METRIC_L2) {
        typedef CMax<float, idx_t> C;
        reorder_2_heaps<C>(
                n, k, labels, distances, k_base, base_labels, base_distances);

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        typedef CMin<float, idx_t> C;
        reorder_2_heaps<C>(
                n, k, labels, distances, k_base, base_labels, base_distances);
    } else {
        FAISS_THROW_MSG("Metric type not supported");
    }
}

void IndexScaNN::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(n == 1);  // currently knowhere will split nq to 1

    FAISS_THROW_IF_NOT(is_trained);

    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexScaNN params have incorrect type");
    }

    FAISS_THROW_IF_NOT_MSG(
            is_supported_fast_scan_base(base_index),
            "IndexScaNN base index must be IndexIVFPQFastScan");
    auto base_ivf = dynamic_cast<const ::faiss::IndexIVF*>(base_index);
    FAISS_THROW_IF_NOT(base_ivf);

    IVFSearchParameters ivf_search_params;
    ivf_search_params.nprobe = base_ivf->nlist;
    ivf_search_params.max_empty_result_buckets =
            params ? params->max_empty_result_buckets : 0;
    // todo aguzhva: this is somewhat hacky
    ivf_search_params.sel = params ? params->sel : nullptr;

    base_index->range_search(n, x, radius, result, &ivf_search_params);

    // nothing to refine, directly return the result
    if (refine_index == nullptr) {
        return;
    }

    // compute refined distances. refine_index may be deserialized as a
    // baseline ::faiss::IndexFlat (IxFI/IxF2) or as the knowhere IndexFlat
    // subclass (IxFl); cast to the common baseline type.
    auto rf = dynamic_cast<const ::faiss::IndexFlat*>(refine_index);
    FAISS_THROW_IF_NOT(rf);

    rf->compute_distance_subset(n, x, result->lims[1], result->distances, result->labels);

    auto inverse_l2_norms = get_inverse_l2_norms_or_null(base_index);

    idx_t current = 0;
    for (idx_t i = 0; i < result->lims[1]; ++i) {
        if (inverse_l2_norms) {
            result->distances[i] *= inverse_l2_norms[result->labels[i]];
        }
        if (metric_type == METRIC_L2) {
            if (result->distances[i] < radius) {
                result->distances[current] = result->distances[i];
                result->labels[current] = result->labels[i];
                current++;
            }

        } else if (metric_type == METRIC_INNER_PRODUCT) {
            if (result->distances[i] > radius) {
                result->distances[current] = result->distances[i];
                result->labels[current] = result->labels[i];
                current++;
            }
        } else {
            FAISS_THROW_MSG("Metric type not supported");
        }
    }
    result->lims[1] = current;
}

}
