// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License

#include <faiss/IndexIVF.h>

#include <faiss/utils/utils.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <omp.h>
#include <cinttypes>
namespace faiss {

namespace {
IVFSearchParameters gen_search_param(
        const size_t& nprobe,
        const int parallel_mode,
        const size_t& max_codes) {
    IVFSearchParameters params;
    params.nprobe = nprobe;
    params.max_codes = max_codes;
    params.parallel_mode = parallel_mode;
    return params;
}
} // namespace

void IndexIVF::search_thread_safe(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const size_t nprobe,
        const size_t max_codes,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);
    const size_t final_nprobe = std::min(nlist, nprobe);
    FAISS_THROW_IF_NOT(final_nprobe > 0);
    IVFSearchParameters params = gen_search_param(final_nprobe, 0, max_codes);

    // search function for a subset of queries
    auto sub_search_func = [this, k, final_nprobe, bitset, &params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * final_nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * final_nprobe]);

        double t0 = getmillisecs();
        quantizer->search(n, x, final_nprobe, coarse_dis.get(), idx.get());

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * final_nprobe);

        search_preassigned(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                &params,
                ivf_stats,
                bitset);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle paralellization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void IndexIVF::range_search_thread_safe(
        idx_t nx,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const size_t nprobe,
        const size_t max_codes,
        const BitsetView bitset) const {
    const size_t final_nprobe = std::min(nlist, nprobe);
    std::unique_ptr<idx_t[]> keys(new idx_t[nx * final_nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nx * final_nprobe]);

    double t0 = getmillisecs();
    quantizer->search(nx, x, final_nprobe, coarse_dis.get(), keys.get());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(keys.get(), nx * final_nprobe);

    IVFSearchParameters params = gen_search_param(final_nprobe, 0, max_codes);

    range_search_preassigned(
            nx,
            x,
            radius,
            keys.get(),
            coarse_dis.get(),
            result,
            false,
            &params,
            &indexIVF_stats,
            bitset);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

} // namespace faiss
