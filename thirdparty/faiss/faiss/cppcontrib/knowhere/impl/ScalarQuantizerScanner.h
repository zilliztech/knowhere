#pragma once

#include <faiss/cppcontrib/knowhere/IndexIVF.h>
//#include <faiss/impl/IDSelector.h>

//struct InvertedListScanner;
//struct IDSelector;

#include <faiss/cppcontrib/knowhere/utils/distances_if.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/*******************************************************************
 * IndexScalarQuantizer/IndexIVFScalarQuantizer scanner object
 *
 * It is an InvertedListScanner, but is designed to work with
 * IndexScalarQuantizer as well.
 ********************************************************************/
template <
        // A predicate for filtering elements.
        //   std::optional<bool> Pred(const size_t idx);
        // * return true to accept an element.
        // * return false to reject an element.
        // * return std::nullopt to break the iteration loop.
        typename Pred,
        // Apply an element.
        //   void Apply(const float dis, const size_t idx);
        typename Apply,
        typename DCClass>
void fvec_distance_ny_scalar_if(
        const DCClass& dc,
        const uint8_t* __restrict codes,
        const size_t code_size,
        const size_t ny,
        Pred pred,
        Apply apply) {
    // compute a distance from the query to 1 element
    auto distance1 = [&dc, codes, code_size](const size_t idx) {
        return dc.query_to_code(codes + idx * code_size);
    };

    // compute distances from the query to 4 elements
    auto distance4 = [&dc, codes, code_size](
                             const std::array<size_t, 4> indices,
                             std::array<float, 4>& dis) {
        dc.query_to_codes_batch_4(
                codes + indices[0] * code_size,
                codes + indices[1] * code_size,
                codes + indices[2] * code_size,
                codes + indices[3] * code_size,
                dis[0],
                dis[1],
                dis[2],
                dis[3]);
    };

    NoRemapping remapper;

    fvec_distance_ny_if<
            Pred,
            decltype(distance1),
            decltype(distance4),
            decltype(remapper),
            Apply,
            4,
            DEFAULT_BUFFER_SIZE>(
            ny, pred, distance1, distance4, remapper, apply);
}

/* use_sel = 0: don't check selector
 * = 1: check on ids[j]
 * = 2: check in j directly (normally ids is nullptr and store_pairs)
 */

template <class DCClass, int use_sel>
struct IVFSQScannerIP : InvertedListScanner {
    DCClass dc;
    bool by_residual;

    float accu0; /// added to all distances

    IVFSQScannerIP(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained), by_residual(by_residual), accu0(0) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
        this->keep_max = true;
    }

    void set_query(const float* query) override {
        dc.set_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        accu0 = by_residual ? coarse_dis : 0;
    }

    float distance_to_code(const uint8_t* code) const final {
        return accu0 + dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k,
            size_t& scan_cnt) const override {
        size_t nup = 0;
        // baseline
        // for (size_t j = 0; j < list_size; j++, codes += code_size) {
        //     if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
        //         continue;
        //     }

        //     // todo aguzhva: upgrade
        //     float accu = accu0 + dc.query_to_code(codes);

        //     if (accu > simi[0]) {
        //         int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
        //         minheap_replace_top(k, simi, idxi, accu, id);
        //         nup++;
        //     }
        // }

        // the lambda that filters acceptable elements.
        auto filter = [&](const size_t j) {
            return (!use_sel || sel->is_member(use_sel == 1 ? ids[j] : j));
        };

        // the lambda that applies a filtered element.
        auto apply = [&](const float dis_in, const size_t j) {
            const float dis = accu0 + dis_in;
            if (dis > simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                minheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
        };

        // compute distances
        fvec_distance_ny_scalar_if(
                dc, codes, code_size, list_size, filter, apply);
        return nup;
    }

    void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        // the lambda that filters acceptable elements.
        auto filter = [&](const size_t j) {
            return (!use_sel || sel->is_member(use_sel == 1 ? ids[j] : j));
        };
        // the lambda that applies a valid element.
        auto apply = [&](const float dis_in, const size_t j) {
            const float dis = accu0 + dis_in;
            out.emplace_back(ids[j], dis);
        };
        fvec_distance_ny_scalar_if(
                dc, codes, code_size, list_size, filter, apply);
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            // todo aguzhva: upgrade
            float accu = accu0 + dc.query_to_code(codes);
            if (accu > radius) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(accu, id);
            }
        }
    }
};

/* use_sel = 0: don't check selector
 * = 1: check on ids[j]
 * = 2: check in j directly (normally ids is nullptr and store_pairs)
 */

template <class DCClass, int use_sel>
struct IVFSQScannerL2 : InvertedListScanner {
    DCClass dc;

    bool by_residual;
    const Index* quantizer;
    const float* x; /// current query

    std::vector<float> tmp;

    IVFSQScannerL2(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            const Index* quantizer,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained),
              by_residual(by_residual),
              quantizer(quantizer),
              x(nullptr),
              tmp(d) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        x = query;
        if (!quantizer) {
            dc.set_query(query);
        }
    }

    void set_list(idx_t list_no, float) override {
        this->list_no = list_no;
        if (by_residual) {
            // shift of x_in wrt centroid
            quantizer->compute_residual(x, tmp.data(), list_no);
            dc.set_query(tmp.data());
        } else {
            dc.set_query(x);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k,
            size_t& scan_cnt) const override {
        size_t nup = 0;

        // // baseline
        // for (size_t j = 0; j < list_size; j++, codes += code_size) {
        //     if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
        //         continue;
        //     }
        //
        //     float dis = dc.query_to_code(codes);
        //
        //     if (dis < simi[0]) {
        //         int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
        //         maxheap_replace_top(k, simi, idxi, dis, id);
        //         nup++;
        //     }
        // }        

        // the lambda that filters acceptable elements.
        auto filter = 
            [&](const size_t j) { return (!use_sel || sel->is_member(use_sel == 1 ? ids[j] : j)); };

        // the lambda that applies a filtered element.
        auto apply = 
            [&](const float dis, const size_t j) {
                if (dis < simi[0]) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    maxheap_replace_top(k, simi, idxi, dis, id);
                    nup++;
                }
            };

        // compute distances
        fvec_distance_ny_scalar_if(
                dc, codes, code_size, list_size, filter, apply);

        return nup;
    }

    void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        // the lambda that filters acceptable elements.
        auto filter = [&](const size_t j) {
            return (!use_sel || sel->is_member(use_sel == 1 ? ids[j] : j));
        };
        // the lambda that applies a valid element.
        auto apply = [&](const float dis_in, const size_t j) {
            out.emplace_back(ids[j], dis_in);
        };
        fvec_distance_ny_scalar_if(
                dc, codes, code_size, list_size, filter, apply);
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            // todo aguzhva: upgrade
            float dis = dc.query_to_code(codes);
            if (dis < radius) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
            }
        }
    }
};

}
}
}