/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/IndexIVFRaBitQ.h>

#include <memory>
#include <utility>
#include <vector>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/cppcontrib/knowhere/utils/distances_if.h>

namespace faiss::cppcontrib::knowhere {

namespace {

struct ScannerDistanceComputer : ::faiss::FlatCodesDistanceComputer {
    const ::faiss::InvertedListScanner& scanner;

    explicit ScannerDistanceComputer(
            const ::faiss::InvertedListScanner& scanner_in)
            : scanner(scanner_in) {}

    void set_query(const float* /* x */) override {}

    float symmetric_dis(idx_t /* i */, idx_t /* j */) override {
        FAISS_THROW_MSG("symmetric_dis not implemented");
    }

    float distance_to_code(const uint8_t* code) override {
        return scanner.distance_to_code(code);
    }
};

struct RaBitScannerHookShim : InvertedListScanner {
    std::unique_ptr<::faiss::InvertedListScanner> scanner;

    explicit RaBitScannerHookShim(
            std::unique_ptr<::faiss::InvertedListScanner> scanner_in)
            : InvertedListScanner(scanner_in->store_pairs, scanner_in->sel),
              scanner(std::move(scanner_in)) {
        keep_max = scanner->keep_max;
        code_size = scanner->code_size;
        list_no = scanner->list_no;
    }

    void set_query(const float* query_vector) override {
        scanner->set_query(query_vector);
    }

    void set_list(idx_t list_no_in, float coarse_dis) override {
        scanner->set_list(list_no_in, coarse_dis);
        list_no = scanner->list_no;
    }

    float distance_to_code(const uint8_t* code) const override {
        return scanner->distance_to_code(code);
    }

    size_t iterate_codes(
            InvertedListsIterator* iterator,
            float* distances,
            idx_t* labels,
            size_t k,
            size_t& list_size) const override {
        return scanner->iterate_codes(iterator, distances, labels, k, list_size);
    }

    void scan_codes_range(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& result) const override {
        scanner->scan_codes_range(n, codes, ids, radius, result);
    }

    void iterate_codes_range(
            InvertedListsIterator* iterator,
            float radius,
            RangeQueryResult& result,
            size_t& list_size) const override {
        return scanner->iterate_codes_range(iterator, radius, result, list_size);
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return scanner->scan_codes(n, codes, ids, handler);
    }

    void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        ScannerDistanceComputer dc(*scanner);

        auto filter = [&](const size_t j) {
            return sel == nullptr || sel->is_member(ids[j]);
        };

        auto apply = [&](const float dis, const size_t j) {
            out.emplace_back(ids[j], dis);
        };

        // Mirrors the legacy fork RaBitQ hook (parent of
        // 562404012ab320e9c4b7fec3442a52f5f977aee0, lines 181-202):
        // selector-aware flat-code distance streaming. The concrete
        // baseline scanner remains anonymous, so this shim adapts its
        // distance_to_code() instead of reaching into the old RaBitQ
        // distance computer directly.
        distance_compute_by_idx_if_flatcodes(
                codes, code_size, list_size, &dc, filter, apply);
    }
};

} // namespace

IndexIVFRaBitQ::IndexIVFRaBitQ(
        Index* quantizer,
        const size_t d,
        const size_t nlist,
        MetricType metric,
        bool own_invlists,
        uint8_t nb_bits)
        : ::faiss::IndexIVFRaBitQ(
                  quantizer,
                  d,
                  nlist,
                  metric,
                  own_invlists,
                  nb_bits) {}

IndexIVFRaBitQ::IndexIVFRaBitQ() = default;

::faiss::InvertedListScanner* IndexIVFRaBitQ::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* params) const {
    return new RaBitScannerHookShim(
            std::unique_ptr<::faiss::InvertedListScanner>(
                    ::faiss::IndexIVFRaBitQ::get_InvertedListScanner(
                            store_pairs, sel, params)));
}

} // namespace faiss::cppcontrib::knowhere
