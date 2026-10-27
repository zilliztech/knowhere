// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <array>
#include <atomic>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

#include "catch2/catch_test_macros.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/index/index.h"

namespace {

// Keep the public foundation API minimal; sparse input lanes are not exposed.
template <typename Filter>
concept HasMaskedTest =
    requires(const Filter& filter, const int32_t* ids) { filter.test(ids, uint32_t{1}, uint64_t{1}); };
template <typename Filter>
concept HasMaskedMappedTest = requires(const Filter& filter, const int32_t* ids) {
                                  filter.test_mapped(ids, uint32_t{1}, ids, size_t{1}, uint64_t{1});
                              };
static_assert(!HasMaskedTest<knowhere::BitsetView>);
static_assert(!HasMaskedMappedTest<knowhere::BitsetView>);

// Unchanged Cohere train IDs, rows [0, 64); only callback contract inputs.
// Source: https://s3.us-west-2.amazonaws.com/assets.zilliz.com/benchmark/cohere_medium_1m/train.parquet
// Parquet SHA256: de6b84eb08fc5203a56eef619970a14f5cae08d0a0ae0bcd23b8058d7045f381
const std::vector<int64_t>&
OriginalIds() {
    static const std::vector<int64_t> ids{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    };
    return ids;
}

struct Factory {
    const std::vector<int64_t>& values = OriginalIds();
    mutable std::atomic<size_t> created{0}, destroyed{0}, batches{0}, evaluated{0};
    bool fail_create = false;
    bool null_create = false;
    bool fail_eval = false;
    bool invalid_accepted = false;
};

struct Worker {
    const Factory* factory;
    size_t batches = 0;  // Mutable state is private to each bound task.
};

knowhere::CandidateEvalStatus
Create(const void* context, void** output) noexcept {
    try {
        const auto* factory = static_cast<const Factory*>(context);
        *output = factory->null_create ? nullptr : new Worker{factory};
        if (*output != nullptr) {
            ++factory->created;
        }
        return factory->fail_create ? knowhere::CandidateEvalStatus::Failed : knowhere::CandidateEvalStatus::Success;
    } catch (...) {
        *output = nullptr;
        return knowhere::CandidateEvalStatus::Failed;
    }
}

void
Destroy(void* opaque) noexcept {
    if (opaque != nullptr) {
        auto* worker = static_cast<Worker*>(opaque);
        ++worker->factory->destroyed;
        delete worker;
    }
}

knowhere::CandidateEvalStatus
Evaluate(void* opaque, const int32_t* ids, uint32_t count, uint64_t active, uint64_t* accepted) noexcept {
    const auto* worker = static_cast<Worker*>(opaque);
    const auto* factory = worker->factory;
    if (factory->fail_eval) {
        return knowhere::CandidateEvalStatus::Failed;
    }
    *accepted = 0;
    ++factory->batches;
    ++static_cast<Worker*>(opaque)->batches;
    if (factory->invalid_accepted) {
        *accepted = ~active;
        return knowhere::CandidateEvalStatus::Success;
    }
    for (uint32_t lane = 0; lane < count; ++lane) {
        if ((active & (uint64_t{1} << lane)) == 0) {
            continue;
        }
        if (ids[lane] < 0 || static_cast<size_t>(ids[lane]) >= factory->values.size()) {
            return knowhere::CandidateEvalStatus::InvalidArgument;
        }
        ++factory->evaluated;
        if (factory->values[ids[lane]] % 5 < 3) {
            *accepted |= uint64_t{1} << lane;
        }
    }
    return knowhere::CandidateEvalStatus::Success;
}

knowhere::CandidateEvaluatorView
Descriptor(const Factory& factory) {
    return {.context = &factory, .create_worker = Create, .destroy_worker = Destroy, .eval_batch = Evaluate};
}

std::vector<uint8_t>
Mandatory(size_t rows, const std::vector<int32_t>& excluded) {
    std::vector<uint8_t> result((rows + 7) / 8, 0);
    for (const auto row : excluded) {
        result[row >> 3] |= uint8_t{1} << (row & 7);
    }
    return result;
}

}  // namespace

TEST_CASE("Candidate evaluator combines mandatory bits, lanes and original scalar truth", "[bitset][candidate]") {
    Factory factory;
    auto view = Descriptor(factory);
    auto bits = Mandatory(factory.values.size(), {1, 9, 31, 63});
    knowhere::BitsetView description(bits.data(), factory.values.size());
    description.count_filtered_bits(0, factory.values.size());
    description.set_candidate_evaluator(&view);
    REQUIRE_FALSE(description.empty());
    REQUIRE_FALSE(description.is_bitmap());
    REQUIRE(description.count() == 4);
    REQUIRE(factory.created == 0);
    REQUIRE_THROWS_AS(description.test(0), std::invalid_argument);
    REQUIRE_THROWS_AS(description.data(), std::logic_error);
    REQUIRE_FALSE(description.supports_direct_test(64));
    // test_direct has the original unchecked-fast-access precondition:
    // supports_direct_test must have succeeded before entering its hot loop.
    std::array<int32_t, 64> ids;
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<int32_t>(i);
    }
    {
        auto bound = description.bind();
        auto copy = bound;
        REQUIRE(factory.created == 1);
        for (const uint32_t count : {0, 1, 31, 32, 33, 63, 64}) {
            const auto lanes = knowhere::CandidateEvaluatorExecution::LaneMask(count);
            const auto actual = bound.test(ids.data(), count);
            for (uint32_t lane = 0; lane < count; ++lane) {
                const bool excluded = ((bits[lane >> 3] >> (lane & 7)) & 1) != 0 || factory.values[lane] % 5 >= 3;
                REQUIRE(((actual >> lane) & 1) == excluded);
                REQUIRE(bound.test(ids[lane]) == excluded);
            }
            REQUIRE((actual & ~lanes) == 0);
        }
        const auto batches = factory.batches.load();
        REQUIRE(copy.test(nullptr, 0) == 0);
        REQUIRE(factory.batches == batches);
        const int32_t checked[] = {-1, 0, std::numeric_limits<int32_t>::max(), 3};
        const auto before_rows = factory.evaluated.load();
        REQUIRE(copy.test(checked, 4) == 13);
        REQUIRE(factory.evaluated == before_rows + 2);
        REQUIRE_THROWS_AS(copy.test(ids.data(), 65), std::invalid_argument);
        REQUIRE_THROWS_AS(copy.test(nullptr, 1), std::invalid_argument);
        auto independent = description.bind();
        REQUIRE(factory.created == 2);
        REQUIRE(independent.callback_batches() == 0);
        REQUIRE(copy.callback_batches() == bound.callback_batches());
    }
    REQUIRE(factory.created == factory.destroyed);
}

TEST_CASE("Candidate callback-only public domain survives backend projection", "[bitset][candidate][id_map]") {
    Factory factory;
    auto view = Descriptor(factory);
    knowhere::BitsetView description;
    REQUIRE_THROWS_AS(description.set_candidate_evaluator(&view), std::invalid_argument);
    description.set_candidate_evaluator(&view, factory.values.size());
    REQUIRE(description.num_bits() == 0);
    REQUIRE(description.size() == factory.values.size());
    REQUIRE(description.has_known_count());
    REQUIRE(description.count() == 0);
    REQUIRE_FALSE(description.empty());
    auto filter = description.bind();
    const int32_t ids[] = {2, 0, 3, 2, -1, 64};
    const int32_t reorder[] = {4, 1, 2, 60, -1};
    const auto before = factory.evaluated.load();
    const auto actual = filter.test_mapped(ids, 6, reorder, 5);
    REQUIRE(factory.evaluated == before + 4);
    for (size_t lane = 0; lane < 6; ++lane) {
        const bool invalid = ids[lane] < 0 || ids[lane] >= 5;
        const bool excluded = invalid || factory.values[reorder[ids[lane]]] % 5 >= 3;
        REQUIRE(((actual >> lane) & 1) == excluded);
    }
    knowhere::IdArray out_ids;
    out_ids.SetType(knowhere::IdArray::Type::ARRAY);
    out_ids.Set(reorder, 5);
    filter.set_out_ids(out_ids, 5);
    filter.set_id_offset(1);
    filter.set_vector_count(4);
    REQUIRE(filter.public_row_count() == factory.values.size());
    REQUIRE(filter.count() == 0);  // Mandatory only, not an estimate of user truth.
    for (int32_t id = 0; id < 4; ++id) {
        const auto row = reorder[id + 1];
        REQUIRE(filter.test(id) == (row < 0 || factory.values[row] % 5 >= 3));
    }
    REQUIRE(filter.test(4));
    REQUIRE(filter.test(-1));
    filter.set_out_ids({}, 0);
    filter.set_id_offset(std::numeric_limits<size_t>::max());
    REQUIRE(filter.test(1));
    REQUIRE(filter.test(0));
}

TEST_CASE("Candidate bitmap projection never mistakes a local zero count for public visibility",
          "[bitset][candidate]") {
    Factory factory;
    auto view = Descriptor(factory);
    auto bits = Mandatory(factory.values.size(), {2});
    knowhere::BitsetView filter(bits.data(), factory.values.size());
    filter.set_candidate_evaluator(&view);
    filter.set_vector_count(1);
    filter.set_filter_count(0);
    filter = filter.bind();
    const int32_t ids[] = {0, 1};
    const int32_t rows[] = {2, 3};
    const auto before = factory.evaluated.load();
    REQUIRE(filter.test_mapped(ids, 2, rows, 2) == 3);
    REQUIRE(factory.evaluated == before + 1);
}

#if defined(__unix__) || defined(__APPLE__)
TEST_CASE("Candidate attachment preserves saturated bitmap domains and projected row errors",
          "[bitset][candidate][id_map]") {
    // Protocol metadata, not generated/extended scalar data: reserve a sparse
    // virtual bitmap to exercise the original int64 API beyond UINT32_MAX.
    // Only a few pages and an eight-byte count window are touched. Never use
    // an undersized backing buffer or scan all512MiB to test this boundary.
    const size_t boundary = std::numeric_limits<uint32_t>::max();
    const size_t rows = boundary + 65;
    const size_t bytes = (rows + 7) / 8;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    struct SparseBitmap {
        void* address;
        size_t bytes;
        ~SparseBitmap() {
            if (address != MAP_FAILED) {
                ::munmap(address, bytes);
            }
        }
    } mapping{::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0), bytes};
    REQUIRE(mapping.address != MAP_FAILED);
    auto* bits = static_cast<uint8_t*>(mapping.address);
    for (const auto row : {boundary, boundary + 2}) {
        bits[row >> 3] |= uint8_t{1} << (row & 7);
    }
    knowhere::BitsetView description(bits, rows);
    REQUIRE(description.num_bits() == rows);
    REQUIRE(description.byte_size() == bytes);
    for (const auto row : {boundary - 1, boundary, boundary + 1, boundary + 2, rows - 1, rows}) {
        REQUIRE(description.test(row) == (row >= rows || row == boundary || row == boundary + 2));
    }
    description.count_filtered_bits(boundary - 4, 64);
    REQUIRE(description.size() == 64);
    REQUIRE(description.count() == 2);
    description.set_vector_count(rows);
    description.set_id_offset(boundary);
    const int32_t window[] = {0, 1, 2, 64, 65, -1};
    REQUIRE(description.test(window, 6) == 53);
    description.set_id_offset(0);

    Factory factory;
    auto view = Descriptor(factory);
    description.set_candidate_evaluator(&view);
    REQUIRE(description.num_bits() == rows);
    REQUIRE_FALSE(description.supports_direct_test(1));
    REQUIRE(description.test(-1));
    REQUIRE(description.test(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1));
    REQUIRE_THROWS_AS(description.test(0), std::invalid_argument);
    {
        auto bound = description.bind();
        REQUIRE(bound.test(0) == (factory.values[0] % 5 >= 3));
        bound.set_id_offset(static_cast<size_t>(std::numeric_limits<int32_t>::max()));
        REQUIRE_THROWS_AS(bound.test(1), std::out_of_range);
        bound.set_id_offset(0);
        auto copy = bound;
        copy.set_candidate_evaluator(nullptr);
        REQUIRE(copy.is_bitmap());
        REQUIRE(copy.num_bits() == rows);
        REQUIRE(copy.test(boundary));
        REQUIRE_FALSE(copy.test(boundary + 1));
        REQUIRE(copy.test(boundary + 2));
        REQUIRE_FALSE(bound.is_bitmap());
    }
    REQUIRE(factory.created == factory.destroyed);
}
#endif

TEST_CASE("Candidate projected single preserves checked overflow, invalid maps and binding order",
          "[bitset][candidate][id_map]") {
    Factory factory;
    auto view = Descriptor(factory);
    const int32_t mapped_rows[] = {0, 1, -1};
    knowhere::IdArray projection;
    projection.SetType(knowhere::IdArray::Type::ARRAY);
    projection.Set(mapped_rows, 3);
    knowhere::BitsetView description;
    description.set_candidate_evaluator(&view, factory.values.size());
    description.set_out_ids(projection, 3);
    description.set_id_offset(std::numeric_limits<size_t>::max());
    REQUIRE(description.test(-1));
    // The legacy callback single API rejects too-wide backend IDs BEFORE
    // binding; valid-width inputs must still reject an unbound description.
    REQUIRE(description.test(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1));
    REQUIRE_THROWS_AS(description.test(1), std::invalid_argument);
    {
        auto bound = description.bind();
        const auto before = factory.evaluated.load();
        // Unsigned sum wraps to valid map index0, but callbacks require
        // checked addition and must not accidentally evaluate mapped row0.
        REQUIRE(bound.test(1));
        REQUIRE(factory.evaluated == before);
        bound.set_id_offset(0);
        REQUIRE(bound.test(0) == (factory.values[0] % 5 >= 3));
        REQUIRE(bound.test(2));
        REQUIRE(bound.test(3));
        const auto after_valid = factory.evaluated.load();
        REQUIRE(bound.test(2));
        REQUIRE(factory.evaluated == after_valid);
        bound.set_candidate_evaluator(nullptr);
        bound.set_id_offset(std::numeric_limits<size_t>::max());
        // Bitmap-only original int64 test keeps its historical unsigned
        // wrap semantics; an absent bitmap still excludes the resulting row.
        REQUIRE(bound.test(1));
    }
    REQUIRE(factory.created == factory.destroyed);
}

TEST_CASE("Candidate all-visible and all-excluded mandatory forms keep their predicate", "[bitset][candidate]") {
    Factory factory;
    auto view = Descriptor(factory);
    const int32_t ids[] = {0, 1, 2, 3};
    auto visible_bits = Mandatory(factory.values.size(), {});
    knowhere::BitsetView visible(visible_bits.data(), factory.values.size(), 0);
    visible.set_candidate_evaluator(&view);
    visible = visible.bind();
    REQUIRE(visible.count() == 0);
    REQUIRE_FALSE(visible.empty());
    REQUIRE(visible.test(ids, 4) == 8);
    REQUIRE(visible.get_first_valid_index() == 0);
    REQUIRE(visible.previous_valid_index(4).value() == 2);
    REQUIRE_FALSE(visible.range_all_filtered(0, 4));
    auto excluded_bits = Mandatory(factory.values.size(), {0, 1, 2, 3});
    knowhere::BitsetView excluded(excluded_bits.data(), factory.values.size());
    excluded.set_candidate_evaluator(&view);
    excluded = excluded.bind();
    const auto before = factory.evaluated.load();
    REQUIRE(excluded.test(ids, 4) == 15);
    REQUIRE(excluded.range_all_filtered(0, 4));
    REQUIRE(factory.evaluated == before);
    const int32_t pass[] = {0, 1, 2, 5, 6, 7};
    const int32_t reject[] = {3, 4, 8, 9};
    REQUIRE(visible.test(pass, 6) == 0);
    REQUIRE(visible.test(reject, 4) == 15);
}

TEST_CASE("Candidate bitmap-only batch preserves original tests and mapping", "[bitset][candidate][id_map]") {
    auto bits = Mandatory(OriginalIds().size(), {1, 3, 63});
    knowhere::BitsetView bitmap(bits.data(), OriginalIds().size());
    const int32_t ids[] = {0, 1, 2, 3, -1, 63, 64, 2};
    for (const size_t window : {0, 1, 31}) {
        bitmap.set_id_offset(window);
        const auto actual = bitmap.test(ids, 8);
        for (uint32_t lane = 0; lane < 8; ++lane) {
            REQUIRE(((actual >> lane) & 1) == bitmap.test(ids[lane]));
        }
    }
    const int32_t mapping[] = {3, 2, 1, -1};
    knowhere::IdArray out_ids;
    out_ids.SetType(knowhere::IdArray::Type::ARRAY);
    out_ids.Set(mapping, 4);
    bitmap.set_out_ids(out_ids, 4);
    bitmap.set_id_offset(0);
    const auto actual = bitmap.test(ids, 8);
    for (uint32_t lane = 0; lane < 8; ++lane) {
        REQUIRE(((actual >> lane) & 1) == bitmap.test(ids[lane]));
    }
    knowhere::BitsetView empty;
    REQUIRE(empty.test(ids, 8) == 255);
    REQUIRE(empty.empty());
}

TEST_CASE("Candidate count projection rejects without collapsing its public domain", "[bitset][candidate][count]") {
    Factory factory;
    auto view = Descriptor(factory);
    knowhere::BitsetView description;
    description.set_candidate_evaluator(&view, factory.values.size());
    auto filter = description.bind();
    const auto expected_size = factory.values.size();
    REQUIRE(filter.size() == expected_size);
    REQUIRE(filter.public_row_count() == expected_size);
    REQUIRE(filter.count() == 0);

    REQUIRE_THROWS_AS(filter.count_filtered_bits(0, expected_size), std::logic_error);
    REQUIRE(filter.size() == expected_size);
    REQUIRE(filter.public_row_count() == expected_size);
    REQUIRE(filter.count() == 0);

    knowhere::BitmapArray valid;
    REQUIRE_THROWS_AS(filter.count_filtered_bits(0, expected_size, valid), std::logic_error);
    REQUIRE(filter.size() == expected_size);
    REQUIRE(filter.public_row_count() == expected_size);
    REQUIRE(filter.count() == 0);
    REQUIRE_FALSE(filter.empty());
    REQUIRE(filter.get_first_valid_index() == 0);
    REQUIRE(filter.previous_valid_index(expected_size).value() == 62);
    REQUIRE_FALSE(filter.range_all_filtered(0, 1));
}

TEST_CASE("Candidate new bitmap batch rejects projection addition overflow", "[bitset][candidate][id_map]") {
    auto bits = Mandatory(OriginalIds().size(), {});
    knowhere::BitsetView bitmap(bits.data(), OriginalIds().size(), 0);
    const int32_t ids[] = {0, 1, 2, -1, std::numeric_limits<int32_t>::max()};
    bitmap.set_id_offset(std::numeric_limits<size_t>::max() - 1);
    REQUIRE(bitmap.test(ids, 5) == 31);
    const int32_t mapped[] = {0, 1};
    knowhere::IdArray out_ids;
    out_ids.SetType(knowhere::IdArray::Type::ARRAY);
    out_ids.Set(mapped, 2);
    bitmap.set_out_ids(out_ids, 2);
    REQUIRE(bitmap.test(ids, 5) == 31);
}

TEST_CASE("Candidate construction and callback errors preserve ownership", "[bitset][candidate]") {
    Factory factory;
    auto view = Descriptor(factory);
    for (int invalid = 0; invalid < 6; ++invalid) {
        auto broken = view;
        switch (invalid) {
            case 0:
                broken.abi_major = 2;
                break;
            case 1:
                --broken.struct_size;
                break;
            case 2:
                broken.context = nullptr;
                break;
            case 3:
                broken.create_worker = nullptr;
                break;
            case 4:
                broken.destroy_worker = nullptr;
                break;
            case 5:
                broken.eval_batch = nullptr;
                break;
        }
        knowhere::BitsetView filter;
        REQUIRE_THROWS_AS(filter.set_candidate_evaluator(&broken, factory.values.size()), std::invalid_argument);
        REQUIRE_THROWS_AS(knowhere::CandidateEvaluatorExecution(broken), std::invalid_argument);
    }
    knowhere::BitsetView description;
    description.set_candidate_evaluator(&view, factory.values.size());
    factory.fail_create = true;
    REQUIRE_THROWS_AS(description.bind(), std::runtime_error);
    REQUIRE(factory.created == 1);
    REQUIRE(factory.destroyed == 1);
    factory.fail_create = false;
    factory.null_create = true;
    REQUIRE_THROWS_AS(description.bind(), std::runtime_error);
    REQUIRE(factory.created == factory.destroyed);
    factory.null_create = false;
    {
        auto filter = description.bind();
        factory.fail_eval = true;
        REQUIRE_THROWS_AS(filter.test(0), std::runtime_error);
        factory.fail_eval = false;
        REQUIRE_FALSE(filter.test(0));
        factory.invalid_accepted = true;
        REQUIRE_THROWS_AS(filter.test(0), std::runtime_error);
        factory.invalid_accepted = false;
        REQUIRE_FALSE(filter.test(0));
    }
    REQUIRE(factory.created == factory.destroyed);
}

TEST_CASE("Candidate tasks independently reuse workspaces across batches", "[bitset][candidate][concurrency]") {
    for (const size_t nq : {0, 1, 8, 16}) {
        Factory factory;
        auto view = Descriptor(factory);
        knowhere::BitsetView description;
        description.set_candidate_evaluator(&view, factory.values.size());
        std::vector<std::future<bool>> tasks;
        for (size_t query = 0; query < nq; ++query) {
            tasks.push_back(std::async(std::launch::async, [&] {
                auto filter = description.bind();
                const int32_t ids[] = {0, 1, 2, 3};
                for (size_t batch = 0; batch < 8; ++batch) {
                    if (filter.test(ids, 4) != 8) {
                        return false;
                    }
                }
                return filter.callback_batches() == 8;
            }));
        }
        for (auto& task : tasks) {
            REQUIRE(task.get());
        }
        REQUIRE(factory.created == nq);
        REQUIRE(factory.destroyed == nq);
        REQUIRE(factory.batches == nq * 8);
        REQUIRE(factory.evaluated == nq * 32);
    }
}

TEST_CASE("Candidate transport foundation rejects unadapted production entry points", "[bitset][candidate][rollout]") {
    Factory factory;
    auto view = Descriptor(factory);
    knowhere::BitsetView filter;
    filter.set_candidate_evaluator(&view, factory.values.size());
    const auto expected = knowhere::Status::not_implemented;
    // Deliberately absent nodes/datasets: the filter guard must run BEFORE
    // configuration, vector access, task creation or any backend fast path.
    const knowhere::Index<knowhere::IndexNode> index;
    const auto search = index.Search(nullptr, {}, filter);
    REQUIRE(search.error() == expected);
    REQUIRE(search.what().find("deferred") != std::string::npos);
    REQUIRE(index.RangeSearch(nullptr, {}, filter).error() == expected);
    REQUIRE(index.AnnIterator(nullptr, {}, filter).error() == expected);
    REQUIRE(index.CalcDistByIDs(nullptr, filter, nullptr, 0, false).error() == expected);
    REQUIRE(knowhere::BruteForce::Search<knowhere::fp32>(nullptr, nullptr, {}, filter).error() == expected);
    REQUIRE(knowhere::BruteForce::RangeSearch<knowhere::fp32>(nullptr, nullptr, {}, filter).error() == expected);
    REQUIRE(knowhere::BruteForce::AnnIterator<knowhere::fp32>(nullptr, nullptr, {}, filter).error() == expected);
    REQUIRE(knowhere::BruteForce::AnnIteratorOnChunk<knowhere::fp32>(nullptr, nullptr, {}, filter).error() == expected);
    REQUIRE(knowhere::BruteForce::SearchWithBuf<knowhere::fp32>(nullptr, nullptr, nullptr, nullptr, {}, filter) ==
            expected);
    REQUIRE(knowhere::BruteForce::SearchOnChunkWithBuf<knowhere::fp32>(nullptr, nullptr, nullptr, nullptr, {},
                                                                       filter) == expected);
    REQUIRE(knowhere::BruteForce::SearchSparse(nullptr, nullptr, {}, filter).error() == expected);
    REQUIRE(knowhere::BruteForce::SearchSparseWithBuf(nullptr, nullptr, nullptr, nullptr, {}, filter) == expected);
    REQUIRE(factory.created == 0);
    REQUIRE(factory.evaluated == 0);
}
