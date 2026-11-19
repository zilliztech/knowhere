// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/cppcontrib/knowhere/IndexFlat.h"
#include "faiss/cppcontrib/knowhere/IndexIVFPQ.h"
#include "faiss/cppcontrib/knowhere/IndexRefine.h"
#include "faiss/cppcontrib/knowhere/index_io.h"
#include "index/ivf/ivf_wrapper.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"

namespace {

constexpr int64_t kDim = 32;
constexpr int64_t kAxes = 4;
constexpr int64_t kNlist = 4;
constexpr int64_t kTopk = 3;
// Residuals of these integer axis vectors fit a small codebook; only allow small errors from empty-centroid
// splitting during training.
constexpr float kScoreTolerance = 0.02f;
using AxisValue = std::pair<int64_t, float>;
using TestIndex = knowhere::Index<knowhere::IndexNode>;

std::vector<AxisValue>
MakeAxes(const std::vector<float>& levels) {
    std::vector<AxisValue> values;
    for (int64_t axis = 0; axis < kAxes; ++axis) {
        for (const auto level : levels) {
            values.emplace_back(axis, level);
        }
    }
    return values;
}

knowhere::DataSetPtr
MakeFp16DataSet(const std::vector<AxisValue>& values, int repetitions = 1) {
    auto raw = GenDataSet(values.size() * repetitions, kDim, kSeed);
    auto* data = const_cast<float*>(static_cast<const float*>(raw->GetTensor()));
    std::fill_n(data, raw->GetRows() * kDim, 0.0f);
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        for (size_t row = 0; row < values.size(); ++row) {
            const auto [axis, value] = values[row];
            data[(repeat * values.size() + row) * kDim + axis * (kDim / kAxes)] = value;
        }
    }
    return knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(raw);
}

struct Fp16Fixture {
    knowhere::DataSetPtr train = MakeFp16DataSet(MakeAxes({1, 2, 3, 4, 8}), 32);
    knowhere::DataSetPtr base = MakeFp16DataSet(MakeAxes({1, 2, 3, 4}));
    knowhere::DataSetPtr extra = MakeFp16DataSet(MakeAxes({8}));
    knowhere::DataSetPtr query = MakeFp16DataSet({{0, 1}, {1, 1}, {2, 1}, {3, 1}, {0, 2}, {0, -1}});
    knowhere::DataSetPtr combined;

    Fp16Fixture() {
        auto values = MakeAxes({1, 2, 3, 4});
        const auto appended = MakeAxes({8});
        values.insert(values.end(), appended.begin(), appended.end());
        combined = MakeFp16DataSet(values);
        REQUIRE(sizeof(knowhere::fp16) == 2);
        REQUIRE(train->GetRows() == 640);
        REQUIRE(base->GetRows() == 16);
        REQUIRE(extra->GetRows() == 4);
    }
};

TestIndex
CreateFp16Index() {
    auto expected = knowhere::IndexFactory::Instance().Create<knowhere::fp16>(
        knowhere::IndexEnum::INDEX_FAISS_IVFPQ, knowhere::Version::GetMaximumVersion().VersionNumber());
    REQUIRE(expected.has_value());
    return expected.value();
}

knowhere::Json
MakeBaseConfig() {
    knowhere::Json config = {{knowhere::meta::DIM, kDim},           {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
                             {knowhere::meta::TOPK, kTopk},         {knowhere::meta::NUM_BUILD_THREAD, 2},
                             {knowhere::indexparam::NLIST, kNlist}, {knowhere::indexparam::NPROBE, kNlist},
                             {knowhere::indexparam::M, 4},          {knowhere::indexparam::NBITS, 4},
                             {"use_super_kmeans", false},           {"max_empty_result_buckets", 0}};
    return config;
}

knowhere::Json
MakeConfig(const std::string& kernel, const std::string& refine_type = "") {
    auto config = MakeBaseConfig();
    config["coarse_quantizer"] = "sq_fp16_gemm";
    config["coarse_kernel"] = kernel;
    if (!refine_type.empty()) {
        config[knowhere::indexparam::REFINE] = true;
        config[knowhere::indexparam::REFINE_TYPE] = refine_type;
        // 3 x 8 candidates cover all 20 vectors present after the second Add.
        config[knowhere::indexparam::REFINE_K] = 8.0f;
    }
    return config;
}

double
ScalarIp(const knowhere::DataSet& base, int64_t id, const knowhere::DataSet& query, int64_t row) {
    const auto* x = static_cast<const knowhere::fp16*>(base.GetTensor()) + id * kDim;
    const auto* q = static_cast<const knowhere::fp16*>(query.GetTensor()) + row * kDim;
    double score = 0;
    for (int64_t dim = 0; dim < kDim; ++dim) {
        score += double(float(x[dim])) * double(float(q[dim]));
    }
    return score;
}

void
CheckKnn(const TestIndex& index, const knowhere::DataSetPtr& base, const knowhere::DataSetPtr& query,
         const knowhere::Json& config, const knowhere::BitsetView& bitset = {}) {
    auto result = index.Search(query, config, bitset);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetRows() == query->GetRows());
    REQUIRE(result.value()->GetDim() == kTopk);
    for (int64_t row = 0; row < query->GetRows(); ++row) {
        std::vector<double> expected;
        for (int64_t id = 0; id < base->GetRows(); ++id) {
            if (bitset.empty() || !bitset.test(id)) {
                expected.push_back(ScalarIp(*base, id, *query, row));
            }
        }
        std::sort(expected.begin(), expected.end(), std::greater<double>());
        std::set<int64_t> ids;
        for (int64_t rank = 0; rank < kTopk; ++rank) {
            const auto offset = row * kTopk + rank;
            const auto id = result.value()->GetIds()[offset];
            CAPTURE(row, rank, id);
            if (rank >= int64_t(expected.size())) {
                REQUIRE(id == -1);
                continue;
            }
            REQUIRE(id >= 0);
            REQUIRE(id < base->GetRows());
            REQUIRE((bitset.empty() || !bitset.test(id)));
            REQUIRE(ids.insert(id).second);
            const auto score = result.value()->GetDistance()[offset];
            REQUIRE(std::isfinite(score));
            REQUIRE(score == Catch::Approx(ScalarIp(*base, id, *query, row)).margin(kScoreTolerance));
            REQUIRE(score == Catch::Approx(expected[rank]).margin(kScoreTolerance));
            if (rank != 0) {
                REQUIRE(score <= result.value()->GetDistance()[offset - 1] + kScoreTolerance);
            }
        }
    }
}

void
CheckRange(const TestIndex& index, const knowhere::DataSetPtr& base, const knowhere::DataSetPtr& query,
           knowhere::Json config, float radius, const knowhere::BitsetView& bitset = {}) {
    config[knowhere::meta::RADIUS] = radius;
    config[knowhere::meta::RANGE_FILTER] = 100.0f;
    auto result = index.RangeSearch(query, config, bitset);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetRows() == query->GetRows());
    const auto* lims = result.value()->GetLims();
    REQUIRE(lims != nullptr);
    REQUIRE(lims[0] == 0);
    for (int64_t row = 0; row < query->GetRows(); ++row) {
        std::set<int64_t> expected;
        for (int64_t id = 0; id < base->GetRows(); ++id) {
            if ((bitset.empty() || !bitset.test(id)) && ScalarIp(*base, id, *query, row) > radius) {
                expected.insert(id);
            }
        }
        REQUIRE(lims[row + 1] >= lims[row]);
        std::set<int64_t> actual;
        for (size_t offset = lims[row]; offset < lims[row + 1]; ++offset) {
            const auto id = result.value()->GetIds()[offset];
            CAPTURE(row, id, radius);
            REQUIRE(id >= 0);
            REQUIRE(id < base->GetRows());
            REQUIRE(actual.insert(id).second);
            REQUIRE(result.value()->GetDistance()[offset] ==
                    Catch::Approx(ScalarIp(*base, id, *query, row)).margin(kScoreTolerance));
        }
        REQUIRE(actual == expected);
    }
}

void
CheckQueries(const TestIndex& index, const knowhere::DataSetPtr& base, const Fp16Fixture& data,
             const knowhere::Json& config) {
    CheckKnn(index, base, data.query, config);
    auto filtered_data = GenerateBitsetWithFirstTbitsSet(base->GetRows(), 4);
    const knowhere::BitsetView filtered(filtered_data.data(), base->GetRows());
    CheckKnn(index, base, data.query, config, filtered);
    CheckRange(index, base, data.query, config, 2.5f);
    CheckRange(index, base, data.query, config, 2.5f, filtered);
    CheckRange(index, base, data.query, config, 50.0f);
    auto excluded_data = GenerateBitsetWithFirstTbitsSet(base->GetRows(), base->GetRows());
    const knowhere::BitsetView excluded(excluded_data.data(), base->GetRows());
    CheckKnn(index, base, data.query, config, excluded);
    CheckRange(index, base, data.query, config, 2.5f, excluded);
}

std::unique_ptr<faiss::Index>
ReadStoredIndex(const knowhere::BinarySet& binary_set) {
    auto binary = binary_set.GetByName(knowhere::IndexEnum::INDEX_FAISS_IVFPQ);
    REQUIRE(binary != nullptr);
    knowhere::MemoryIOReader reader(binary->data.get(), binary->size);
    std::unique_ptr<faiss::Index> stored(faiss::cppcontrib::knowhere::read_index(&reader));
    REQUIRE(stored != nullptr);
    return stored;
}

bool
KernelAvailable(const std::string& kernel) {
    return kernel == "scalar" || (kernel == "fma" && faiss::fp16_gemm::fma_available()) ||
           (kernel == "amx" && faiss::fp16_gemm::amx_fp16_available());
}

void
CheckStoredSettings(const knowhere::BinarySet& binary_set, const knowhere::Json& config, int64_t count) {
    auto stored = ReadStoredIndex(binary_set);
    const auto* refine = dynamic_cast<const faiss::cppcontrib::knowhere::IndexRefine*>(stored.get());
    REQUIRE((refine != nullptr) == config.value("refine", false));
    const auto* pq = dynamic_cast<const faiss::cppcontrib::knowhere::IndexIVFPQ*>(refine != nullptr ? refine->base_index
                                                                                                    : stored.get());
    REQUIRE(pq != nullptr);
    REQUIRE(pq->ntotal == count);
    REQUIRE(pq->d == kDim);
    REQUIRE(pq->nlist == kNlist);
    REQUIRE(pq->metric_type == faiss::METRIC_INNER_PRODUCT);
    // The stream stores a plain FP16 SQ coarse quantizer; the GEMM settings are not serialized.
    const auto* stored_coarse = dynamic_cast<const faiss::IndexScalarQuantizer*>(pq->quantizer);
    REQUIRE(stored_coarse != nullptr);
    REQUIRE(stored_coarse->sq.qtype == faiss::ScalarQuantizer::QT_fp16);
    REQUIRE(stored_coarse->ntotal == kNlist);
    REQUIRE_FALSE(stored_coarse->use_fp16_gemm);
    if (refine != nullptr) {
        REQUIRE(refine->ntotal == count);
        REQUIRE(refine->refine_index != nullptr);
        REQUIRE(refine->refine_index->ntotal == count);
    }

    // Loading through the IVF_PQ wrapper re-enables the GEMM path with the AUTO kernel.
    auto wrapper = knowhere::IndexIVFPQWrapper::from_deserialized(std::move(stored));
    REQUIRE(wrapper != nullptr);
    auto* coarse = dynamic_cast<faiss::IndexScalarQuantizer*>(wrapper->get_base_ivf_index()->quantizer);
    REQUIRE(coarse != nullptr);
    REQUIRE(coarse->use_fp16_gemm);
    REQUIRE(coarse->fp16_gemm_kernel == faiss::fp16_gemm::GemmKernel::AUTO);
    REQUIRE(coarse->fp16_gemm_effective_kernel() ==
            faiss::fp16_gemm::resolve_kernel(faiss::fp16_gemm::GemmKernel::AUTO));
    std::vector<float> queries(kAxes * kDim, 0.0f);
    for (int64_t axis = 0; axis < kAxes; ++axis) {
        queries[axis * kDim + axis * (kDim / kAxes)] = 1.0f;
    }
    std::vector<float> distances(kAxes);
    std::vector<faiss::idx_t> labels(kAxes);
    coarse->search(kAxes, queries.data(), 1, distances.data(), labels.data());
    REQUIRE(coarse->fp16_gemm_nb_queries() == static_cast<size_t>(kAxes));
    for (const auto label : labels) {
        REQUIRE(label >= 0);
        REQUIRE(label < kNlist);
    }
}

struct TemporaryIndexFile {
    std::filesystem::path path;

    explicit TemporaryIndexFile(const knowhere::BinarySet& binary_set) {
        static std::atomic<uint64_t> sequence{0};
        path = std::filesystem::temp_directory_path() / ("knowhere_ivfpq_amx_fp16_coarse_" + std::to_string(getpid()) +
                                                         "_" + std::to_string(sequence.fetch_add(1)) + ".index");
        auto binary = binary_set.GetByName(knowhere::IndexEnum::INDEX_FAISS_IVFPQ);
        REQUIRE(binary != nullptr);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        REQUIRE(output.is_open());
        output.write(reinterpret_cast<const char*>(binary->data.get()), binary->size);
        output.close();
        REQUIRE(output.good());
    }

    ~TemporaryIndexFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

void
CheckLifecycle(const knowhere::Json& config) {
    CAPTURE(config.dump());
    const Fp16Fixture data;
    auto index = CreateFp16Index();
    const auto kernel = config["coarse_kernel"].get<std::string>();
    if (!KernelAvailable(kernel)) {
        // An explicitly requested kernel must not silently fall back to another one.
        REQUIRE(index.Train(data.train, config) != knowhere::Status::success);
        return;
    }
    REQUIRE(index.Train(data.train, config) == knowhere::Status::success);
    REQUIRE(index.Count() == 0);
    REQUIRE(index.Add(data.base, config) == knowhere::Status::success);
    REQUIRE(index.Type() == knowhere::IndexEnum::INDEX_FAISS_IVFPQ);
    REQUIRE(index.Dim() == kDim);
    REQUIRE(index.Count() == data.base->GetRows());
    CheckQueries(index, data.base, data, config);

    knowhere::BinarySet saved;
    REQUIRE(index.Serialize(saved) == knowhere::Status::success);
    CheckStoredSettings(saved, config, data.base->GetRows());

    // The load config deliberately differs from the build config: the coarse quantizer comes from the stream.
    auto load_config = config;
    load_config["coarse_quantizer"] = "flat";
    load_config["coarse_kernel"] = "auto";
    auto restored = CreateFp16Index();
    REQUIRE(restored.Deserialize(saved, load_config) == knowhere::Status::success);
    CheckQueries(restored, data.base, data, config);
    REQUIRE(restored.Add(data.extra, load_config) == knowhere::Status::success);
    REQUIRE(restored.Count() == data.combined->GetRows());
    CheckQueries(restored, data.combined, data, config);
    knowhere::BinarySet added;
    REQUIRE(restored.Serialize(added) == knowhere::Status::success);
    CheckStoredSettings(added, config, data.combined->GetRows());

    const TemporaryIndexFile file(saved);
    for (const bool mmap : {false, true}) {
        CAPTURE(mmap);
        auto loaded = CreateFp16Index();
        load_config["enable_mmap"] = mmap;
        REQUIRE(loaded.DeserializeFromFile(file.path.string(), load_config) == knowhere::Status::success);
        REQUIRE(loaded.Count() == data.base->GetRows());
        CheckQueries(loaded, data.base, data, config);
        // mmap-backed inverted lists are only checked for reads.
        if (!mmap) {
            REQUIRE(loaded.Add(data.extra, load_config) == knowhere::Status::success);
            REQUIRE(loaded.Count() == data.combined->GetRows());
            CheckQueries(loaded, data.combined, data, config);
        }
    }
}

}  // namespace

TEST_CASE("IVF_PQ FP16 GEMM coarse quantizer lifecycle with IP metric", "[ivfpq_amx][ip][fp16][indexnode]") {
    const auto kernel = GENERATE(as<std::string>{}, "scalar", "fma", "amx");
    CheckLifecycle(MakeConfig(kernel));
}

TEST_CASE("IVF_PQ FP16 GEMM coarse quantizer with refine", "[ivfpq_amx][ip][fp16][indexnode][refine]") {
    const auto kernel = GENERATE(as<std::string>{}, "scalar", "fma", "amx");
    const auto refine_type = GENERATE(as<std::string>{}, "fp16", "fp32");
    CheckLifecycle(MakeConfig(kernel, refine_type));
}

TEST_CASE("IVF_PQ keeps the flat coarse quantizer by default", "[ivfpq_amx][fp16][default]") {
    const Fp16Fixture data;
    auto index = CreateFp16Index();
    const auto config = MakeBaseConfig();
    REQUIRE(index.Train(data.train, config) == knowhere::Status::success);
    REQUIRE(index.Add(data.base, config) == knowhere::Status::success);
    knowhere::BinarySet saved;
    REQUIRE(index.Serialize(saved) == knowhere::Status::success);
    auto stored = ReadStoredIndex(saved);
    const auto* pq = dynamic_cast<const faiss::cppcontrib::knowhere::IndexIVFPQ*>(stored.get());
    REQUIRE(pq != nullptr);
    REQUIRE(dynamic_cast<const faiss::IndexScalarQuantizer*>(pq->quantizer) == nullptr);
    REQUIRE(dynamic_cast<const faiss::IndexFlat*>(pq->quantizer) != nullptr);
}

TEST_CASE("IVF_PQ rejects invalid coarse quantizer settings", "[ivfpq_amx][fp16][config]") {
    const Fp16Fixture data;
    const auto check_rejected = [&](const knowhere::Json& config) {
        CAPTURE(config.dump());
        auto index = CreateFp16Index();
        REQUIRE(index.Train(data.train, config) != knowhere::Status::success);
    };
    auto config = MakeBaseConfig();
    config["coarse_quantizer"] = "hnsw";
    check_rejected(config);
    config = MakeBaseConfig();
    config["coarse_kernel"] = "amx";
    check_rejected(config);
    config = MakeConfig("avx2");
    check_rejected(config);
}
