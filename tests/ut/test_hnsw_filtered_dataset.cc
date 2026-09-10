// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {

class NpyFloatMmap {
 public:
    explicit NpyFloatMmap(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open npy file: " + path);
        }
        struct stat file_stat {};
        if (fstat(fd_, &file_stat) != 0) {
            throw std::runtime_error("cannot stat npy file: " + path);
        }
        bytes_ = static_cast<size_t>(file_stat.st_size);
        mapping_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            throw std::runtime_error("cannot mmap npy file: " + path);
        }

        const auto* raw = static_cast<const uint8_t*>(mapping_);
        if (bytes_ < 12 || std::memcmp(raw, "\x93NUMPY", 6) != 0) {
            throw std::runtime_error("invalid npy header: " + path);
        }
        const uint8_t major = raw[6];
        size_t header_size = 0;
        size_t header_offset = 0;
        if (major == 1) {
            header_size = static_cast<size_t>(raw[8]) | (static_cast<size_t>(raw[9]) << 8);
            header_offset = 10;
        } else {
            header_size = static_cast<size_t>(raw[8]) | (static_cast<size_t>(raw[9]) << 8) |
                          (static_cast<size_t>(raw[10]) << 16) | (static_cast<size_t>(raw[11]) << 24);
            header_offset = 12;
        }
        if (header_offset + header_size > bytes_) {
            throw std::runtime_error("truncated npy header: " + path);
        }
        const std::string header(reinterpret_cast<const char*>(raw + header_offset), header_size);
        if (header.find("'descr': '<f4'") == std::string::npos &&
            header.find("\"descr\": \"<f4\"") == std::string::npos) {
            throw std::runtime_error("npy array is not little-endian float32: " + path);
        }
        const size_t shape_key = header.find("shape");
        const size_t left = header.find('(', shape_key);
        const size_t comma = header.find(',', left);
        const size_t right = header.find(')', comma);
        if (shape_key == std::string::npos || left == std::string::npos || comma == std::string::npos ||
            right == std::string::npos) {
            throw std::runtime_error("unsupported npy shape: " + path);
        }
        rows_ = std::stoll(header.substr(left + 1, comma - left - 1));
        dim_ = std::stoll(header.substr(comma + 1, right - comma - 1));
        data_ = reinterpret_cast<const float*>(raw + header_offset + header_size);
        const size_t required = static_cast<size_t>(rows_) * static_cast<size_t>(dim_) * sizeof(float);
        if (header_offset + header_size + required > bytes_) {
            throw std::runtime_error("truncated npy data: " + path);
        }
    }

    ~NpyFloatMmap() {
        if (mapping_ != nullptr) {
            munmap(mapping_, bytes_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    NpyFloatMmap(const NpyFloatMmap&) = delete;
    NpyFloatMmap& operator=(const NpyFloatMmap&) = delete;

    const float*
    data() const {
        return data_;
    }

    int64_t
    rows() const {
        return rows_;
    }

    int64_t
    dim() const {
        return dim_;
    }

 private:
    int fd_ = -1;
    void* mapping_ = nullptr;
    size_t bytes_ = 0;
    const float* data_ = nullptr;
    int64_t rows_ = 0;
    int64_t dim_ = 0;
};

struct DatasetMeasurement {
    knowhere::DataSetPtr result;
    double milliseconds;
};

DatasetMeasurement
MeasureDatasetSearch(knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSetPtr& queries,
                     const knowhere::Json& config, const knowhere::BitsetView& bitset) {
    auto warmup = index.Search(queries, config, bitset);
    REQUIRE(warmup.has_value());
    knowhere::DataSetPtr result;
    const auto start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < 3; ++repetition) {
        auto current = index.Search(queries, config, bitset);
        REQUIRE(current.has_value());
        result = current.value();
    }
    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 3.0;
    return {std::move(result), elapsed};
}

void
PrintDatasetMeasurement(const char* dataset, const char* mode, double filtered_rate, const char* algorithm, int64_t ef,
                        double bridge, double threshold, const knowhere::DataSet& exact,
                        const DatasetMeasurement& measurement) {
    const float recall = GetKNNRecall(exact, *measurement.result);
    std::printf(
        "FILTERED_HNSW_DATASET dataset=%s mode=%s filtered=%.3f pass=%.3f algorithm=%s ef=%ld "
        "bridge=%.3f threshold=%.2f recall=%.4f ms=%.3f\n",
        dataset,
        mode,
        filtered_rate,
        1.0 - filtered_rate,
        algorithm,
        ef,
        bridge,
        threshold,
        recall,
        measurement.milliseconds);
}

}  // namespace

TEST_CASE("Filtered HNSW dataset benchmark", "[hnsw][filtered_dataset_benchmark]") {
    const char* base_path = std::getenv("KNOWHERE_FILTER_BENCH_BASE");
    const char* query_path = std::getenv("KNOWHERE_FILTER_BENCH_QUERY");
    if (base_path == nullptr || query_path == nullptr) {
        SKIP("set KNOWHERE_FILTER_BENCH_BASE and KNOWHERE_FILTER_BENCH_QUERY to run");
    }

    NpyFloatMmap base_file(base_path);
    NpyFloatMmap query_file(query_path);
    REQUIRE(base_file.dim() == query_file.dim());

    int64_t nb = std::min<int64_t>(base_file.rows(), 100000);
    if (const char* limit = std::getenv("KNOWHERE_FILTER_BENCH_NB")) {
        nb = std::min<int64_t>(base_file.rows(), std::stoll(limit));
    }
    int64_t nq = std::min<int64_t>(query_file.rows(), 100);
    if (const char* limit = std::getenv("KNOWHERE_FILTER_BENCH_NQ")) {
        nq = std::min<int64_t>(query_file.rows(), std::stoll(limit));
    }
    const int64_t dim = base_file.dim();
    const std::string metric = std::getenv("KNOWHERE_FILTER_BENCH_METRIC") != nullptr
                                   ? std::getenv("KNOWHERE_FILTER_BENCH_METRIC")
                                   : knowhere::metric::L2;
    constexpr int64_t topk = 10;

    const auto base = knowhere::GenDataSet(nb, dim, base_file.data());
    const auto queries = knowhere::GenDataSet(nq, dim, query_file.data());
    knowhere::Json build_config = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::indexparam::HNSW_M, 16},
        {knowhere::indexparam::EFCONSTRUCTION, 160},
    };

    auto created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, knowhere::Version::GetCurrentVersion().VersionNumber());
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index.Build(base, build_config) == knowhere::Status::success);

    const std::string dataset = std::getenv("KNOWHERE_FILTER_BENCH_NAME") != nullptr
                                    ? std::getenv("KNOWHERE_FILTER_BENCH_NAME")
                                    : "npy";
    for (const std::string mode : {"random", "id_range"}) {
        for (const double filtered_rate : {0.50, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999}) {
            const size_t filtered_count = static_cast<size_t>(nb * filtered_rate);
            const auto bitset_data = mode == "random" ? GenerateBitsetWithRandomTbitsSet(nb, filtered_count)
                                                       : GenerateBitsetWithFirstTbitsSet(nb, filtered_count);
            const knowhere::BitsetView bitset(bitset_data.data(), nb, filtered_count);
            knowhere::Json config = {
                {knowhere::meta::DIM, dim},
                {knowhere::meta::METRIC_TYPE, metric},
                {knowhere::meta::TOPK, topk},
                {knowhere::indexparam::EF, 128},
                {"use_adaptive_filter", false},
                {"kalpha_factor", 0.7},
                {"disable_fallback_brute_force", true},
            };
            auto exact = knowhere::BruteForce::Search<knowhere::fp32>(base, queries, config, bitset);
            REQUIRE(exact.has_value());

            auto automatic_config = config;
            automatic_config[knowhere::indexparam::EF] = 128;
            automatic_config["use_adaptive_filter"] = true;
            automatic_config["disable_fallback_brute_force"] = false;
            const auto automatic = MeasureDatasetSearch(index, queries, automatic_config, bitset);
            PrintDatasetMeasurement(dataset.c_str(),
                                    mode.c_str(),
                                    filtered_rate,
                                    "production_auto",
                                    128,
                                    -1.0,
                                    -1.0,
                                    *exact.value(),
                                    automatic);

            for (const int64_t ef : {32, 64, 128}) {
                auto adaptive_config = config;
                adaptive_config[knowhere::indexparam::EF] = ef;
                adaptive_config["use_adaptive_filter"] = true;
                adaptive_config["adaptive_filter_threshold"] = 0.0;
                const auto adaptive = MeasureDatasetSearch(index, queries, adaptive_config, bitset);
                PrintDatasetMeasurement(dataset.c_str(),
                                        mode.c_str(),
                                        filtered_rate,
                                        "adaptive_graph",
                                        ef,
                                        -1.0,
                                        -1.0,
                                        *exact.value(),
                                        adaptive);

                auto kalpha_config = config;
                kalpha_config[knowhere::indexparam::EF] = ef;
                const auto kalpha = MeasureDatasetSearch(index, queries, kalpha_config, bitset);
                PrintDatasetMeasurement(
                    dataset.c_str(), mode.c_str(), filtered_rate, "kalpha", ef, -1.0, -1.0, *exact.value(), kalpha);
            }
        }
    }
}
