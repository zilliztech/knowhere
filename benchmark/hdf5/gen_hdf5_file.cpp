// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "benchmark_hdf5.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"

knowhere::DataSetPtr
GenDataSet(int rows, int dim) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<> distrib(-1.0, 1.0);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) {
        ts[i] = (float)distrib(rng);
    }
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

knowhere::DataSetPtr
GenBinDataSet(int rows, int dim) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<> distrib(0, 255);
    int uint8_num = dim / 8;
    uint8_t* ts = new uint8_t[rows * uint8_num];
    for (int i = 0; i < rows * uint8_num; ++i) {
        ts[i] = (uint8_t)distrib(rng);
    }
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

class Create_HDF5 : public Benchmark_hdf5, public ::testing::Test {
 protected:
    void
    SetUp() override {
    }

    void
    TearDown() override {
    }

    template <bool is_binary>
    void
    create_hdf5_file(const knowhere::MetricType& metric_type, const int64_t nb, const int64_t nq, const int64_t dim,
                     const int64_t topk) {
        std::string metric_str = metric_type;
        transform(metric_str.begin(), metric_str.end(), metric_str.begin(), ::tolower);
        std::string fn = "rand-" + std::to_string(dim) + "-" + metric_str + ".hdf5";

        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric_type;
        json[knowhere::meta::TOPK] = topk;

        knowhere::DataSetPtr xb_ds, xq_ds;
        if (is_binary) {
            xb_ds = GenBinDataSet(nb, dim);
            xq_ds = GenBinDataSet(nq, dim);
        } else {
            xb_ds = GenDataSet(nb, dim);
            xq_ds = GenDataSet(nq, dim);
        }

        auto result = knowhere::BruteForce::Search<knowhere::fp32>(xb_ds, xq_ds, json, nullptr);
        assert(result.has_value());

        // convert golden_ids to int32
        auto elem_cnt = nq * topk;
        std::vector<int32_t> gt_ids_int(elem_cnt);
        for (int32_t i = 0; i < elem_cnt; i++) {
            gt_ids_int[i] = result.value()->GetIds()[i];
        }

        hdf5_write<is_binary>(fn.c_str(), dim, topk, xb_ds->GetTensor(), nb, xq_ds->GetTensor(), nq, gt_ids_int.data(),
                              result.value()->GetDistance());
    }

    template <bool is_binary>
    void
    create_range_hdf5_file(const knowhere::MetricType& metric_type, const int64_t nb, const int64_t nq,
                           const int64_t dim, const float radius) {
        std::string metric_str = metric_type;
        transform(metric_str.begin(), metric_str.end(), metric_str.begin(), ::tolower);
        std::string fn = "rand-" + std::to_string(dim) + "-" + metric_str + "-range.hdf5";

        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric_type;
        json[knowhere::meta::RADIUS] = radius;

        knowhere::DataSetPtr xb_ds, xq_ds;
        if (is_binary) {
            xb_ds = GenBinDataSet(nb, dim);
            xq_ds = GenBinDataSet(nq, dim);
        } else {
            xb_ds = GenDataSet(nb, dim);
            xq_ds = GenDataSet(nq, dim);
        }

        auto result = knowhere::BruteForce::RangeSearch<knowhere::fp32>(xb_ds, xq_ds, json, nullptr);
        assert(result.has_value());

        // convert golden_lims to int32
        std::vector<int32_t> gt_lims_int(nq + 1);
        for (int32_t i = 0; i <= nq; i++) {
            gt_lims_int[i] = result.value()->GetLims()[i];
        }

        // convert golden_ids to int32
        auto elem_cnt = result.value()->GetLims()[nq];
        std::vector<int32_t> gt_ids_int(elem_cnt);
        for (size_t i = 0; i < elem_cnt; i++) {
            gt_ids_int[i] = result.value()->GetIds()[i];
        }

        hdf5_write_range<is_binary>(fn.c_str(), dim, xb_ds->GetTensor(), nb, xq_ds->GetTensor(), nq, radius,
                                    gt_lims_int.data(), gt_ids_int.data(), result.value()->GetDistance());
    }
};

TEST_F(Create_HDF5, CREATE_FLOAT) {
    int64_t nb = 10000;
    int64_t nq = 100;
    int64_t dim = 128;
    int64_t topk = 100;

    create_hdf5_file<false>(knowhere::metric::L2, nb, nq, dim, topk);
    create_hdf5_file<false>(knowhere::metric::IP, nb, nq, dim, topk);
    create_hdf5_file<false>(knowhere::metric::COSINE, nb, nq, dim, topk);
}

TEST_F(Create_HDF5, CREATE_FLOAT_RANGE) {
    int64_t nb = 10000;
    int64_t nq = 100;
    int64_t dim = 128;

    create_range_hdf5_file<false>(knowhere::metric::L2, nb, nq, dim, 65.0);
    create_range_hdf5_file<false>(knowhere::metric::IP, nb, nq, dim, 8.7);
    create_range_hdf5_file<false>(knowhere::metric::COSINE, nb, nq, dim, 0.2);
}

TEST_F(Create_HDF5, CREATE_BINARY) {
    int64_t nb = 10000;
    int64_t nq = 100;
    int64_t dim = 1024;
    int64_t topk = 100;

    create_hdf5_file<true>(knowhere::metric::HAMMING, nb, nq, dim, topk);
    create_hdf5_file<true>(knowhere::metric::JACCARD, nb, nq, dim, topk);
}

TEST_F(Create_HDF5, CREATE_BINARY_RANGE) {
    int64_t nb = 10000;
    int64_t nq = 100;
    int64_t dim = 1024;

    create_range_hdf5_file<true>(knowhere::metric::HAMMING, nb, nq, dim, 476);
    create_range_hdf5_file<true>(knowhere::metric::JACCARD, nb, nq, dim, 0.63);
}
