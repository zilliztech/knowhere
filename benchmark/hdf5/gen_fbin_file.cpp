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

#include "benchmark/utils.h"
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

class Create_FBIN : public Benchmark_hdf5, public ::testing::Test {
 protected:
    void
    SetUp() override {
    }

    void
    TearDown() override {
    }

    void
    fbin_write(const std::string& filename, const uint32_t rows, const uint32_t dim, const void* data) {
        FileIOWriter writer(filename);
        writer((void*)&rows, sizeof(rows));
        writer((void*)&dim, sizeof(dim));
        writer((void*)data, rows * dim * sizeof(float));
    }

    void
    fbin_write_binary(const std::string& filename, const uint32_t rows, const uint32_t dim, const void* data) {
        FileIOWriter writer(filename);
        writer((void*)&rows, sizeof(rows));
        writer((void*)&dim, sizeof(dim));
        writer((void*)data, rows * (dim / 8) * sizeof(uint8_t));
    }

    void
    fbin_read(const std::string& filename, uint32_t& rows, uint32_t& dim, void* data) {
        FileIOReader reader(filename);
        reader((void*)&rows, sizeof(rows));
        reader((void*)&dim, sizeof(dim));
        reader((void*)data, rows * dim * sizeof(float));
    }

    void
    fbin_read_binary(const std::string& filename, uint32_t& rows, uint32_t& dim, void* data) {
        FileIOReader reader(filename);
        reader((void*)&rows, sizeof(rows));
        reader((void*)&dim, sizeof(dim));
        reader((void*)data, rows * (dim / 8) * sizeof(uint8_t));
    }

    void
    fbin_result_write(const std::string& filename, const uint32_t rows, const uint32_t topk, const uint32_t* ids,
                      const float* dist) {
        FileIOWriter writer(filename);
        writer((void*)&rows, sizeof(rows));
        writer((void*)&topk, sizeof(topk));
        writer((void*)ids, rows * topk * sizeof(uint32_t));
        writer((void*)dist, rows * topk * sizeof(float));
    }

    void
    fbin_range_result_write(const std::string& filename, const uint32_t rows, const float radius, const uint32_t* lims,
                            const uint32_t* ids, const float* dist) {
        FileIOWriter writer(filename);
        writer((void*)&rows, sizeof(rows));
        writer((void*)&radius, sizeof(radius));
        writer((void*)lims, (rows + 1) * sizeof(uint32_t));
        writer((void*)ids, lims[rows] * sizeof(uint32_t));
        writer((void*)dist, lims[rows] * sizeof(float));
    }

    void
    create_fbin_files(const int64_t nb, const int64_t nq, const int64_t dim, const int64_t topk,
                      const std::vector<knowhere::MetricType>& metric_types) {
        knowhere::DataSetPtr xb_ds, xq_ds;
        xb_ds = GenDataSet(nb, dim);
        xq_ds = GenDataSet(nq, dim);

        std::string prefix = "rand-" + std::to_string(dim) + "-";
        std::string postfix = ".fbin";
        std::string filename;

        filename = prefix + "base" + postfix;
        fbin_write(filename, nb, dim, xb_ds->GetTensor());

        filename = prefix + "query" + postfix;
        fbin_write(filename, nq, dim, xq_ds->GetTensor());

        for (knowhere::MetricType metric_type : metric_types) {
            std::string metric_str = metric_type;
            transform(metric_str.begin(), metric_str.end(), metric_str.begin(), ::tolower);

            knowhere::Json json;
            json[knowhere::meta::DIM] = dim;
            json[knowhere::meta::METRIC_TYPE] = metric_type;
            json[knowhere::meta::TOPK] = topk;

            auto result = knowhere::BruteForce::Search<knowhere::fp32>(xb_ds, xq_ds, json, nullptr);
            assert(result.has_value());

            // convert golden_ids to int32
            auto elem_cnt = nq * topk;
            std::vector<uint32_t> gt_ids_int(elem_cnt);
            for (int32_t i = 0; i < elem_cnt; i++) {
                gt_ids_int[i] = result.value()->GetIds()[i];
            }

            filename = prefix + metric_str + "-gt" + postfix;
            fbin_result_write(filename, nq, topk, gt_ids_int.data(), result.value()->GetDistance());
        }
    }
};

TEST_F(Create_FBIN, CREATE_FLOAT) {
    int64_t nb = 10000;
    int64_t nq = 100;
    int64_t dim = 128;
    int64_t topk = 100;

    create_fbin_files(nb, nq, dim, topk, {knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE});
}

TEST_F(Create_FBIN, HDF5_TO_FBIN) {
    set_ann_test_name("rand-128-l2");
    parse_ann_test_name();
    load_hdf5_data<false>();

    std::string prefix = dataset_name_ + "-" + std::to_string(dim_) + "-";
    std::string postfix = ".fbin";
    std::string filename;

    filename = prefix + "base" + postfix;
    fbin_write(filename, nb_, dim_, xb_);

    filename = prefix + "query" + postfix;
    fbin_write(filename, nq_, dim_, xq_);

    filename = prefix + metric_str_ + "-gt" + postfix;
    fbin_result_write(filename, nq_, gt_k_, (uint32_t*)gt_ids_, gt_dist_);

    free_all();
}

TEST_F(Create_FBIN, HDF5_RANGE_TO_FBIN) {
    set_ann_test_name("rand-128-l2-range");
    parse_ann_test_name_with_range();
    load_hdf5_data_range<false>();

    std::string prefix = dataset_name_ + "-" + std::to_string(dim_) + "-range-";
    std::string postfix = ".fbin";
    std::string filename;

    filename = prefix + "base" + postfix;
    fbin_write(filename, nb_, dim_, xb_);

    filename = prefix + "query" + postfix;
    fbin_write(filename, nq_, dim_, xq_);

    filename = prefix + metric_str_ + "-gt" + postfix;
    fbin_range_result_write(filename, nq_, *gt_radius_, (uint32_t*)gt_lims_, (uint32_t*)gt_ids_, gt_dist_);

    free_all();
}

TEST_F(Create_FBIN, HDF5_BIN_TO_FBIN) {
    set_ann_test_name("rand-1024-hamming");
    parse_ann_test_name();
    load_hdf5_data<true>();

    std::string prefix = dataset_name_ + "-" + std::to_string(dim_) + "-";
    std::string postfix = ".fbin";
    std::string filename;

    filename = prefix + "base" + postfix;
    fbin_write_binary(filename, nb_, dim_, xb_);

    filename = prefix + "query" + postfix;
    fbin_write_binary(filename, nq_, dim_, xq_);

    filename = prefix + metric_str_ + "-gt" + postfix;
    fbin_result_write(filename, nq_, gt_k_, (uint32_t*)gt_ids_, gt_dist_);

    free_all();
}

TEST_F(Create_FBIN, HDF5_BIN_RANGE_TO_FBIN) {
    set_ann_test_name("rand-1024-hamming-range");
    parse_ann_test_name_with_range();
    load_hdf5_data_range<true>();

    std::string prefix = dataset_name_ + "-" + std::to_string(dim_) + "-range-";
    std::string postfix = ".fbin";
    std::string filename;

    filename = prefix + "base" + postfix;
    fbin_write_binary(filename, nb_, dim_, xb_);

    filename = prefix + "query" + postfix;
    fbin_write_binary(filename, nq_, dim_, xq_);

    filename = prefix + metric_str_ + "-gt" + postfix;
    fbin_range_result_write(filename, nq_, *gt_radius_, (uint32_t*)gt_lims_, (uint32_t*)gt_ids_, gt_dist_);

    free_all();
}
