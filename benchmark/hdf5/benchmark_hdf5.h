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

#pragma once

#include <assert.h>
#include <hdf5.h>

#include <unordered_set>
#include <vector>

#include "benchmark/benchmark_base.h"

/*****************************************************
 * To run this test, please download the HDF5 from
 *  https://support.hdfgroup.org/ftp/HDF5/releases/
 * and install it to /usr/local/hdf5 .
 *****************************************************/
static const char* HDF5_POSTFIX = ".hdf5";
static const char* HDF5_DATASET_TRAIN = "train";
static const char* HDF5_DATASET_TEST = "test";
static const char* HDF5_DATASET_NEIGHBORS = "neighbors";
static const char* HDF5_DATASET_DISTANCES = "distances";
static const char* HDF5_DATASET_LIMS = "lims";
static const char* HDF5_DATASET_RADIUS = "radius";

static const char* METRIC_IP_STR = "angular";
static const char* METRIC_L2_STR = "euclidean";
static const char* METRIC_COS_STR = "cosine";
static const char* METRIC_HAM_STR = "hamming";
static const char* METRIC_JAC_STR = "jaccard";

/************************************************************************************
 * https://github.com/erikbern/ann-benchmarks
 *
 * Dataset  Dimensions  Train_size  Test_size   Neighbors   Distance    Download
 * Fashion-
    MNIST   784         60,000      10,000      100         Euclidean   HDF5 (217MB)
 * GIST     960         1,000,000   1,000       100         Euclidean   HDF5 (3.6GB)
 * GloVe    100         1,183,514   10,000      100         Angular     HDF5 (463MB)
 * GloVe    200         1,183,514   10,000      100         Angular     HDF5 (918MB)
 * MNIST    784         60,000 	    10,000      100         Euclidean   HDF5 (217MB)
 * NYTimes  256         290,000     10,000      100         Angular     HDF5 (301MB)
 * SIFT     128         1,000,000   10,000      100         Euclidean   HDF5 (501MB)
 *************************************************************************************/

class Benchmark_hdf5 : public Benchmark_base {
 public:
    void
    set_ann_test_name(const char* test_name) {
        ann_test_name_ = test_name;
    }

    int32_t
    parse_name_and_dim() {
        size_t pos1, pos2;
        assert(!ann_test_name_.empty() || !"ann_test_name not set");

        pos1 = ann_test_name_.find_first_of('-', 0);
        assert(pos1 != std::string::npos);
        dataset_name_ = ann_test_name_.substr(0, pos1);

        pos2 = ann_test_name_.find_first_of('-', pos1 + 1);
        assert(pos2 != std::string::npos);
        dim_ = std::stoi(ann_test_name_.substr(pos1 + 1, pos2 - pos1 - 1));
        return (pos2 + 1);
    }

    void
    set_metric_type(const std::string& str) {
        if (str == METRIC_L2_STR || str == "l2") {
            metric_type_ = "L2";
        } else if (str == METRIC_IP_STR || str == "ip") {
            metric_type_ = "IP";
        } else if (str == METRIC_COS_STR) {
            metric_type_ = "COSINE";
        } else if (str == METRIC_HAM_STR) {
            metric_type_ = "HAMMING";
        } else if (str == METRIC_JAC_STR) {
            metric_type_ = "JACCARD";
        } else {
            assert(false);
        }
    }
    void
    parse_ann_test_name() {
        auto pos = parse_name_and_dim();
        metric_str_ = ann_test_name_.substr(pos);
        set_metric_type(metric_str_);
    }

    void
    parse_ann_test_name_with_range() {
        auto pos1 = parse_name_and_dim();
        auto pos2 = ann_test_name_.find_first_of('-', pos1);
        assert(pos2 != std::string::npos);
        metric_str_ = ann_test_name_.substr(pos1, pos2 - pos1);
        set_metric_type(metric_str_);
        assert("range" == ann_test_name_.substr(pos2 + 1));
    }

    void
    parse_ann_test_name_with_range_multi() {
        auto pos1 = parse_name_and_dim();
        auto pos2 = ann_test_name_.find_first_of('-', pos1);
        assert(pos2 != std::string::npos);
        metric_str_ = ann_test_name_.substr(pos1, pos2 - pos1);
        set_metric_type(metric_str_);
        assert("range-multi" == ann_test_name_.substr(pos2 + 1));
    }

    template <bool is_binary>
    void
    load_hdf5_data() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;
        int32_t dim;

        printf("[%.3f s] Loading HDF5 file: %s\n", get_time_diff(), ann_file_name.c_str());

        /* load train data */
        printf("[%.3f s] Loading train data\n", get_time_diff());
        if (!is_binary) {
            xb_ = hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_FLOAT, dim, nb_);
            assert(dim == dim_ || !"train dataset has incorrect dimension");
        } else {
            xb_ = hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_INTEGER, dim, nb_);
            assert(dim * 32 == dim_ || !"train dataset has incorrect dimension");
        }

        // if (metric_str_ == METRIC_IP_STR) {
        //     printf("[%.3f s] Normalizing train dataset \n", get_time_diff());
        //     normalize((float*)xb_, nb_, dim_);
        // }

        /* load test data */
        printf("[%.3f s] Loading test data\n", get_time_diff());
        if (!is_binary) {
            xq_ = hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_FLOAT, dim, nq_);
            assert(dim == dim_ || !"test dataset has incorrect dimension");
        } else {
            xq_ = hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_INTEGER, dim, nq_);
            assert(dim * 32 == dim_ || !"test dataset has incorrect dimension");
        }

        // if (metric_str_ == METRIC_IP_STR) {
        //     printf("[%.3f s] Normalizing test dataset \n", get_time_diff());
        //     normalize((float*)xq_, nq_, dim_);
        // }

        /* load ground-truth data */
        int32_t gt_nq;
        printf("[%.3f s] Loading ground truth data\n", get_time_diff());
        gt_ids_ = (int32_t*)hdf5_read(ann_file_name, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, gt_k_, gt_nq);
        assert(gt_nq == nq_ || !"incorrect nq of ground truth labels");

        gt_dist_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_DISTANCES, H5T_FLOAT, gt_k_, gt_nq);
        assert(gt_nq == nq_ || !"incorrect nq of ground truth distance");
    }

    template <bool is_binary>
    void
    load_hdf5_data_range() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;
        int32_t dim;

        printf("[%.3f s] Loading HDF5 file: %s\n", get_time_diff(), ann_file_name.c_str());

        /* load train data */
        printf("[%.3f s] Loading train data\n", get_time_diff());
        if (!is_binary) {
            xb_ = hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_FLOAT, dim, nb_);
            assert(dim == dim_ || !"train dataset has incorrect dimension");
        } else {
            xb_ = hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_INTEGER, dim, nb_);
            assert(dim * 32 == dim_ || !"train dataset has incorrect dimension");
        }

        // if (metric_str_ == METRIC_IP_STR) {
        //     printf("[%.3f s] Normalizing train dataset \n", get_time_diff());
        //     normalize((float*)xb_, nb_, dim_);
        // }

        /* load test data */
        printf("[%.3f s] Loading test data\n", get_time_diff());
        if (!is_binary) {
            xq_ = hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_FLOAT, dim, nq_);
            assert(dim == dim_ || !"test dataset has incorrect dimension");
        } else {
            xq_ = hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_INTEGER, dim, nq_);
            assert(dim * 32 == dim_ || !"test dataset has incorrect dimension");
        }

        // if (metric_str_ == METRIC_IP_STR) {
        //     printf("[%.3f s] Normalizing test dataset \n", get_time_diff());
        //     normalize((float*)xq_, nq_, dim_);
        // }

        /* load ground-truth data */
        int32_t cols, rows;
        printf("[%.3f s] Loading ground truth data\n", get_time_diff());
        gt_radius_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_RADIUS, H5T_FLOAT, cols, rows);
        assert((cols == 1 && rows == 1) || !"incorrect ground truth radius");

        gt_lims_ = (int32_t*)hdf5_read(ann_file_name, HDF5_DATASET_LIMS, H5T_INTEGER, cols, rows);
        assert((cols == nq_ + 1 && rows == 1) || !"incorrect dims of ground truth lims");

        gt_ids_ = (int32_t*)hdf5_read(ann_file_name, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, cols, rows);
        assert((cols == gt_lims_[nq_] && rows == 1) || !"incorrect dims of ground truth labels");

        gt_dist_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_DISTANCES, H5T_FLOAT, cols, rows);
        assert((cols == gt_lims_[nq_] && rows == 1) || !"incorrect dims of ground truth distances");
    }

    template <bool is_binary>
    void
    load_hdf5_data_range_multi() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;
        int32_t dim;

        printf("[%.3f s] Loading HDF5 file: %s\n", get_time_diff(), ann_file_name.c_str());

        /* load train data */
        printf("[%.3f s] Loading train data\n", get_time_diff());
        if (!is_binary) {
            xb_ = hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_FLOAT, dim, nb_);
            assert(dim == dim_ || !"train dataset has incorrect dimension");
        } else {
            xb_ = hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_INTEGER, dim, nb_);
            assert(dim * 32 == dim_ || !"train dataset has incorrect dimension");
        }

        // if (metric_str_ == METRIC_IP_STR) {
        //     printf("[%.3f s] Normalizing train dataset \n", get_time_diff());
        //     normalize((float*)xb_, nb_, dim_);
        // }

        /* load test data */
        printf("[%.3f s] Loading test data\n", get_time_diff());
        if (!is_binary) {
            xq_ = hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_FLOAT, dim, nq_);
            assert(dim == dim_ || !"test dataset has incorrect dimension");
        } else {
            xq_ = hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_INTEGER, dim, nq_);
            assert(dim * 32 == dim_ || !"test dataset has incorrect dimension");
        }

        // if (metric_str_ == METRIC_IP_STR) {
        //     printf("[%.3f s] Normalizing test dataset \n", get_time_diff());
        //     normalize((float*)xq_, nq_, dim_);
        // }

        /* load ground-truth data */
        int32_t cols, rows;
        printf("[%.3f s] Loading ground truth data\n", get_time_diff());
        gt_radius_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_RADIUS, H5T_FLOAT, cols, rows);
        assert((cols == nq_ && rows == 1) || !"incorrect ground truth radius");

        gt_lims_ = (int32_t*)hdf5_read(ann_file_name, HDF5_DATASET_LIMS, H5T_INTEGER, cols, rows);
        assert((cols == nq_ + 1 && rows == 1) || !"incorrect dims of ground truth lims");

        gt_ids_ = (int32_t*)hdf5_read(ann_file_name, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, cols, rows);
        assert((cols == gt_lims_[nq_] && rows == 1) || !"incorrect dims of ground truth labels");

        gt_dist_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_DISTANCES, H5T_FLOAT, cols, rows);
        assert((cols == gt_lims_[nq_] && rows == 1) || !"incorrect dims of ground truth distances");
    }

 protected:
    void*
    hdf5_read(const std::string& file_name, const std::string& dataset_name, H5T_class_t dataset_class, int32_t& d_out,
              int32_t& n_out) {
        hid_t file, dataset, datatype, dataspace, memspace;
        H5T_class_t t_class;      /* data type class */
        hsize_t dimsm[3];         /* memory space dimensions */
        hsize_t dims_out[2];      /* dataset dimensions */
        hsize_t count[2];         /* size of the hyperslab in the file */
        hsize_t offset[2];        /* hyperslab offset in the file */
        hsize_t count_out[3];     /* size of the hyperslab in memory */
        hsize_t offset_out[3];    /* hyperslab offset in memory */
        void* data_out = nullptr; /* output buffer */

        /* Open the file and the dataset. */
        file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);

        /* Get datatype and dataspace handles and then query
         * dataset class, order, size, rank and dimensions. */
        datatype = H5Dget_type(dataset); /* datatype handle */
        t_class = H5Tget_class(datatype);
        assert(t_class == dataset_class || !"Illegal dataset class type");

        dataspace = H5Dget_space(dataset); /* dataspace handle */
        H5Sget_simple_extent_dims(dataspace, dims_out, nullptr);
        n_out = dims_out[0];
        d_out = dims_out[1];

        /* Define hyperslab in the dataset. */
        offset[0] = offset[1] = 0;
        count[0] = dims_out[0];
        count[1] = dims_out[1];
        H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

        /* Define the memory dataspace. */
        dimsm[0] = dims_out[0];
        dimsm[1] = dims_out[1];
        dimsm[2] = 1;
        memspace = H5Screate_simple(3, dimsm, nullptr);

        /* Define memory hyperslab. */
        offset_out[0] = offset_out[1] = offset_out[2] = 0;
        count_out[0] = dims_out[0];
        count_out[1] = dims_out[1];
        count_out[2] = 1;
        H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, nullptr, count_out, nullptr);

        /* Read data from hyperslab in the file into the hyperslab in memory and display. */
        switch (t_class) {
            case H5T_INTEGER:
                data_out = new int32_t[dims_out[0] * dims_out[1]];
                H5Dread(dataset, H5T_NATIVE_INT32, memspace, dataspace, H5P_DEFAULT, data_out);
                break;
            case H5T_FLOAT:
                data_out = new float[dims_out[0] * dims_out[1]];
                H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data_out);
                break;
            default:
                printf("Illegal dataset class type\n");
                break;
        }

        /* Close/release resources. */
        H5Tclose(datatype);
        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Sclose(memspace);
        H5Fclose(file);

        return data_out;
    }

    void
    write_hdf5_dataset(hid_t file, const char* dataset_name, hid_t type_id, int32_t rows, int32_t cols,
                       const void* data) {
        hsize_t dims[2];
        dims[0] = rows;
        dims[1] = cols;
        auto dataspace = H5Screate_simple(2, dims, NULL);
        auto dataset = H5Dcreate2(file, dataset_name, type_id, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        auto err = H5Dwrite(dataset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        if (err != 0) {
            std::abort();
        }
        H5Dclose(dataset);
        H5Sclose(dataspace);
    }

    // For binary vector, dim should be divided by 32, since we use int32 to store binary vector data */
    template <bool is_binary>
    void
    hdf5_write(const char* file_name, const int32_t dim, const int32_t k, const void* xb, const int32_t nb,
               const void* xq, const int32_t nq, const void* g_ids, const void* g_dist) {
        /* Open the file and the dataset. */
        hid_t file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        /* write train dataset */
        if (!is_binary) {
            write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_FLOAT, nb, dim, xb);
        } else {
            write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_INT32, nb, dim / 32, xb);
        }

        /* write test dataset */
        if (!is_binary) {
            write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_FLOAT, nq, dim, xq);
        } else {
            write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_INT32, nq, dim / 32, xq);
        }

        /* write ground-truth labels dataset */
        write_hdf5_dataset(file, HDF5_DATASET_NEIGHBORS, H5T_NATIVE_INT32, nq, k, g_ids);

        /* write ground-truth distance dataset */
        write_hdf5_dataset(file, HDF5_DATASET_DISTANCES, H5T_NATIVE_FLOAT, nq, k, g_dist);

        /* Close/release resources. */
        H5Fclose(file);
    }

    // For binary vector, dim should be divided by 32, since we use int32 to store binary vector data */
    // Write HDF5 file with following dataset:
    //    HDF5_DATASET_RADIUS    - H5T_NATIVE_FLOAT, [1, 1]
    //    HDF5_DATASET_LIMS      - H5T_NATIVE_INT32, [1, nq+1]
    //    HDF5_DATASET_NEIGHBORS - H5T_NATIVE_INT32, [1, lims[nq]]
    //    HDF5_DATASET_DISTANCES - H5T_NATIVE_FLOAT, [1, lims[nq]]
    template <bool is_binary>
    void
    hdf5_write_range(const char* file_name, const int32_t dim, const void* xb, const int32_t nb, const void* xq,
                     const int32_t nq, const float radius, const void* g_lims, const void* g_ids, const void* g_dist) {
        /* Open the file and the dataset. */
        hid_t file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        /* write train dataset */
        if (!is_binary) {
            write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_FLOAT, nb, dim, xb);
        } else {
            write_hdf5_dataset(file, HDF5_DATASET_TRAIN, H5T_NATIVE_INT32, nb, dim / 32, xb);
        }

        /* write test dataset */
        if (!is_binary) {
            write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_FLOAT, nq, dim, xq);
        } else {
            write_hdf5_dataset(file, HDF5_DATASET_TEST, H5T_NATIVE_INT32, nq, dim / 32, xq);
        }

        /* write ground-truth radius */
        write_hdf5_dataset(file, HDF5_DATASET_RADIUS, H5T_NATIVE_FLOAT, 1, 1, &radius);

        /* write ground-truth lims dataset */
        write_hdf5_dataset(file, HDF5_DATASET_LIMS, H5T_NATIVE_INT32, 1, nq + 1, g_lims);

        /* write ground-truth labels dataset */
        write_hdf5_dataset(file, HDF5_DATASET_NEIGHBORS, H5T_NATIVE_INT32, 1, ((int32_t*)g_lims)[nq], g_ids);

        /* write ground-truth distance dataset */
        write_hdf5_dataset(file, HDF5_DATASET_DISTANCES, H5T_NATIVE_FLOAT, 1, ((int32_t*)g_lims)[nq], g_dist);

        /* Close/release resources. */
        H5Fclose(file);
    }

 protected:
    std::string ann_test_name_ = "";
    std::string dataset_name_;
    std::string metric_str_;
};
