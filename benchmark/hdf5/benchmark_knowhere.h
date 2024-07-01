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

#include <exception>
#include <vector>

#include "benchmark/utils.h"
#include "benchmark_hdf5.h"
#include "knowhere/binaryset.h"
#include "knowhere/config.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

namespace fs = std::filesystem;
std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kIPIndexDir = kDir + "/ip_index";
std::string kL2IndexPrefix = kL2IndexDir + "/l2";
std::string kIPIndexPrefix = kIPIndexDir + "/ip";

class Benchmark_knowhere : public Benchmark_hdf5 {
 public:
    void
    write_index(knowhere::Index<knowhere::IndexNode>& index, const std::string& filename, const knowhere::Json& conf) {
        FileIOWriter writer(filename);

        knowhere::BinarySet binary_set;
        index.Serialize(binary_set);

        const auto& m = binary_set.binary_map_;
        for (auto it = m.begin(); it != m.end(); ++it) {
            const std::string& name = it->first;
            size_t name_size = name.length();
            const knowhere::BinaryPtr data = it->second;
            size_t data_size = data->size;

            writer(&name_size, sizeof(name_size));
            writer(&data_size, sizeof(data_size));
            writer((void*)name.c_str(), name_size);
            writer(data->data.get(), data_size);
        }
    }

    void
    read_index(knowhere::Index<knowhere::IndexNode>& index, const std::string& filename, const knowhere::Json& conf) {
        FileIOReader reader(filename);
        int64_t file_size = reader.size();
        if (file_size < 0) {
            throw std::exception();
        }

        knowhere::BinarySet binary_set;
        int64_t offset = 0;
        while (offset < file_size) {
            size_t name_size, data_size;
            reader(&name_size, sizeof(size_t));
            offset += sizeof(size_t);
            reader(&data_size, sizeof(size_t));
            offset += sizeof(size_t);

            std::string name;
            name.resize(name_size);
            reader(name.data(), name_size);
            offset += name_size;
            auto data = new uint8_t[data_size];
            reader(data, data_size);
            offset += data_size;

            std::shared_ptr<uint8_t[]> data_ptr(data);
            binary_set.Append(name, data_ptr, data_size);
        }

        // IVFFLAT_NM should load raw data
        if (index_type_ == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT && binary_set.GetByName("RAW_DATA") == nullptr) {
            knowhere::BinaryPtr bin = std::make_shared<knowhere::Binary>();
            bin->data = std::shared_ptr<uint8_t[]>((uint8_t*)xb_);
            bin->size = dim_ * nb_ * sizeof(float);
            binary_set.Append("RAW_DATA", bin);
        }
        index.Deserialize(binary_set, conf);
    }

    std::string
    get_index_name(const std::vector<int32_t>& params) {
        std::string params_str = "";
        for (size_t i = 0; i < params.size(); i++) {
            params_str += "_" + std::to_string(params[i]);
        }
        return ann_test_name_ + "_" + index_type_ + params_str + ".index";
    }

    knowhere::Index<knowhere::IndexNode>
    create_index(const std::string& index_file_name, const knowhere::Json& conf) {
        auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
        printf("[%.3f s] Creating index \"%s\"\n", get_time_diff(), index_type_.c_str());
        index_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type_, version);

        try {
            printf("[%.3f s] Reading index file: %s\n", get_time_diff(), index_file_name.c_str());
            read_index(index_.value(), index_file_name, conf);
        } catch (...) {
            printf("[%.3f s] Building all on %d vectors\n", get_time_diff(), nb_);
            knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(nb_, dim_, xb_);
            index_.value().Build(ds_ptr, conf);

            printf("[%.3f s] Writing index file: %s\n", get_time_diff(), index_file_name.c_str());
            write_index(index_.value(), index_file_name, conf);
        }
        return index_.value();
    }

    knowhere::Index<knowhere::IndexNode>
    create_golden_index(const knowhere::Json& conf) {
        auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
        golden_index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

        std::string golden_index_file_name = ann_test_name_ + "_" + golden_index_type_ + "_GOLDEN" + ".index";
        printf("[%.3f s] Creating golden index \"%s\"\n", get_time_diff(), golden_index_type_.c_str());
        golden_index_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(golden_index_type_, version);

        try {
            printf("[%.3f s] Reading golden index file: %s\n", get_time_diff(), golden_index_file_name.c_str());
            read_index(golden_index_.value(), golden_index_file_name, conf);
        } catch (...) {
            printf("[%.3f s] Building golden index on %d vectors\n", get_time_diff(), nb_);
            knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(nb_, dim_, xb_);
            golden_index_.value().Build(ds_ptr, conf);

            printf("[%.3f s] Writing golden index file: %s\n", get_time_diff(), golden_index_file_name.c_str());
            write_index(golden_index_.value(), golden_index_file_name, conf);
        }
        return golden_index_.value();
    }

    void
    WriteRawDataToDisk(const std::string data_path, const float* raw_data, const uint32_t num, const uint32_t dim) {
        std::ofstream writer(data_path.c_str(), std::ios::binary);
        writer.write((char*)&num, sizeof(uint32_t));
        writer.write((char*)&dim, sizeof(uint32_t));
        writer.write((char*)raw_data, sizeof(float) * num * dim);
        writer.close();
    }

 protected:
    std::string index_type_;
    knowhere::Json cfg_;
    knowhere::expected<knowhere::Index<knowhere::IndexNode>> index_;

    std::string golden_index_type_;
    knowhere::Json golden_cfg_;
    knowhere::expected<knowhere::Index<knowhere::IndexNode>> golden_index_;
};
