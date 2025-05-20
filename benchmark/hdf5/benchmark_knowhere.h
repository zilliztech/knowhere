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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include "benchmark/utils.h"
#include "benchmark_hdf5.h"
#include "knowhere/binaryset.h"
#include "knowhere/config.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

static const size_t default_build_thread_num = 32;
static const size_t default_search_thread_num = 32;

namespace fs = std::filesystem;
std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kIPIndexDir = kDir + "/ip_index";
std::string kL2IndexPrefix = kL2IndexDir + "/l2";
std::string kIPIndexPrefix = kIPIndexDir + "/ip";

class Benchmark_knowhere : public Benchmark_hdf5 {
 public:
    static void
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

    static void
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

        index.Deserialize(binary_set, conf);
    }

    template <typename T>
    static std::string
    get_data_type_name() {
        if constexpr (std::is_same_v<T, knowhere::fp32>) {
            return "FP32";
        } else if constexpr (std::is_same_v<T, knowhere::fp16>) {
            return "FP16";
        } else if constexpr (std::is_same_v<T, knowhere::bf16>) {
            return "BF16";
        } else if constexpr (std::is_same_v<T, knowhere::int8>) {
            return "INT8";
        } else {
            return "";
        }
    }

    template <typename T>
    static std::string
    get_index_name(const std::string& ann_test_name, const std::string& index_type,
                   const std::vector<std::string>& params) {
        std::string params_str = "";
        for (size_t i = 0; i < params.size(); i++) {
            params_str += "_" + params[i];
        }
        if constexpr (std::is_same_v<T, knowhere::bin1>) {
            return ann_test_name + "_" + index_type + params_str + "_bin" + ".index";
        } else if constexpr (std::is_same_v<T, knowhere::fp32>) {
            return ann_test_name + "_" + index_type + params_str + "_fp32" + ".index";
        } else if constexpr (std::is_same_v<T, knowhere::fp16>) {
            return ann_test_name + "_" + index_type + params_str + "_fp16" + ".index";
        } else if constexpr (std::is_same_v<T, knowhere::bf16>) {
            return ann_test_name + "_" + index_type + params_str + "_bf16" + ".index";
        } else if constexpr (std::is_same_v<T, knowhere::int8>) {
            return ann_test_name + "_" + index_type + params_str + "_int8" + ".index";
        } else {
            assert("unknown data type");
        }
    }

    template <typename T>
    std::string
    get_index_name(const std::vector<int32_t>& params) {
        std::vector<std::string> str_params;
        for (auto param : params) {
            str_params.push_back(std::to_string(param));
        }
        return this->get_index_name<T>(ann_test_name_, index_type_, str_params);
    }

    template <typename T>
    std::string
    get_index_name(const std::vector<std::string>& params) {
        return this->get_index_name<T>(ann_test_name_, index_type_, params);
    }

    template <typename T>
    knowhere::Index<knowhere::IndexNode>
    create_index(const std::string& index_type, const std::string& index_file_name,
                 const knowhere::DataSetPtr& default_ds_ptr, const knowhere::Json& conf,
                 const std::optional<std::string>& additional_name = std::nullopt) {
        std::string additional_name_s = additional_name.value_or("");

        printf("[%.3f s] Creating %sindex \"%s\"\n", get_time_diff(), additional_name_s.c_str(), index_type.c_str());

        auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
        auto index = knowhere::IndexFactory::Instance().Create<T>(index_type, version);

        try {
            printf("[%.3f s] Reading %sindex file: %s\n", get_time_diff(), additional_name_s.c_str(),
                   index_file_name.c_str());

            read_index(index.value(), index_file_name, conf);
        } catch (...) {
            printf("[%.3f s] Building %sindex all on %ld vectors\n", get_time_diff(), additional_name_s.c_str(),
                   default_ds_ptr->GetRows());

            auto base = knowhere::ConvertToDataTypeIfNeeded<T>(default_ds_ptr);
            CALC_TIME_SPAN(index.value().Build(base, conf));

            printf("[%.3f s] Writing %sindex file: %s\n", get_time_diff(), additional_name_s.c_str(),
                   index_file_name.c_str());
            printf("Build index %s time: %.3fs \n", index.value().Type().c_str(), TDIFF_);

            write_index(index.value(), index_file_name, conf);
        }

        return index.value();
    }

    template <typename T>
    knowhere::Index<knowhere::IndexNode>
    create_index(const std::string& index_file_name, const knowhere::Json& conf) {
        auto idx = this->create_index<T>(index_type_, index_file_name, knowhere::GenDataSet(nb_, dim_, xb_), conf);
        index_ = idx;
        return idx;
    }

    knowhere::Index<knowhere::IndexNode>
    create_golden_index(const knowhere::Json& conf) {
        golden_index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;
        std::string golden_index_file_name = ann_test_name_ + "_" + golden_index_type_ + "_GOLDEN" + ".index";

        auto idx = this->create_index<knowhere::fp32>(golden_index_type_, golden_index_file_name,
                                                      knowhere::GenDataSet(nb_, dim_, xb_), conf, "golden ");
        golden_index_ = idx;
        return idx;
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
