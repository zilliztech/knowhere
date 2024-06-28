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

#ifndef CONFIG_H
#define CONFIG_H

#include <omp.h>

#include <iostream>
#include <limits>
#include <list>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "knowhere/comp/materialized_view.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "nlohmann/json.hpp"

namespace knowhere {

typedef nlohmann::json Json;

#ifndef CFG_INT
#define CFG_INT std::optional<int32_t>
#endif

#ifndef CFG_STRING
#define CFG_STRING std::optional<std::string>
#endif

#ifndef CFG_FLOAT
#define CFG_FLOAT std::optional<float>
#endif

#ifndef CFG_LIST
#define CFG_LIST std::optional<std::list<int>>
#endif

#ifndef CFG_BOOL
#define CFG_BOOL std::optional<bool>
#endif

#ifndef CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE
#define CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE std::optional<MaterializedViewSearchInfo>
#endif

#ifndef CFG_BYTES
#define CFG_BYTES std::optional<std::vector<uint8_t>>
#endif

template <typename T>
struct Entry {};

enum PARAM_TYPE {
    TRAIN = 1 << 0,
    SEARCH = 1 << 1,
    RANGE_SEARCH = 1 << 2,
    FEDER = 1 << 3,
    DESERIALIZE = 1 << 4,
    DESERIALIZE_FROM_FILE = 1 << 5,
    ITERATOR = 1 << 6,
    CLUSTER = 1 << 7,
};

template <>
struct Entry<CFG_STRING> {
    explicit Entry<CFG_STRING>(CFG_STRING* v) {
        val = v;
        type = 0x0;
        default_val = std::nullopt;
        desc = std::nullopt;
    }
    Entry<CFG_STRING>() {
        val = nullptr;
        type = 0x0;
        default_val = std::nullopt;
        desc = std::nullopt;
    }
    CFG_STRING* val;
    uint32_t type;
    std::optional<CFG_STRING::value_type> default_val;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_FLOAT> {
    explicit Entry<CFG_FLOAT>(CFG_FLOAT* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry<CFG_FLOAT>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    CFG_FLOAT* val;
    std::optional<CFG_FLOAT::value_type> default_val;
    uint32_t type;
    std::optional<std::pair<CFG_FLOAT::value_type, CFG_FLOAT::value_type>> range;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_INT> {
    explicit Entry<CFG_INT>(CFG_INT* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry<CFG_INT>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    CFG_INT* val;
    std::optional<CFG_INT::value_type> default_val;
    uint32_t type;
    std::optional<std::pair<CFG_INT::value_type, CFG_INT::value_type>> range;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_LIST> {
    explicit Entry<CFG_LIST>(CFG_LIST* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    Entry<CFG_LIST>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    CFG_LIST* val;
    std::optional<CFG_LIST::value_type> default_val;
    uint32_t type;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_BOOL> {
    explicit Entry<CFG_BOOL>(CFG_BOOL* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    Entry<CFG_BOOL>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    CFG_BOOL* val;
    std::optional<CFG_BOOL::value_type> default_val;
    uint32_t type;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE> {
    explicit Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE>(CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE* val;
    std::optional<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE::value_type> default_val;
    uint32_t type;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_BYTES> {
    explicit Entry<CFG_BYTES>(CFG_BYTES* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    Entry<CFG_BYTES>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    CFG_BYTES* val;
    std::optional<CFG_BYTES::value_type> default_val;
    uint32_t type;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <typename T>
class EntryAccess {
 public:
    EntryAccess(Entry<T>* entry) : entry(entry){};

    EntryAccess&
    set_default(const typename T::value_type dft) {
        entry->default_val = dft;
        *entry->val = dft;
        return *this;
    }

    EntryAccess&
    set_range(typename T::value_type a, typename T::value_type b) {
        entry->range = std::make_pair(a, b);
        return *this;
    }

    EntryAccess&
    allow_empty_without_default() {
        entry->allow_empty_without_default = true;
        return *this;
    }

    EntryAccess&
    description(const std::string& desc) {
        entry->desc = desc;
        return *this;
    }

    EntryAccess&
    for_train() {
        entry->type |= PARAM_TYPE::TRAIN;
        return *this;
    }

    EntryAccess&
    for_search() {
        entry->type |= PARAM_TYPE::SEARCH;
        return *this;
    }

    EntryAccess&
    for_range_search() {
        entry->type |= PARAM_TYPE::RANGE_SEARCH;
        return *this;
    }

    EntryAccess&
    for_iterator() {
        entry->type |= PARAM_TYPE::ITERATOR;
        return *this;
    }

    EntryAccess&
    for_feder() {
        entry->type |= PARAM_TYPE::FEDER;
        return *this;
    }

    EntryAccess&
    for_cluster() {
        entry->type |= PARAM_TYPE::CLUSTER;
        return *this;
    }

    EntryAccess&
    for_deserialize() {
        entry->type |= PARAM_TYPE::DESERIALIZE;
        return *this;
    }

    EntryAccess&
    for_deserialize_from_file() {
        entry->type |= PARAM_TYPE::DESERIALIZE_FROM_FILE;
        return *this;
    }

    EntryAccess&
    for_train_and_search() {
        entry->type |= PARAM_TYPE::TRAIN;
        entry->type |= PARAM_TYPE::SEARCH;
        entry->type |= PARAM_TYPE::RANGE_SEARCH;
        return *this;
    }

 private:
    Entry<T>* entry;
};

class Config {
 public:
    static Status
    FormatAndCheck(const Config& cfg, Json& json, std::string* const err_msg = nullptr);

    static Status
    Load(Config& cfg, const Json& json, PARAM_TYPE type, std::string* const err_msg = nullptr) {
        for (const auto& it : cfg.__DICT__) {
            const auto& var = it.second;

            if (const Entry<CFG_INT>* ptr = std::get_if<Entry<CFG_INT>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    if (ptr->allow_empty_without_default) {
                        continue;
                    }
                    LOG_KNOWHERE_ERROR_ << "Invalid param [" << it.first << "] in json.";
                    if (err_msg) {
                        *err_msg = std::string("invalid param ") + it.first;
                    }
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val;
                    continue;
                }
                if (!json[it.first].is_number_integer()) {
                    LOG_KNOWHERE_ERROR_ << "Type conflict in json: param [" << it.first << "] should be integer.";
                    if (err_msg) {
                        *err_msg = std::string("param ") + it.first + " should be integer";
                    }
                    return Status::type_conflict_in_json;
                }
                if (ptr->range.has_value()) {
                    if (json[it.first].get<long>() > std::numeric_limits<CFG_INT::value_type>::max()) {
                        LOG_KNOWHERE_ERROR_ << "Arithmetic overflow: param [" << it.first << "] should be at most "
                                            << std::numeric_limits<CFG_INT::value_type>::max();
                        if (err_msg) {
                            *err_msg = std::string("param ") + it.first + " should be at most 2147483647";
                        }
                        return Status::arithmetic_overflow;
                    }
                    CFG_INT::value_type v = json[it.first];
                    if (ptr->range.value().first <= v && v <= ptr->range.value().second) {
                        *ptr->val = v;
                    } else {
                        LOG_KNOWHERE_ERROR_ << "Out of range in json: param [" << it.first << "] should be in ["
                                            << ptr->range.value().first << ", " << ptr->range.value().second << "].";
                        if (err_msg) {
                            *err_msg = std::string("param ") + it.first + " out of range " + "[ " +
                                       std::to_string(ptr->range.value().first) + "," +
                                       std::to_string(ptr->range.value().second) + " ]";
                        }
                        return Status::out_of_range_in_json;
                    }
                } else {
                    *ptr->val = json[it.first];
                }
            }

            if (const Entry<CFG_FLOAT>* ptr = std::get_if<Entry<CFG_FLOAT>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    if (ptr->allow_empty_without_default) {
                        continue;
                    }
                    LOG_KNOWHERE_ERROR_ << "Invalid param [" << it.first << "] in json.";
                    if (err_msg) {
                        *err_msg = std::string("invalid param ") + it.first;
                    }

                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val;
                    continue;
                }
                if (!json[it.first].is_number()) {
                    LOG_KNOWHERE_ERROR_ << "Type conflict in json: param [" << it.first << "] should be a number.";
                    if (err_msg) {
                        *err_msg = std::string("param ") + it.first + " should be a number";
                    }

                    return Status::type_conflict_in_json;
                }
                if (ptr->range.has_value()) {
                    if (json[it.first].get<double>() > std::numeric_limits<CFG_FLOAT::value_type>::max()) {
                        LOG_KNOWHERE_ERROR_ << "Arithmetic overflow: param [" << it.first << "] should be at most "
                                            << std::numeric_limits<CFG_FLOAT::value_type>::max();
                        if (err_msg) {
                            *err_msg = std::string("param ") + it.first + " should be at most 3.402823e+38";
                        }

                        return Status::arithmetic_overflow;
                    }
                    CFG_FLOAT::value_type v = json[it.first];
                    if (ptr->range.value().first <= v && v <= ptr->range.value().second) {
                        *ptr->val = v;
                    } else {
                        LOG_KNOWHERE_ERROR_ << "Out of range in json: param [" << it.first << "] should be in ["
                                            << ptr->range.value().first << ", " << ptr->range.value().second << "].";
                        if (err_msg) {
                            *err_msg = std::string("param ") + it.first + " out of range " + "[ " +
                                       std::to_string(ptr->range.value().first) + "," +
                                       std::to_string(ptr->range.value().second) + " ]";
                        }

                        return Status::out_of_range_in_json;
                    }
                } else {
                    *ptr->val = json[it.first];
                }
            }

            if (const Entry<CFG_STRING>* ptr = std::get_if<Entry<CFG_STRING>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    if (ptr->allow_empty_without_default) {
                        continue;
                    }
                    LOG_KNOWHERE_ERROR_ << "Invalid param [" << it.first << "] in json.";
                    if (err_msg) {
                        *err_msg = std::string("invalid param ") + it.first;
                    }
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val;
                    continue;
                }
                if (!json[it.first].is_string()) {
                    LOG_KNOWHERE_ERROR_ << "Type conflict in json: param [" << it.first << "] should be a string.";
                    if (err_msg) {
                        *err_msg = std::string("param ") + it.first + " should be a string";
                    }
                    return Status::type_conflict_in_json;
                }
                *ptr->val = json[it.first];
            }

            if (const Entry<CFG_LIST>* ptr = std::get_if<Entry<CFG_LIST>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    if (ptr->allow_empty_without_default) {
                        continue;
                    }
                    LOG_KNOWHERE_ERROR_ << "Invalid param [" << it.first << "] in json.";
                    if (err_msg) {
                        *err_msg = std::string("invalid param ") + it.first;
                    }

                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val;
                    continue;
                }
                if (!json[it.first].is_array()) {
                    LOG_KNOWHERE_ERROR_ << "Type conflict in json: param [" << it.first << "] should be an array.";
                    if (err_msg) {
                        *err_msg = std::string("param ") + it.first + " should be an array";
                    }

                    return Status::type_conflict_in_json;
                }
                *ptr->val = CFG_LIST::value_type();
                for (auto&& i : json[it.first]) {
                    ptr->val->value().push_back(i);
                }
            }

            if (const Entry<CFG_BOOL>* ptr = std::get_if<Entry<CFG_BOOL>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    if (ptr->allow_empty_without_default) {
                        continue;
                    }
                    LOG_KNOWHERE_ERROR_ << "Invalid param [" << it.first << "] in json.";
                    if (err_msg) {
                        *err_msg = std::string("invalid param ") + it.first;
                    }

                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val;
                    continue;
                }
                if (!json[it.first].is_boolean()) {
                    LOG_KNOWHERE_ERROR_ << "Type conflict in json: param [" << it.first << "] should be a boolean.";
                    if (err_msg) {
                        *err_msg = std::string("param ") + it.first + " should be a boolean";
                    }

                    return Status::type_conflict_in_json;
                }
                *ptr->val = json[it.first];
            }

            if (const Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE>* ptr =
                    std::get_if<Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    if (ptr->allow_empty_without_default) {
                        continue;
                    }
                    LOG_KNOWHERE_ERROR_ << "Invalid param [" << it.first << "] in json.";
                    if (err_msg) {
                        *err_msg = std::string("invalid param ") + it.first;
                    }
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val;
                    continue;
                }
                *ptr->val = json[it.first];
            }

            if (const Entry<CFG_BYTES>* ptr = std::get_if<Entry<CFG_BYTES>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    if (ptr->allow_empty_without_default) {
                        continue;
                    }
                    LOG_KNOWHERE_ERROR_ << "Invalid param [" << it.first << "] in json.";
                    if (err_msg) {
                        *err_msg = std::string("invalid param ") + it.first;
                    }
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val;
                    continue;
                }
                if (!json[it.first].is_array()) {
                    LOG_KNOWHERE_ERROR_ << "Type conflict in json: param [" << it.first << "] should be an array.";
                    if (err_msg) {
                        *err_msg = std::string("param ") + it.first + " should be an array";
                    }
                    return Status::type_conflict_in_json;
                }
                *ptr->val = CFG_BYTES::value_type();
                for (auto&& i : json[it.first]) {
                    ptr->val->value().push_back(i);
                }
            }
        }

        if (!err_msg) {
            std::string tem_msg;
            return cfg.CheckAndAdjust(type, &tem_msg);
        }
        return cfg.CheckAndAdjust(type, err_msg);
    }

    virtual ~Config() {
    }

    using VarEntry = std::variant<Entry<CFG_STRING>, Entry<CFG_FLOAT>, Entry<CFG_INT>, Entry<CFG_LIST>, Entry<CFG_BOOL>,
                                  Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE>, Entry<CFG_BYTES>>;
    std::unordered_map<std::string, VarEntry> __DICT__;

 protected:
    inline virtual Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* const err_msg) {
        return Status::success;
    }
};

#define KNOHWERE_DECLARE_CONFIG(CONFIG) CONFIG()

#define KNOWHERE_CONFIG_DECLARE_FIELD(PARAM)                                                             \
    __DICT__[#PARAM] = knowhere::Config::VarEntry(std::in_place_type<Entry<decltype(PARAM)>>, &PARAM);   \
    EntryAccess<decltype(PARAM)> PARAM##_access(std::get_if<Entry<decltype(PARAM)>>(&__DICT__[#PARAM])); \
    PARAM##_access

const float defaultRangeFilter = 1.0f / 0.0;

class BaseConfig : public Config {
 public:
    CFG_STRING metric_type;
    CFG_INT k;
    CFG_INT num_build_thread;
    CFG_BOOL retrieve_friendly;
    CFG_STRING data_path;
    CFG_STRING index_prefix;
    // for distance metrics, we search for vectors with distance in [range_filter, radius).
    // for similarity metrics, we search for vectors with similarity in (radius, range_filter].
    CFG_FLOAT radius;
    CFG_FLOAT range_filter;
    CFG_FLOAT range_search_level;
    CFG_BOOL trace_visit;
    CFG_BOOL enable_mmap;
    CFG_BOOL enable_mmap_pop;
    CFG_BOOL for_tuning;
    CFG_BOOL shuffle_build;
    CFG_BYTES trace_id;
    CFG_BYTES span_id;
    CFG_INT trace_flags;
    CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE materialized_view_search_info;
    CFG_STRING opt_fields_path;
    CFG_FLOAT iterator_refine_ratio;
    KNOHWERE_DECLARE_CONFIG(BaseConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(metric_type)
            .set_default("L2")
            .description("metric type")
            .for_train_and_search()
            .for_iterator()
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(retrieve_friendly)
            .description("whether the index holds raw data for fast retrieval")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(data_path)
            .description("raw data path.")
            .allow_empty_without_default()
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(index_prefix)
            .description("path prefix to load or save index.")
            .allow_empty_without_default()
            .for_train()
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(k)
            .set_default(10)
            .description("search for top k similar vector.")
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(num_build_thread)
            .description("index thread limit for build.")
            .allow_empty_without_default()
            .set_range(1, std::thread::hardware_concurrency())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(radius)
            .set_default(0.0)
            .description("radius for range search")
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(range_filter)
            .set_default(defaultRangeFilter)
            .description("result filter for range search")
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(range_search_level)
            .set_default(0.01f)
            .description("control the accurancy of range search, [0.0 - 0.5], the larger the more accurate")
            .set_range(0, 0.5)
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(trace_visit)
            .set_default(false)
            .description("trace visit for feder")
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(enable_mmap)
            .set_default(false)
            .description("enable mmap for load index")
            .for_deserialize()
            .for_deserialize_from_file();
        KNOWHERE_CONFIG_DECLARE_FIELD(enable_mmap_pop)
            .set_default(false)
            .description("enable map_populate option for mmap")
            .for_deserialize()
            .for_deserialize_from_file();
        KNOWHERE_CONFIG_DECLARE_FIELD(for_tuning).set_default(false).description("for tuning").for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(shuffle_build)
            .set_default(true)
            .description("shuffle ids before index building")
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(trace_id)
            .description("trace id")
            .allow_empty_without_default()
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(span_id)
            .description("span id")
            .allow_empty_without_default()
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(trace_flags)
            .set_default(0)
            .description("trace flags")
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(materialized_view_search_info)
            .description("materialized view search info")
            .allow_empty_without_default()
            .for_search()
            .for_iterator()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(opt_fields_path)
            .description("materialized view optional fields path")
            .allow_empty_without_default()
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(iterator_refine_ratio)
            .set_default(0.5)
            .description("refine ratio for iterator")
            .for_iterator()
            .for_range_search();
    }
};
}  // namespace knowhere

#endif /* CONFIG_H */
