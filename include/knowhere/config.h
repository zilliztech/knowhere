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

#include "knowhere/comp/index_param.h"
#include "knowhere/comp/materialized_view.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "nlohmann/json.hpp"

namespace knowhere {

typedef nlohmann::json Json;

#ifndef CFG_INT
#define CFG_INT std::optional<int32_t>
#endif

#ifndef CFG_INT64
#define CFG_INT64 std::optional<int64_t>
#endif

#ifndef CFG_STRING
#define CFG_STRING std::optional<std::string>
#endif

#ifndef CFG_FLOAT
#define CFG_FLOAT std::optional<float>
#endif

#ifndef CFG_BOOL
#define CFG_BOOL std::optional<bool>
#endif

#ifndef CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE
#define CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE std::optional<knowhere::MaterializedViewSearchInfo>
#endif

template <typename T>
struct Range {
    T left;
    T right;
    bool include_left;
    bool include_right;

    Range(T left, T right, bool includeLeft, bool includeRight)
        : left(left), right(right), include_left(includeLeft), include_right(includeRight) {
    }

    bool
    within(T val) {
        bool left_range_check = left < val || (include_left && left <= val);
        bool right_range_check = val < right || (include_right && val <= right);
        return left_range_check && right_range_check;
    }

    std::string
    to_string() {
        std::string left_mark = include_left ? "[" : "(";
        std::string right_mark = include_right ? "]" : ")";
        return left_mark + std::to_string(left) + ", " + std::to_string(right) + right_mark;
    }
};

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
    STATIC = 1 << 8,
};

template <>
struct Entry<CFG_STRING> {
    explicit Entry(CFG_STRING* v) {
        val = v;
        type = 0x0;
        default_val = std::nullopt;
        desc = std::nullopt;
    }
    Entry() {
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
    explicit Entry(CFG_FLOAT* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    CFG_FLOAT* val;
    std::optional<CFG_FLOAT::value_type> default_val;
    uint32_t type;
    std::optional<Range<CFG_FLOAT::value_type>> range;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_INT> {
    explicit Entry(CFG_INT* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    CFG_INT* val;
    std::optional<CFG_INT::value_type> default_val;
    uint32_t type;
    std::optional<Range<CFG_INT::value_type>> range;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_INT64> {
    explicit Entry(CFG_INT64* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    CFG_INT64* val;
    std::optional<CFG_INT64::value_type> default_val;
    uint32_t type;
    std::optional<Range<CFG_INT64::value_type>> range;
    std::optional<std::string> desc;
    bool allow_empty_without_default = false;
};

template <>
struct Entry<CFG_BOOL> {
    explicit Entry(CFG_BOOL* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    Entry() {
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
    explicit Entry(CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    Entry() {
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
    set_range(typename T::value_type a, typename T::value_type b, bool include_left = true, bool include_right = true) {
        entry->range = Range<typename T::value_type>(a, b, include_left, include_right);
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
    for_static() {
        entry->type |= PARAM_TYPE::STATIC;
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
                if (json.find(it.first) == json.end()) {
                    if (!ptr->default_val.has_value()) {
                        if (ptr->allow_empty_without_default) {
                            continue;
                        }
                        std::string msg = "param '" + it.first + "' not exist in json";
                        return HandleError(err_msg, msg, Status::invalid_param_in_json);
                    } else {
                        *ptr->val = ptr->default_val;
                        continue;
                    }
                }
                if (!json[it.first].is_number_integer()) {
                    std::string msg = "Type conflict in json: param '" + it.first + "' (" + to_string(json[it.first]) +
                                      ") should be integer";
                    return HandleError(err_msg, msg, Status::type_conflict_in_json);
                }
                if (ptr->range.has_value()) {
                    if (json[it.first].get<int64_t>() > std::numeric_limits<CFG_INT::value_type>::max()) {
                        std::string msg = "Arithmetic overflow: param '" + it.first + "' (" +
                                          to_string(json[it.first]) + ") should not bigger than " +
                                          std::to_string(std::numeric_limits<CFG_INT::value_type>::max());
                        return HandleError(err_msg, msg, Status::arithmetic_overflow);
                    }
                    CFG_INT::value_type v = json[it.first];
                    auto range_val = ptr->range.value();
                    if (range_val.within(v)) {
                        *ptr->val = v;
                    } else {
                        std::string msg = "Out of range in json: param '" + it.first + "' (" +
                                          to_string(json[it.first]) + ") should be in range " + range_val.to_string();
                        return HandleError(err_msg, msg, Status::out_of_range_in_json);
                    }
                } else {
                    *ptr->val = json[it.first];
                }
            }

            if (const Entry<CFG_INT64>* ptr = std::get_if<Entry<CFG_INT64>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end()) {
                    if (!ptr->default_val.has_value()) {
                        if (ptr->allow_empty_without_default) {
                            continue;
                        }
                        std::string msg = "param '" + it.first + "' not exist in json";
                        return HandleError(err_msg, msg, Status::invalid_param_in_json);
                    } else {
                        *ptr->val = ptr->default_val;
                        continue;
                    }
                }
                if (!json[it.first].is_number_integer()) {
                    std::string msg = "Type conflict in json: param '" + it.first + "' (" + to_string(json[it.first]) +
                                      ") should be long integer";
                    return HandleError(err_msg, msg, Status::type_conflict_in_json);
                }
                if (ptr->range.has_value()) {
                    if (json[it.first].get<int64_t>() > std::numeric_limits<CFG_INT64::value_type>::max()) {
                        std::string msg = "Arithmetic overflow: param '" + it.first + "' (" +
                                          to_string(json[it.first]) + ") should not bigger than " +
                                          std::to_string(std::numeric_limits<CFG_INT64::value_type>::max());
                        return HandleError(err_msg, msg, Status::arithmetic_overflow);
                    }
                    CFG_INT64::value_type v = json[it.first];
                    auto range_val = ptr->range.value();
                    if (range_val.within(v)) {
                        *ptr->val = v;
                    } else {
                        std::string msg = "Out of range in json: param '" + it.first + "' (" +
                                          to_string(json[it.first]) + ") should be in range " + range_val.to_string();
                        return HandleError(err_msg, msg, Status::out_of_range_in_json);
                    }
                } else {
                    *ptr->val = json[it.first];
                }
            }

            if (const Entry<CFG_FLOAT>* ptr = std::get_if<Entry<CFG_FLOAT>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end()) {
                    if (!ptr->default_val.has_value()) {
                        if (ptr->allow_empty_without_default) {
                            continue;
                        }
                        std::string msg = "param '" + it.first + "' not exist in json";
                        return HandleError(err_msg, msg, Status::invalid_param_in_json);
                    } else {
                        *ptr->val = ptr->default_val;
                        continue;
                    }
                }
                if (!json[it.first].is_number()) {
                    std::string msg = "Type conflict in json: param '" + it.first + "' (" + to_string(json[it.first]) +
                                      ") should be a number";
                    return HandleError(err_msg, msg, Status::type_conflict_in_json);
                }
                if (ptr->range.has_value()) {
                    if (json[it.first].get<double>() > std::numeric_limits<CFG_FLOAT::value_type>::max()) {
                        std::string msg = "Arithmetic overflow: param '" + it.first + "' (" +
                                          to_string(json[it.first]) + ") should not bigger than " +
                                          std::to_string(std::numeric_limits<CFG_FLOAT::value_type>::max());
                        return HandleError(err_msg, msg, Status::arithmetic_overflow);
                    }
                    CFG_FLOAT::value_type v = json[it.first];
                    auto range_val = ptr->range.value();
                    if (range_val.within(v)) {
                        *ptr->val = v;
                    } else {
                        std::string msg = "Out of range in json: param '" + it.first + "' (" +
                                          to_string(json[it.first]) + ") should be in range " + range_val.to_string();
                        return HandleError(err_msg, msg, Status::out_of_range_in_json);
                    }
                } else {
                    *ptr->val = json[it.first];
                }
            }

            if (const Entry<CFG_STRING>* ptr = std::get_if<Entry<CFG_STRING>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end()) {
                    if (!ptr->default_val.has_value()) {
                        if (ptr->allow_empty_without_default) {
                            continue;
                        }
                        std::string msg = "param [" + it.first + "] not exist in json";
                        return HandleError(err_msg, msg, Status::invalid_param_in_json);
                    } else {
                        *ptr->val = ptr->default_val;
                        continue;
                    }
                }
                if (!json[it.first].is_string()) {
                    std::string msg = "Type conflict in json: param '" + it.first + "' (" + to_string(json[it.first]) +
                                      ") should be a string";
                    return HandleError(err_msg, msg, Status::type_conflict_in_json);
                }
                *ptr->val = json[it.first];
            }

            if (const Entry<CFG_BOOL>* ptr = std::get_if<Entry<CFG_BOOL>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end()) {
                    if (!ptr->default_val.has_value()) {
                        if (ptr->allow_empty_without_default) {
                            continue;
                        }
                        std::string msg = "param '" + it.first + "' not exist in json";
                        return HandleError(err_msg, msg, Status::invalid_param_in_json);
                    } else {
                        *ptr->val = ptr->default_val;
                        continue;
                    }
                }
                if (!json[it.first].is_boolean()) {
                    std::string msg = "Type conflict in json: param '" + it.first + "' (" + to_string(json[it.first]) +
                                      ") should be a boolean";
                    return HandleError(err_msg, msg, Status::type_conflict_in_json);
                }
                *ptr->val = json[it.first];
            }

            if (const Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE>* ptr =
                    std::get_if<Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end()) {
                    if (!ptr->default_val.has_value()) {
                        if (ptr->allow_empty_without_default) {
                            continue;
                        }
                        std::string msg = "param '" + it.first + "' not exist in json";
                        return HandleError(err_msg, msg, Status::invalid_param_in_json);
                    } else {
                        *ptr->val = ptr->default_val;
                        continue;
                    }
                }
                *ptr->val = json[it.first];
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

    using VarEntry = std::variant<Entry<CFG_STRING>, Entry<CFG_FLOAT>, Entry<CFG_INT>, Entry<CFG_INT64>,
                                  Entry<CFG_BOOL>, Entry<CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE>>;
    std::unordered_map<std::string, VarEntry> __DICT__;

 protected:
    inline virtual Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* const err_msg) {
        return Status::success;
    }

    static knowhere::Status
    HandleError(std::string* error_msg, const std::string& msg, const knowhere::Status& status) {
        if (error_msg) {
            *error_msg = msg;
        }
        LOG_KNOWHERE_ERROR_ << msg;
        return status;
    }
};

#define KNOHWERE_DECLARE_CONFIG(CONFIG) CONFIG()

#define KNOWHERE_CONFIG_DECLARE_FIELD(PARAM)                                                                     \
    __DICT__[#PARAM] = knowhere::Config::VarEntry(std::in_place_type<knowhere::Entry<decltype(PARAM)>>, &PARAM); \
    knowhere::EntryAccess<decltype(PARAM)> PARAM##_access(                                                       \
        std::get_if<knowhere::Entry<decltype(PARAM)>>(&__DICT__[#PARAM]));                                       \
    PARAM##_access

const float defaultRangeFilter = 1.0f / 0.0;

class BaseConfig : public Config {
 public:
    CFG_INT64 dim;  // just used for config verify
    CFG_STRING metric_type;
    CFG_INT k;
    CFG_INT num_build_thread;
    CFG_BOOL retrieve_friendly;
    CFG_STRING data_path;
    CFG_STRING index_prefix;
    // the size of the raw vector data
    CFG_FLOAT vec_field_size_gb;
    // for distance metrics, we search for vectors with distance in [range_filter, radius).
    // for similarity metrics, we search for vectors with similarity in (radius, range_filter].
    CFG_FLOAT radius;
    CFG_INT range_search_k;
    CFG_FLOAT range_filter;
    CFG_FLOAT range_search_level;
    CFG_BOOL retain_iterator_order;
    CFG_BOOL trace_visit;
    CFG_BOOL enable_mmap;
    CFG_BOOL enable_mmap_pop;
    CFG_BOOL shuffle_build;
    CFG_STRING trace_id;
    CFG_STRING span_id;
    CFG_INT trace_flags;
    CFG_MATERIALIZED_VIEW_SEARCH_INFO_TYPE materialized_view_search_info;
    CFG_STRING opt_fields_path;
    CFG_FLOAT iterator_refine_ratio;
    /**
     * k1, b, avgdl are used by BM25 metric only.
     * - k1, b, avgdl must be provided at load time.
     * - k1 and b can be overridden at search time for SPARSE_INVERTED_INDEX
     *   but not for SPARSE_WAND.
     * - avgdl must always be provided at search time.
     */
    CFG_FLOAT bm25_k1;
    CFG_FLOAT bm25_b;
    CFG_FLOAT bm25_avgdl;
    /**
     * refine_type and refine_with_quant only used by data view index
     * - refine_type, train parameter, has several config:
     *    DATA_VIEW, not alloc extra memory in refiner
     *    FLOAT16_QUANT, keep data as float16 vector in memory in refiner
     *    BFLOAT16_QUANT, keep data as bfloat16 vector in memory in refiner
     *    UINT8_QUANT, keep data as uint8 vector in memory in refiner
     * - refine_with_quant, search parameter, whether to use quantized data to refine, faster but lost a little
     * precision
     */
    CFG_INT refine_type;
    CFG_BOOL refine_with_quant;
    /*
     * mh_lsh_band is a special parameters of BF search and MinHash index node train.
     */
    CFG_INT mh_lsh_band;
    CFG_BOOL mh_search_with_jaccard;
    CFG_INT mh_element_bit_width;
    /*
     * retrieval_ann_ratio only used for emb_list index search.
     * - A factor for top-k in the first ANNS round.
     */
    CFG_FLOAT retrieval_ann_ratio;
    KNOHWERE_DECLARE_CONFIG(BaseConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(dim).allow_empty_without_default().description("vector dim").for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(metric_type)
            .set_default("L2")
            .description("metric type")
            .for_train_and_search()
            .for_static()
            .for_iterator()
            .for_deserialize()
            .for_deserialize_from_file();
        KNOWHERE_CONFIG_DECLARE_FIELD(retrieve_friendly)
            .description("whether the index holds raw data for fast retrieval")
            .set_default(false)
            .for_static()
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
        KNOWHERE_CONFIG_DECLARE_FIELD(vec_field_size_gb)
            .description("the size (in GB) of the raw vector data.")
            .set_default(0)
            .for_train();
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
        KNOWHERE_CONFIG_DECLARE_FIELD(range_search_k)
            .set_default(-1)
            .description("limit the number of similar results returned by range_search. -1 means no limitations.")
            .set_range(-1, std::numeric_limits<CFG_INT::value_type>::max())
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
            .for_static()
            .for_deserialize()
            .for_deserialize_from_file();
        KNOWHERE_CONFIG_DECLARE_FIELD(enable_mmap_pop)
            .set_default(false)
            .description("enable map_populate option for mmap")
            .for_deserialize()
            .for_deserialize_from_file();
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
        KNOWHERE_CONFIG_DECLARE_FIELD(retain_iterator_order)
            .set_default(false)
            .description("whether the result of iterator monotonically ordered")
            .for_iterator()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(bm25_k1)
            .allow_empty_without_default()
            .set_range(0.0, 3.0)
            .description("BM25 k1 to tune the term frequency scaling factor")
            .for_train_and_search()
            .for_iterator()
            .for_deserialize()
            .for_deserialize_from_file();
        KNOWHERE_CONFIG_DECLARE_FIELD(bm25_b)
            .allow_empty_without_default()
            .set_range(0.0, 1.0)
            .description("BM25 beta to tune the document length scaling factor")
            .for_train_and_search()
            .for_iterator()
            .for_deserialize()
            .for_deserialize_from_file();
        // This must be provided in any BM25 type search request.
        // This is necessary for building/training/deserializing only if the index
        // type is WAND.
        KNOWHERE_CONFIG_DECLARE_FIELD(bm25_avgdl)
            .allow_empty_without_default()
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .description("average document length")
            .for_train_and_search()
            .for_iterator()
            .for_deserialize()
            .for_deserialize_from_file();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_type)
            .description("refiner type , no memory by default")
            .set_default(RefineType::DATA_VIEW)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_with_quant)
            .description("search parameters, whether use quantized data to refine")
            .set_default(false)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(mh_lsh_band)
            .description("param of MinHashLSH")
            .set_default(1)
            .for_train()
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(mh_element_bit_width)
            .description("sizeof(hash code), the hash element should be aligned on 8 bits")
            .set_default(8)
            .set_range(8, 256)
            .for_train()
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(mh_search_with_jaccard)
            .description("return the jaccard distance of minhash vector search or minhashlsh hit flag.")
            .set_default(false)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(retrieval_ann_ratio)
            .description("Factor for top-k in the first ANNS round, only used for emb_list")
            .set_default(1.0f)
            .set_range(0.01f, 100.0f)
            .for_search();
    }
};
}  // namespace knowhere

#endif /* CONFIG_H */
