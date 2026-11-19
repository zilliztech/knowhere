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

#ifndef DISKANN_CONFIG_H
#define DISKANN_CONFIG_H

#include "diskann_ssd_config.h"
#include "knowhere/emb_list_utils.h"
#include "navigation_config.h"

namespace knowhere {

namespace {

// Native defaults match the values Milvus used to inject before it passed DiskANN parameters through.
constexpr const CFG_INT::value_type kNativeDefaultSearchListSizeForBuild = 100;
constexpr const CFG_INT::value_type kNativeDefaultSearchBeamwidth = 16;

}  // namespace

// Native DiskANN/AiSAQ parameters. Cardinal also derives from DiskANNSSDConfig, so native defaults and the
// resolution of budgets from the actual raw data live here instead of in the shared base.
class DiskANNConfig : public DiskANNSSDConfig {
 public:
    CFG_STRING quant_type;
    // Public search spelling of search_list_size; it takes precedence when both are supplied.
    CFG_INT search_list;
    // Disk PQ bytes divided by raw vector bytes; an explicit disk_pq_dims takes precedence.
    CFG_FLOAT disk_pq_code_budget_gb_ratio;
    std::unique_ptr<NavigationConfig> navigation;

    KNOWHERE_DECLARE_CONFIG(DiskANNConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(quant_type)
            .description("navigation codec: build defaults to PQ; load reads the stored type, optionally constrained")
            .allow_empty_without_default()
            .for_train()
            .for_deserialize()
            .for_static();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_degree)
            .description("the degree of the graph index.")
            .set_default(56)
            .set_range(1, diskann::defaults::MAX_GRAPH_DEGREE)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_cache_budget_gb_ratio)
            .description("the ratio of the size reserved for the search cache to the size of the raw data.")
            .set_default(0.10f)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train()
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(disk_pq_dims)
            .description("the dimension of compressed vectors stored on the ssd, use 0 to store uncompressed data.")
            .allow_empty_without_default()
            .set_range(0, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train()
            .for_static();
        KNOWHERE_CONFIG_DECLARE_FIELD(disk_pq_code_budget_gb_ratio)
            .description("disk PQ bytes divided by raw vector bytes; explicit disk_pq_dims takes precedence.")
            .set_default(0)
            .set_range(0, 1)
            .for_train()
            .for_static();
        KNOWHERE_CONFIG_DECLARE_FIELD(beamwidth)
            .description("the maximum number of IO requests each query will issue per iteration of search code.")
            .allow_empty_without_default()
            .set_range(1, 128)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_list)
            .description("alias of search_list_size for search.")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_search()
            .for_range_search()
            .for_iterator();
        // DiskANNSSDConfig's set_default already wrote these members; an omitted value must stay empty.
        beamwidth.reset();
        disk_pq_dims.reset();
    }

    bool
    UsesDiskPQ() const {
        return disk_pq_dims.has_value() ? disk_pq_dims.value() > 0 : disk_pq_code_budget_gb_ratio.value_or(0) > 0;
    }

    // Resolve build budgets from the raw data being indexed, once its metadata is known. Absolute budgets remain
    // floors, as in max(absolute, ratio * raw size). An explicit disk_pq_dims above the dimension is clamped by the
    // backend, as before.
    void
    ResolveBuildBudgets(size_t count, size_t dimension, size_t element_size) {
        ResolveCacheBudget(count, dimension, element_size);
        navigation->ResolveBuildBudget(RawDataGB(count, dimension, element_size));
        if (!disk_pq_dims.has_value()) {
            const double bytes = dimension * static_cast<double>(element_size) * disk_pq_code_budget_gb_ratio.value();
            disk_pq_dims = bytes == 0 ? 0 : static_cast<int>(std::max(1.0, std::min<double>(dimension, bytes)));
        }
    }

    void
    ResolveCacheBudget(size_t count, size_t dimension, size_t element_size) {
        search_cache_budget_gb =
            std::max<double>(search_cache_budget_gb.value_or(0),
                             search_cache_budget_gb_ratio.value() * RawDataGB(count, dimension, element_size));
    }

    Status
    LoadSubConfig(const Json& json, PARAM_TYPE type, std::string* error) override {
        if (quant_type && (quant_type->empty() || ((type == PARAM_TYPE::TRAIN || type == PARAM_TYPE::STATIC) &&
                                                   !diskann::IsSupportedNavigationCodec(*quant_type))))
            return HandleError(error, "Unsupported DiskANN navigation codec: " + *quant_type, Status::invalid_args);
        if (type == PARAM_TYPE::TRAIN || type == PARAM_TYPE::STATIC)
            navigation = CreateNavigationConfig(quant_type.value_or("PQ"));
        return navigation ? navigation->Load(json, type, vec_field_size_gb.value_or(0), error) : Status::success;
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        switch (param_type) {
            case PARAM_TYPE::TRAIN: {
                if (!search_list_size.has_value()) {
                    search_list_size = kNativeDefaultSearchListSizeForBuild;
                }
                if (UsesDiskPQ() && (get_el_metric_type(metric_type.value_or("")).has_value() ||
                                     emb_list_offset_file_path.has_value())) {
                    return HandleError(err_msg, "disk PQ is unsupported for embedding-list exact reranking",
                                       Status::invalid_param_in_json);
                }
                break;
            }
            case PARAM_TYPE::SEARCH:
            case PARAM_TYPE::RANGE_SEARCH:
            case PARAM_TYPE::ITERATOR: {
                if (search_list.has_value()) {
                    search_list_size = search_list;
                }
                // Milvus used to force 16 for search and range search; iterators kept the backend default.
                if (!beamwidth.has_value()) {
                    beamwidth = param_type == PARAM_TYPE::ITERATOR ? diskann::defaults::DEFAULT_DISKANN_BEAMWIDTH
                                                                   : kNativeDefaultSearchBeamwidth;
                }
                break;
            }
            default:
                break;
        }
        return DiskANNSSDConfig::CheckAndAdjust(param_type, err_msg);
    }

 private:
    static double
    RawDataGB(size_t count, size_t dimension, size_t element_size) {
        return static_cast<double>(count) * dimension * element_size / (1024.0 * 1024 * 1024);
    }
};
}  // namespace knowhere
#endif /* DISKANN_CONFIG_H */
