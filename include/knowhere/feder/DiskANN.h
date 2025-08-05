// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace knowhere::feder::diskann {

///////////////////////////////////////////////////////////////////////////////
// index view

struct DiskANNBuildConfig {
    std::string data_path;
    uint32_t max_degree;
    uint32_t search_list_size;
    float pq_code_budget_gb;
    float build_dram_budget_gb;
    uint32_t num_threads;
    uint32_t disk_pq_dims;
    bool accelerate_build;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DiskANNBuildConfig, data_path, max_degree, search_list_size, pq_code_budget_gb,
                                   build_dram_budget_gb, num_threads, disk_pq_dims, accelerate_build)
};

class DiskANNMeta {
 public:
    DiskANNMeta() = default;

    DiskANNMeta(const std::string& data_path, const int32_t max_degree, const int32_t search_list_size,
                const float pq_code_budget_gb, const float build_dram_budget_gb, const int32_t disk_pq_dims,
                const bool accelerate_build, const int64_t num_elem, const std::vector<int64_t>& entry_points)
        : num_elem_(num_elem), entry_points_(entry_points) {
        build_params_.data_path = data_path;
        build_params_.max_degree = max_degree;
        build_params_.search_list_size = search_list_size;
        build_params_.pq_code_budget_gb = pq_code_budget_gb;
        build_params_.build_dram_budget_gb = build_dram_budget_gb;
        build_params_.disk_pq_dims = disk_pq_dims;
        build_params_.accelerate_build = accelerate_build;
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DiskANNMeta, build_params_, num_elem_, entry_points_)

 private:
    DiskANNBuildConfig build_params_;
    int64_t num_elem_;
    std::vector<int64_t> entry_points_;
};

///////////////////////////////////////////////////////////////////////////////
// search view

struct DiskANNQueryConfig {
    uint64_t k;
    uint32_t search_list_size;
    uint32_t beamwidth;
    uint32_t vectors_beamwidth;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DiskANNQueryConfig, k, search_list_size, beamwidth, vectors_beamwidth)
};

class TopCandidateInfo {
 public:
    TopCandidateInfo() = default;

    TopCandidateInfo(int64_t id, float dist) : id_(id), real_distance_from_q_(dist) {
    }

    int64_t
    GetID() {
        return id_;
    }

    float
    GetDistance() {
        return real_distance_from_q_;
    }

    const std::vector<std::pair<int64_t, float>>&
    GetNeighbors() const {
        return neighbors_;
    }

    void
    AddNeighbor(int64_t id, float distance) {
        neighbors_.emplace_back(std::make_pair(id, distance));
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(TopCandidateInfo, id_, real_distance_from_q_, neighbors_)

 private:
    int64_t id_;
    float real_distance_from_q_;
    std::vector<std::pair<int64_t, float>> neighbors_;
};

class DiskANNVisitInfo {
 public:
    DiskANNVisitInfo() = default;

    void
    SetQueryConfig(const int32_t k, const int32_t search_list_size, const int32_t beamwidth,
                   const int32_t vectors_beamwidth) {
        query_params_.k = k;
        query_params_.beamwidth = beamwidth;
        query_params_.search_list_size = search_list_size;
        query_params_.vectors_beamwidth = vectors_beamwidth;
    }

    const std::vector<TopCandidateInfo>&
    GetInfos() const {
        return infos_;
    }

    void
    AddTopCandidateInfo(int64_t id, float dist) {
        infos_.emplace_back(TopCandidateInfo(id, dist));
    }

    void
    AddTopCandidateNeighbor(int64_t id, int64_t nid, float ndist) {
        auto& curr = infos_.back();
        assert(curr.GetID() == id);
        curr.AddNeighbor(nid, ndist);
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DiskANNVisitInfo, query_params_, infos_)

 private:
    DiskANNQueryConfig query_params_;
    std::vector<TopCandidateInfo> infos_;
};

struct FederResult {
    DiskANNVisitInfo visit_info_;
    std::unordered_set<int64_t> id_set_;
};
using FederResultUniq = std::unique_ptr<FederResult>;

}  // namespace knowhere::feder::diskann
