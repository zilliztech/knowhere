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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "knowhere/log.h"

namespace knowhere {

namespace fs = std::filesystem;

/*
 * Try to obtain the number of cpu from cgroups when limited by cpu quota and period.
 * Only support cgroup v1 Linux system.
 * If failed, fallback to std::thread::hardware_concurrency().
 */
class CgroupCpuReader {
 private:
    static auto
    split(const std::string& s, char delimiter) -> std::vector<std::string> {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    // Returns cgroup path of cpu subsystem. The fields /proc/self/cgroup are separted by columns
    //  id:subsystem:path
    // The subsystem may be a comma-separated list of subsystems.
    static auto
    getCgroupCpuPath() -> fs::path {
        std::ifstream fin("/proc/self/cgroup");
        std::string line;
        while (std::getline(fin, line)) {
            auto fields = split(line, ':');
            if (fields.size() >= 3) {
                if (auto sub_sys = split(fields[1], ',');
                    std::find(sub_sys.cbegin(), sub_sys.cend(), "cpu") != sub_sys.cend()) {
                    return fields[2];
                }
            }
        }
        throw std::runtime_error("Unable to get cgroup file");
    }

    // Finds the line which filesystem_type is cgroup and superReturns the root and mount path of cgroup cpu subsystem
    // The fields /proc/self/mountinfo are separted by space
    //   (0)mount_id (1)parent_id (2)major:minor (3)root (4)mount_point (5)mount_options (6)optional_fields(var length)
    //   separator(-) filesystem_type mount_source super_options
    static auto
    getCgroupMountPath() -> std::pair<fs::path, fs::path> {
        std::ifstream fin("/proc/self/mountinfo");
        std::string line;
        while (std::getline(fin, line)) {
            auto fields = split(line, ' ');
            if (auto it = std::find(fields.cbegin(), fields.cend(), "-"); it != fields.cend()) {
                if (*std::next(it) == "cgroup") {
                    auto sub_systems = split(*std::next(it, 3), ',');
                    if (std::find(sub_systems.cbegin(), sub_systems.cend(), "cpu") != sub_systems.cend()) {
                        return {fields[3], fields[4]};
                    }
                }
            }
        }
        throw std::runtime_error("Unable to get mount info");
    }

    CgroupCpuReader() = default;
    ~CgroupCpuReader() = default;

 public:
    CgroupCpuReader(const CgroupCpuReader&) = delete;
    CgroupCpuReader(CgroupCpuReader&&) noexcept = delete;
    auto
    operator=(const CgroupCpuReader&) -> CgroupCpuReader& = delete;
    auto
    operator=(CgroupCpuReader&&) noexcept -> CgroupCpuReader& = delete;

    static auto
    GetCpuNum() -> int {
#ifdef __linux__
        try {
            auto readIntFromFile = [](const fs::path& path) -> int {
                std::ifstream fin(path);
                int ret;
                if (fin >> ret) {
                    return ret;
                }
                throw std::runtime_error("Failed to get int value from " + path.generic_string());
            };
            auto [root, mount_path] = getCgroupMountPath();

            // Get cgroup path of cpu subsystem
            // The root_path and cgroup_cpu_path sometimes maybe different, for example:
            //  root_path: /, cgroup_cpu_path: /user.slice
            // Note that the base path of fs::relative is the second argument
            auto cgroup_path = (mount_path / fs::relative(getCgroupCpuPath(), root));

            // The quota and period file contains only one int value shows its cpu quota and period in ms
            auto quota_file_path = cgroup_path / "cpu.cfs_quota_us";
            auto period_file_path = cgroup_path / "cpu.cfs_period_us";

            int quota = readIntFromFile(quota_file_path);
            // if no limit on cpu quota
            if (quota < 0) {
                return std::thread::hardware_concurrency();
            } else if (quota == 0) {
                throw std::runtime_error("Cpu quota is 0");
            }
            int period = readIntFromFile(period_file_path);
            int cpu_num = quota / period;
            return std::max(cpu_num, 1);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "Failed to get cpu num from cgroups: " << e.what()
                                  << ". Fallback to hardware concurrency";
            return std::thread::hardware_concurrency();
        }
#else
        return std::thread::hardware_concurrency();
#endif
    }
};
}  // namespace knowhere
