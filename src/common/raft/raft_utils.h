#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>

#include "knowhere/log.h"
#include "raft/core/device_resources.hpp"
#include "rmm/cuda_stream_pool.hpp"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/per_device_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"
#include "thrust/optional.h"

namespace raft_utils {

inline auto
get_current_device() {
    auto result = int{};
    RAFT_CUDA_TRY(cudaGetDevice(&result));
    return result;
}

inline auto thread_counter = std::atomic<std::size_t>{};
inline auto
get_thread_id() {
    thread_local std::size_t id = ++thread_counter;
    return id;
}

// TODO(wphicks): Replace this with version from RAFT once merged
struct device_setter {
    device_setter(int new_device) : prev_device_{[]() { return get_current_device(); }()} {
        RAFT_CUDA_TRY(cudaSetDevice(new_device));
    }

    ~device_setter() {
        RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_));
    }

 private:
    int prev_device_;
};

inline auto raft_mutex = std::mutex{};

struct gpu_resources {
    gpu_resources(std::optional<std::size_t> streams_per_device = std::nullopt)
        : streams_per_device_{streams_per_device.value_or([]() {
              auto result = std::size_t{16};
              if (auto* env_str = std::getenv("KNOWHERE_STREAMS_PER_GPU")) {
                  auto str_stream = std::stringstream{env_str};
                  str_stream >> result;
                  if (str_stream.fail() || result == std::size_t{0}) {
                      LOG_KNOWHERE_WARNING_ << "KNOWHERE_STREAMS_PER_GPU env variable should be a positive integer";
                      result = std::size_t{16};
                  } else {
                      LOG_KNOWHERE_INFO_ << "streams per gpu set to " << result;
                  }
              }
              return result;
          }())},
          stream_pools_{},
          memory_resources_{},
          upstream_mr_{},
          init_mem_pool_size_{},
          max_mem_pool_size_{} {
    }
    ~gpu_resources() {
        memory_resources_.clear();
    }

    /* Set pool size in megabytes */
    void
    set_pool_size(std::size_t init_size, std::size_t max_size) {
        init_mem_pool_size_ = init_size << 20;
        max_mem_pool_size_ = max_size << 20;
    }

    void
    init(int device_id = get_current_device()) {
        auto lock = std::lock_guard{raft_mutex};
        auto stream_iter = stream_pools_.find(device_id);

        if (stream_iter == stream_pools_.end()) {
            auto scoped_device = device_setter{device_id};
            stream_pools_[device_id] = std::make_shared<rmm::cuda_stream_pool>(streams_per_device_);

            // Set up device memory pool for this device
            if (!init_mem_pool_size_.has_value() && !max_mem_pool_size_.has_value()) {
                auto* env_str = getenv("KNOWHERE_GPU_MEM_POOL_SIZE");
                if (env_str != NULL) {
                    auto init_pool_size_tmp = std::size_t{};
                    auto max_pool_size_tmp = std::size_t{};
                    auto stat = sscanf(env_str, "%zu;%zu", &init_pool_size_tmp, &max_pool_size_tmp);
                    if (stat == 2) {
                        LOG_KNOWHERE_INFO_ << "Get Gpu Pool Size From env, init size: " << init_pool_size_tmp
                                           << " MB, max size: " << max_pool_size_tmp << " MB";
                        init_mem_pool_size_ = init_pool_size_tmp << 20;
                        max_mem_pool_size_ = max_pool_size_tmp << 20;
                    } else {
                        LOG_KNOWHERE_WARNING_ << "please check env format";
                    }
                }
            }
            if (max_mem_pool_size_.value_or(std::size_t{1}) != std::size_t{}) {
                memory_resources_[device_id] =
                    std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
                        &upstream_mr_, init_mem_pool_size_, max_mem_pool_size_);
                rmm::mr::set_current_device_resource(memory_resources_[device_id].get());
            }
        }
    }

    auto
    get_streams_per_device() const {
        return streams_per_device_;
    }

    auto
    get_stream_view(int device_id = get_current_device(), std::size_t thread_id = get_thread_id()) {
        return stream_pools_[device_id]->get_stream(thread_id % streams_per_device_);
    }

 private:
    std::size_t streams_per_device_;
    std::map<int, std::shared_ptr<rmm::cuda_stream_pool>> stream_pools_;
    std::map<int, std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>> memory_resources_;
    rmm::mr::cuda_memory_resource upstream_mr_;
    thrust::optional<std::size_t> init_mem_pool_size_;
    thrust::optional<std::size_t> max_mem_pool_size_;
};

inline auto&
get_gpu_resources(std::optional<std::size_t> streams_per_device = std::nullopt) {
    static auto resources = gpu_resources{streams_per_device};
    return resources;
}

inline void
init_gpu_resources(std::optional<std::size_t> streams_per_device = std::nullopt, int device_id = get_current_device()) {
    get_gpu_resources(streams_per_device).init(device_id);
}

class gpu_device_manager {
 public:
    static gpu_device_manager&
    instance();
    int
    random_choose() const;
    int
    choose_with_load(size_t load);

 private:
    gpu_device_manager();
    std::vector<size_t> memory_load_;
    mutable std::mutex mtx_;
};

inline auto&
get_raft_resources(int device_id = get_current_device()) {
    thread_local auto all_resources = std::map<int, std::unique_ptr<raft::device_resources>>{};

    auto iter = all_resources.find(device_id);
    if (iter == all_resources.end()) {
        auto scoped_device = device_setter{device_id};
        all_resources[device_id] = std::make_unique<raft::device_resources>(
            get_gpu_resources().get_stream_view(), nullptr, rmm::mr::get_current_device_resource());
    }
    return *all_resources[device_id];
}

inline void
set_mem_pool_size(size_t init_size, size_t max_size) {
    LOG_KNOWHERE_INFO_ << "Set GPU pool size: init size " << init_size << ", max size " << max_size;
    get_gpu_resources().set_pool_size(init_size, max_size);
}

};  // namespace raft_utils

#define RANDOM_CHOOSE_DEVICE_WITH_ASSIGN(x)                             \
    do {                                                                \
        x = raft_utils::gpu_device_manager::instance().random_choose(); \
    } while (0)
#define MIN_LOAD_CHOOSE_DEVICE_WITH_ASSIGN(x, load)                            \
    do {                                                                       \
        x = raft_utils::gpu_device_manager::instance().choose_with_load(load); \
    } while (0)
