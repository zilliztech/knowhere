#include "raft_utils.h"

namespace raft_utils {
int
gpu_device_manager::random_choose() const {
    srand(time(NULL));
    return rand() % memory_load_.size();
}

int
gpu_device_manager::choose_with_load(size_t load) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = std::min_element(memory_load_.begin(), memory_load_.end());
    *it += load;
    return std::distance(memory_load_.begin(), it);
}

gpu_device_manager::gpu_device_manager() {
    int device_counts;
    try {
        RAFT_CUDA_TRY(cudaGetDeviceCount(&device_counts));
    } catch (const raft::exception& e) {
        LOG_KNOWHERE_FATAL_ << e.what();
    }
    memory_load_.resize(device_counts);
    std::fill(memory_load_.begin(), memory_load_.end(), 0);
}

gpu_device_manager&
gpu_device_manager::instance() {
    static gpu_device_manager mgr;
    return mgr;
}

}  // namespace raft_utils
