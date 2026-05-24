#pragma once

#include <mutex>
#include <queue>
#include <libaio.h>
#include <condition_variable>
#include "utils.h"
#include "concurrent_queue.h"
#include "aux_utils.h"
#include "diskann/ann_exception.h"
#include "diskann/defaults.h"
#include <iostream>
#include <fstream>
#include <string>

constexpr size_t default_max_nr = 65536;
constexpr size_t default_max_events = diskann::defaults::MAX_N_SECTOR_READS / 2;
constexpr size_t default_pool_size = default_max_nr / default_max_events;

struct AioInfo {
    long aio_nr = 0;
    long aio_max_nr = 0;
};

static AioInfo readAioInfo() {
    AioInfo info{0, 0};

    std::ifstream nrFile("/proc/sys/fs/aio-nr");
    std::ifstream maxFile("/proc/sys/fs/aio-max-nr");

    if (!nrFile.is_open()) {
        throw std::runtime_error("Failed to open /proc/sys/fs/aio-nr");
    }
    if (!maxFile.is_open()) {
        throw std::runtime_error("Failed to open /proc/sys/fs/aio-max-nr");
    }

    nrFile >> info.aio_nr;
    maxFile >> info.aio_max_nr;

    return info;
}

class AioContextPool {
 public:
  AioContextPool(const AioContextPool&) = delete;

  AioContextPool& operator=(const AioContextPool&) = delete;

  AioContextPool(AioContextPool&&) noexcept = delete;

  AioContextPool& operator==(AioContextPool&&) noexcept = delete;

  size_t max_events_per_ctx() {
    return max_events_;
  }

  void push(io_context_t ctx) {
    {
      std::scoped_lock lk(ctx_mtx_);
      ctx_q_.push(ctx);
    }
    ctx_cv_.notify_one();
  }

  io_context_t pop() {
    std::unique_lock lk(ctx_mtx_);
    if (stop_) {
      return nullptr;
    }
    ctx_cv_.wait(lk, [this] { return ctx_q_.size(); });
    if (stop_) {
      return nullptr;
    }
    auto ret = ctx_q_.front();
    ctx_q_.pop();
    return ret;
  }

  static bool InitGlobalAioPool(size_t num_ctx, size_t max_events) {
    if (num_ctx <= 0) {
      LOG(ERROR) << "num_ctx should be bigger than 0";
      return false;
    }
    if (max_events > default_max_events) {
      LOG(ERROR) << "max_events " << max_events << " should not be larger than " << default_max_events;
      return false;
    }
    if (global_aio_pool_size == 0) {
      std::scoped_lock lk(global_aio_pool_mut);
      if (global_aio_pool_size == 0) {
        global_aio_pool_size = num_ctx;
        global_aio_max_events = max_events;
        return true;
      }
    }
    LOG(WARNING)
        << "Global AioContextPool has already been inialized with context num: "
        << global_aio_pool_size;
        return true;
  }

  static std::shared_ptr<AioContextPool> GetGlobalAioPool() {
    if (global_aio_pool_size == 0) {
      std::scoped_lock lk(global_aio_pool_mut);
      if (global_aio_pool_size == 0) {
        global_aio_pool_size = default_pool_size;
        global_aio_max_events = default_max_events;
        LOG(WARNING)
            << "Global AioContextPool has not been inialized yet, init "
               "it now with context num: "
            << global_aio_pool_size;
      }
    }
    static auto pool = std::shared_ptr<AioContextPool>(
        new AioContextPool(global_aio_pool_size, global_aio_max_events));
    return pool;
  }

  ~AioContextPool() {
    stop_ = true;
    for (auto ctx : ctx_bak_) {
      io_destroy(ctx);
    }
    ctx_cv_.notify_all();
  }

 private:
  std::vector<io_context_t> ctx_bak_;
  std::queue<io_context_t>  ctx_q_;
  std::mutex                ctx_mtx_;
  std::condition_variable   ctx_cv_;
  bool                      stop_ = false;
  size_t                    num_ctx_;
  size_t                    max_events_;
  inline static size_t           global_aio_pool_size = 0;
  inline static size_t           global_aio_max_events = 0;
  inline static std::mutex       global_aio_pool_mut;

  AioContextPool(size_t num_ctx, size_t max_events)
      : num_ctx_(num_ctx), max_events_(max_events) {
	AioInfo aio_info = readAioInfo();
	size_t available_aio = (size_t)(aio_info.aio_max_nr - aio_info.aio_nr);
	if(num_ctx * max_events > available_aio) {
		LOG_KNOWHERE_INFO_ << " max events requested is too high: " << max_events << " setting to: " << available_aio / num_ctx;
	  max_events_ = available_aio / num_ctx;
		max_events = max_events_;
	}
    for (size_t i = 0; i < num_ctx_; ++i) {
      io_context_t ctx = 0;
      int          ret = -1;
      for (int retry = 0; (ret = io_setup(max_events, &ctx)) != 0 && retry < 5;
           ++retry) {
        if (-ret != EAGAIN) {
          LOG(ERROR) << "Unknown error occur in io_setup, errno: " << -ret
                     << ", " << strerror(-ret);
        }
      }
      if (ret != 0) {
        LOG(ERROR) << "io_setup() failed; returned " << ret
                   << ", errno=" << -ret << ":" << ::strerror(-ret);
      } else {
        LOG_KNOWHERE_DEBUG_ << "allocating ctx: " << ctx << " with max_events " << max_events_;
        ctx_q_.push(ctx);
        ctx_bak_.push_back(ctx);
      }
    }
  }
};
