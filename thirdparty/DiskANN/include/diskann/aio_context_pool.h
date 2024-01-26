#pragma once

#include <mutex>
#include <queue>
#include <libaio.h>
#include <condition_variable>
#include "concurrent_queue.h"

constexpr size_t default_n_sector_read = 256;
constexpr size_t default_max_nr = 65536;
constexpr size_t default_max_events = default_n_sector_read / 2;
constexpr size_t default_pool_size = default_max_nr / default_max_events;

class AioContextPool {
 public:
  AioContextPool(const AioContextPool&) = delete;

  AioContextPool& operator=(const AioContextPool&) = delete;

  AioContextPool(AioContextPool&&) noexcept = delete;

  AioContextPool& operator==(AioContextPool&&) noexcept = delete;

  size_t max_events_per_ctx();

  void push(io_context_t ctx);

  io_context_t pop();

  static bool InitGlobalAioPool(size_t num_ctx, size_t max_events);

  static std::shared_ptr<AioContextPool> GetGlobalAioPool();

  ~AioContextPool();

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

  AioContextPool(size_t num_ctx, size_t max_events);
};
