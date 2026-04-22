// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "aligned_file_reader.h"
#include <memory>
#include "knowhere/io_context_pool.h"
#include "knowhere/aio_context_pool.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t     file_sz;
  FileHandle   file_desc;
  io_context_t bad_ctx = (io_context_t) -1;
  std::shared_ptr<IOContextPool> io_ctx_pool_;
  std::shared_ptr<AioContextPool> ctx_pool_;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  io_context_t get_ctx() override {
    if (io_ctx_pool_ != nullptr && io_ctx_pool_->IsInitialized()) {
#ifdef WITH_IO_URING
      if (io_ctx_pool_->Backend() == IOBackend::IO_URING) {
        return reinterpret_cast<io_context_t>(io_ctx_pool_->PopUring());
      }
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
      if (io_ctx_pool_->Backend() == IOBackend::AIO) {
        return io_ctx_pool_->PopAio();
      }
#endif
    }
    return ctx_pool_->pop();
  }

  void put_ctx(io_context_t ctx) override {
    if (io_ctx_pool_ != nullptr && io_ctx_pool_->IsInitialized()) {
#ifdef WITH_IO_URING
      if (io_ctx_pool_->Backend() == IOBackend::IO_URING) {
        io_ctx_pool_->PushUring(reinterpret_cast<struct io_uring*>(ctx));
        return;
      }
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
      if (io_ctx_pool_->Backend() == IOBackend::AIO) {
        io_ctx_pool_->PushAio(ctx);
        return;
      }
#endif
    }
    ctx_pool_->push(ctx);
  }

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname) override;
  void close() override;

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false) override;

  // async reads
  void get_submitted_req (io_context_t &ctx, size_t n_ops) override;
  void submit_req(io_context_t &ctx, std::vector<AlignedRead> &read_reqs) override;

  size_t max_events_per_ctx() const override {
    if (io_ctx_pool_ != nullptr && io_ctx_pool_->IsInitialized()) {
      return io_ctx_pool_->MaxEventsPerCtx();
    }
    return ctx_pool_ == nullptr ? 0 : ctx_pool_->max_events_per_ctx();
  }
};
