// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "diskann/linux_aligned_file_reader.h"
#include <libaio.h>

#include <cassert>
#include <vector>
#include <cstdio>
#include <iostream>
#include <sstream>
#include "tsl/robin_map.h"
#include "knowhere/io_reader.h"
#include "diskann/utils.h"

namespace {
  static constexpr uint64_t n_retries = 10;

  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;

  void execute_io(io_context_t ctx, uint64_t maxnr, int fd,
                  const std::vector<AlignedRead> &read_reqs) {
#ifdef DEBUG
    for (auto &req : read_reqs) {
      assert(IS_ALIGNED(req.len, 512));
      // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
      assert(IS_ALIGNED(req.offset, 512));
      assert(IS_ALIGNED(req.buf, 512));
      // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

    // break-up requests into chunks of size maxnr each
    int64_t n_iters = ROUND_UP(read_reqs.size(), maxnr) / maxnr;
    for (int64_t iter = 0; iter < n_iters; iter++) {
      int64_t n_ops = std::min(read_reqs.size() - (iter * maxnr), maxnr);
      std::vector<iocb_t *>    cbs(n_ops, nullptr);
      std::vector<io_event_t>  evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (int64_t j = 0; j < n_ops; j++) {
        io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * maxnr].buf,
                      read_reqs[j + iter * maxnr].len,
                      read_reqs[j + iter * maxnr].offset);
      }

      // initialize `cbs` using `cb` array
      //

      for (auto i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      int64_t ret;
      int64_t num_submitted = 0;
      uint64_t submit_retry = 0;
      while (num_submitted < n_ops) {
        while ((ret = io_submit(ctx, n_ops - num_submitted,
                                cbs.data() + num_submitted)) < 0) {
          if (-ret != EINTR) {
            std::stringstream err;
            err << "Unknown error occur in io_submit, errno: " << -ret << ", "
                << strerror(-ret);
            throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
          }
        }
        num_submitted += ret;
        if (num_submitted < n_ops) {
          submit_retry++;
          if (submit_retry <= n_retries) {
            LOG(WARNING) << "io_submit() failed; submit: " << num_submitted
                         << ", expected: " << n_ops
                         << ", retry: " << submit_retry;
          } else {
            std::stringstream err;
            err << "io_submit failed after retried " << n_retries << " times";
            throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
          }
        }
      }

      int64_t num_read = 0;
      uint64_t read_retry = 0;
      while (num_read < n_ops) {
        while ((ret = io_getevents(ctx, n_ops - num_read, n_ops - num_read,
                                   evts.data() + num_read, nullptr)) < 0) {
          if (-ret != EINTR) {
            std::stringstream err;
            err << "Unknown error occur in io_getevents, errno: " << -ret
                << ", " << strerror(-ret);
            throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
          }
        }
        num_read += ret;
        if (num_read < n_ops) {
          read_retry++;
          if (read_retry <= n_retries) {
            LOG(WARNING) << "io_getevents() failed; read: " << num_read
                         << ", expected: " << n_ops
                         << ", retry: " << read_retry;
          } else {
            std::stringstream err;
            err << "io_getevents failed after retried " << n_retries
                << " times";
            throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
          }
        }
      }
      // disabled since req.buf could be an offset into another buf
      /*
      for (auto &req : read_reqs) {
        // corruption check
        assert(malloc_usable_size(req.buf) >= req.len);
      }
      */
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
  this->io_ctx_pool_ = IOContextPool::GetGlobal();
#ifdef MILVUS_COMMON_WITH_LIBAIO
  this->ctx_pool_ = AioContextPool::GetGlobalAioPool();
#endif
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                  << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

void LinuxAlignedFileReader::open(const std::string &fname) {
  int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
  this->file_desc = ::open(fname.c_str(), flags);
  // error checks
  assert(this->file_desc != -1);
  LOG_KNOWHERE_DEBUG_ << "Opened file : " << fname;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                  io_context_t &ctx, bool async) {
  if (async == true) {
    diskann::cout << "Async currently not supported in linux." << std::endl;
  }
  assert(this->file_desc != -1);

#ifdef WITH_IO_URING
  if (io_ctx_pool_ != nullptr && io_ctx_pool_->IsInitialized() &&
      io_ctx_pool_->Backend() == IOBackend::IO_URING) {
    auto* ring = reinterpret_cast<struct io_uring*>(ctx);
    if (ring == nullptr) {
      throw diskann::ANNException("io_uring ring context is null", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    size_t submitted = 0;
    const size_t n_ops = read_reqs.size();
    while (submitted < n_ops) {
      auto* sqe = io_uring_get_sqe(ring);
      if (sqe == nullptr) {
        const int rc = io_uring_submit(ring);
        if (rc < 0) {
          std::stringstream err;
          err << "io_uring_submit failed in read, rc=" << rc;
          throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        continue;
      }
      io_uring_prep_read(sqe, this->file_desc, read_reqs[submitted].buf,
                         read_reqs[submitted].len, read_reqs[submitted].offset);
      sqe->user_data = submitted;
      ++submitted;
    }
    const int submit_rc = io_uring_submit(ring);
    if (submit_rc < 0) {
      std::stringstream err;
      err << "io_uring_submit failed in read flush, rc=" << submit_rc;
      throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    size_t completed = 0;
    while (completed < n_ops) {
      struct io_uring_cqe* cqe = nullptr;
      const int wait_rc = io_uring_wait_cqe(ring, &cqe);
      if (wait_rc < 0 || cqe == nullptr) {
        std::stringstream err;
        err << "io_uring_wait_cqe failed in read, rc=" << wait_rc;
        throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      if (cqe->res < 0) {
        const int cqe_res = cqe->res;
        io_uring_cqe_seen(ring, cqe);
        std::stringstream err;
        err << "io_uring cqe read failed in read, res=" << cqe_res;
        throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      io_uring_cqe_seen(ring, cqe);
      ++completed;
    }
    return;
  }
#endif

  execute_io(ctx, this->max_events_per_ctx(), this->file_desc, read_reqs);
}

void LinuxAlignedFileReader::submit_req(io_context_t             &ctx,
                                        std::vector<AlignedRead> &read_reqs) {
  const auto maxnr = this->max_events_per_ctx();
  if (read_reqs.size() > maxnr) {
    std::stringstream err;
    err << "Async does not support number of read requests ("
        << read_reqs.size() << ") exceeds max number of events per context ("
        << maxnr << ")";
    throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
  const auto n_ops = read_reqs.size();
  if (n_ops == 0) {
    return;
  }

#ifdef WITH_IO_URING
  if (io_ctx_pool_ != nullptr && io_ctx_pool_->IsInitialized() &&
      io_ctx_pool_->Backend() == IOBackend::IO_URING) {
    auto* ring = reinterpret_cast<struct io_uring*>(ctx);
    if (ring == nullptr) {
      throw diskann::ANNException("io_uring ring context is null", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    size_t submitted = 0;
    while (submitted < n_ops) {
      auto* sqe = io_uring_get_sqe(ring);
      if (sqe == nullptr) {
        const auto rc = io_uring_submit(ring);
        if (rc < 0) {
          std::stringstream err;
          err << "io_uring_submit failed during submit_req, rc=" << rc;
          throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        continue;
      }
      io_uring_prep_read(sqe, this->file_desc, read_reqs[submitted].buf,
                         read_reqs[submitted].len, read_reqs[submitted].offset);
      sqe->user_data = submitted;
      ++submitted;
    }
    const auto rc = io_uring_submit(ring);
    if (rc < 0) {
      std::stringstream err;
      err << "io_uring_submit failed at end of submit_req, rc=" << rc;
      throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    return;
  }
#endif

  const int fd = this->file_desc;
  std::vector<iocb_t *>    cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (size_t j = 0; j < n_ops; j++) {
    io_prep_pread(cb.data() + j, fd, read_reqs[j].buf, read_reqs[j].len,
                  read_reqs[j].offset);
  }
  for (uint64_t i = 0; i < n_ops; i++) {
    cbs[i] = cb.data() + i;
  }

  int64_t ret;
  uint64_t num_submitted = 0, submit_retry = 0;
  while (num_submitted < n_ops) {
    while ((ret = io_submit(ctx, n_ops - num_submitted,
                            cbs.data() + num_submitted)) < 0) {
      if (-ret != EINTR) {
        std::stringstream err;
        err << "Unknown error occur in io_submit, errno: " << -ret << ", "
            << strerror(-ret);
        throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }
    num_submitted += ret;
    if (num_submitted < n_ops) {
      submit_retry++;
      if (submit_retry <= n_retries) {
        LOG(WARNING) << "io_submit() failed; submit: " << num_submitted
                     << ", expected: " << n_ops << ", retry: " << submit_retry;
      } else {
        std::stringstream err;
        err << "io_submit failed after retried " << n_retries << " times";
        throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }
  }
}

void LinuxAlignedFileReader::get_submitted_req(io_context_t &ctx, size_t n_ops) {
  if (n_ops > this->max_events_per_ctx()) {
    std::stringstream err;
    err << "Async does not support getting number of read requests (" << n_ops
        << ") exceeds max number of events per context ("
        << this->max_events_per_ctx() << ")";
    throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

#ifdef WITH_IO_URING
  if (io_ctx_pool_ != nullptr && io_ctx_pool_->IsInitialized() &&
      io_ctx_pool_->Backend() == IOBackend::IO_URING) {
    auto* ring = reinterpret_cast<struct io_uring*>(ctx);
    if (ring == nullptr) {
      throw diskann::ANNException("io_uring ring context is null", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    size_t completed = 0;
    while (completed < n_ops) {
      struct io_uring_cqe* cqe = nullptr;
      const int rc = io_uring_wait_cqe(ring, &cqe);
      if (rc < 0 || cqe == nullptr) {
        std::stringstream err;
        err << "io_uring_wait_cqe failed in get_submitted_req, rc=" << rc;
        throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      if (cqe->res < 0) {
        const int cqe_res = cqe->res;
        io_uring_cqe_seen(ring, cqe);
        std::stringstream err;
        err << "io_uring cqe read failed in get_submitted_req, res=" << cqe_res;
        throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      io_uring_cqe_seen(ring, cqe);
      ++completed;
    }
    return;
  }
#endif

  int64_t ret;
  uint64_t                 num_read = 0, read_retry = 0;
  std::vector<io_event_t> evts(n_ops);
  while (num_read < n_ops) {
    while ((ret = io_getevents(ctx, n_ops - num_read, n_ops - num_read,
                               evts.data() + num_read, nullptr)) < 0) {
      if (-ret != EINTR) {
        std::stringstream err;
        err << "Unknown error occur in io_getevents, errno: " << -ret << ", "
            << strerror(-ret);
        throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }
    num_read += ret;
    if (num_read < n_ops) {
      read_retry++;
      if (read_retry <= n_retries) {
        LOG(WARNING) << "io_getevents() failed; read: " << num_read
                     << ", expected: " << n_ops << ", retry: " << read_retry;
      } else {
        std::stringstream err;
        err << "io_getevents failed after retried " << n_retries << " times";
        throw diskann::ANNException(err.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }
  }
}