#pragma once
#include <thread>
#include <vector>

/**
 * @brief Read request for a single graph node.
 */
struct ReadReq{
  uint64_t key;  ///< Node ID
  uint64_t len;  ///< Data length in bytes
  void*    buf;  ///< Output buffer

  ReadReq() : key(0), len(0), buf(nullptr) {
  }

  ReadReq(uint64_t key, uint64_t len, void* buf)
      : key(key), len(len), buf(buf) {
  }
};

/**
 * @brief Abstract interface for reading graph nodes from storage.
 * 
 * Supports both synchronous and asynchronous reads. Implementations:
 * - FileIndexReader: Disk-based reads with sector alignment
 * - NCSReader: Near Compute Storage reads
 */
class IndexReader {
 public:
  virtual ~IndexReader(){};

  /// Synchronously read nodes (blocking)
  virtual void read(std::vector<ReadReq>& read_reqs) = 0;

  /// Submit async read requests (non-blocking)
  virtual void submit_req(std::vector<ReadReq> &read_reqs) = 0;

  /// Wait for async reads to complete (blocking)
  virtual void get_submitted_req() = 0;

protected:
  static inline thread_local size_t saved_n_ops;
};
