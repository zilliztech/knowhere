#pragma once

#include "index_reader.h"
#include "ncs/ncs.h"
#include "ncs/InMemNcsConnector.h"
#include "ncs/InMemoryNcs.h"
#include <memory>
#include <optional>
#include <future>
#include <unordered_map>

/**
 * @brief Near Compute Storage (NCS) implementation of IndexReader.
 * 
 * Reads DiskANN graph nodes from NCS (Near Compute Storage) - a distributed
 * storage system that provides fast access to index data. Uses multi-get
 * operations to batch node requests efficiently.
 * 
 * ## Thread Safety and Concurrency Model
 * 
 * NCSReader uses a **thread_local connector** model for thread safety:
 * - Each thread that accesses NCSReader gets its own NcsConnector instance
 * - Connectors are created lazily on first access per thread
 * - Connectors are stored in a thread_local map, keyed by NCSReader instance
 * - This eliminates the need for connection pooling in NcsConnector implementations
 * 
 * ### Why thread_local?
 * - NcsConnector implementations (e.g., Redis) may not be thread-safe
 * - Connection pooling adds complexity and potential contention
 * - thread_local provides natural thread-safety with zero synchronization overhead
 * 
 * ### Usage Notes
 * - A single NCSReader instance can be used concurrently from multiple threads
 * - Each thread will transparently get its own connector
 * - Memory is released when the NCSReader is destroyed
 */
class NCSReader : public IndexReader {
private:
    // Descriptor stored for creating thread-local connectors
    milvus::NcsDescriptor descriptor_;
    
    // Thread-local storage for per-thread connectors
    // Key: NCSReader instance pointer, Value: connector for this reader
    static inline thread_local std::unordered_map<const NCSReader*, std::unique_ptr<milvus::NcsConnector>> 
        thread_local_connectors_;
    
    // Async read future
    static inline thread_local std::optional<std::future<void>> async_read_future_;

    /**
     * @brief Get or create the connector for the current thread.
     * 
     * Creates a new connector on first access for each thread.
     * The connector is stored in thread_local storage and reused for subsequent calls.
     * 
     * @return Pointer to the thread's connector
     * @throws std::runtime_error if connector creation fails
     */
    milvus::NcsConnector* getThreadLocalConnector();

public:
    NCSReader(const milvus::NcsDescriptor* descriptor);
    ~NCSReader();
    
    // Prevent copying (thread_local connectors are tied to instance pointer)
    NCSReader(const NCSReader&) = delete;
    NCSReader& operator=(const NCSReader&) = delete;

    void read(std::vector<ReadReq> &read_reqs) override;
    void submit_req(std::vector<ReadReq> &read_reqs) override;
    void get_submitted_req() override;
};
