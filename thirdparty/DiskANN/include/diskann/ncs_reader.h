#pragma once

#include "index_reader.h"
#include "ncs/ncs.h"
#include "ncs/InMemNcsConnector.h"
#include "ncs/InMemoryNcs.h"
#include <memory>
#include <optional>
#include <future>

/**
 * @brief Near Compute Storage (NCS) implementation of IndexReader.
 * 
 * Reads DiskANN graph nodes from NCS (Near Compute Storage) - a distributed
 * storage system that provides fast access to index data. Uses multi-get
 * operations to batch node requests efficiently.
 */
class NCSReader : public IndexReader {
private:
    std::unique_ptr<milvus::NcsConnector> ncs_connector;
    static inline thread_local std::optional<std::future<void>> async_read_future;

public:
    NCSReader(const milvus::NcsDescriptor* descriptor);
    ~NCSReader();

    void read(std::vector<ReadReq> &read_reqs) override;
    void submit_req(std::vector<ReadReq> &read_reqs) override;
    void get_submitted_req() override;
};
