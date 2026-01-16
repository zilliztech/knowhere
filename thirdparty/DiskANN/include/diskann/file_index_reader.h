#pragma once

#include "index_reader.h"
#include "linux_aligned_file_reader.h"
#include "aligned_file_reader.h"
#include <map>
#include <memory>
#include <functional>

/**
 * @brief Disk-based IndexReader using sector-aligned I/O.
 * 
 * Reads DiskANN graph nodes from disk files with optimizations:
 * - Coalesces multiple node reads from the same sector
 * - Uses aligned buffers for efficient I/O (default 4096-byte sectors)
 * - Supports both sync and async read operations
 */
class FileIndexReader : public IndexReader {
private:
    using NodeSectorOffsetCallback = std::function<size_t(size_t)>;
    using NodeOffsetCallback = std::function<char*(char*, uint64_t)>;

    std::unique_ptr<LinuxAlignedFileReader> linux_aligned_file_reader;
    static inline thread_local IOContext saved_ctx;
    static inline thread_local size_t saved_n_ops;
    NodeSectorOffsetCallback offset_calc;      ///< Maps node ID to sector offset
    NodeOffsetCallback node_offset_calc;       ///< Maps node ID to offset within sector
    size_t read_len_for_node;
    
    static inline thread_local std::vector<std::unique_ptr<void, decltype(&std::free)>> sector_buffers;
    static inline thread_local std::vector<AlignedRead> saved_aligned_reads;
    static inline thread_local std::vector<ReadReq> saved_read_reqs;

    /// Convert node requests to sector-aligned reads
    std::vector<AlignedRead> convert_to_aligned_read(std::vector<ReadReq>& read_reqs);
    
    /// Copy node data from sector buffers to request buffers
    void copy_sector_data_to_read_reqs(std::vector<AlignedRead>& aligned_read_reqs, 
                                       std::vector<ReadReq>& read_reqs);

public:
    FileIndexReader(const std::string& fname, NodeSectorOffsetCallback offset_calc, 
                   NodeOffsetCallback node_offset_calc, size_t read_len_for_node);
    ~FileIndexReader();

    void read(std::vector<ReadReq>& read_reqs) override;
    void submit_req(std::vector<ReadReq> &read_reqs) override;
    void get_submitted_req() override;
};
