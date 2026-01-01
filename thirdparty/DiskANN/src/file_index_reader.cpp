#include "diskann/file_index_reader.h"
#include "diskann/defaults.h"
#include <stdexcept>
#include <cstring>


FileIndexReader::FileIndexReader(const std::string& fname, NodeSectorOffsetCallback offset_calc, 
                                 NodeOffsetCallback node_offset_calc, size_t read_len_for_node)
    : linux_aligned_file_reader(std::make_unique<LinuxAlignedFileReader>()), 
      offset_calc(std::move(offset_calc)), 
      node_offset_calc(std::move(node_offset_calc)), 
      read_len_for_node(read_len_for_node)
{
    linux_aligned_file_reader->open(fname);
    saved_ctx = IOContext(-1);
}

FileIndexReader::~FileIndexReader() {
    linux_aligned_file_reader->close();
}

std::vector<AlignedRead> FileIndexReader::convert_to_aligned_read(std::vector<ReadReq>& read_reqs) {
    std::map<size_t, std::vector<uint64_t>> sector_to_node_indices;
    for(const auto& req : read_reqs){
        size_t sector_offset = offset_calc(req.key);
        sector_to_node_indices[sector_offset].push_back(req.key);
    }

    // Clear previous buffers and allocate new ones
    sector_buffers.clear();
    sector_buffers.reserve(sector_to_node_indices.size());
    for(size_t i = 0; i < sector_to_node_indices.size(); i++){
        void* aligned_buf = std::aligned_alloc(diskann::defaults::SECTOR_LEN, read_len_for_node);
        if(aligned_buf == nullptr){
            throw std::runtime_error("Failed to allocate aligned buffer for sector read");
        }
        sector_buffers.emplace_back(aligned_buf, &std::free);
    }

    std::vector<AlignedRead> aligned_read_reqs;
    aligned_read_reqs.reserve(sector_to_node_indices.size());

    size_t buf_idx = 0;
    for(const auto& [sector_offset, node_indices] : sector_to_node_indices){
        aligned_read_reqs.push_back(
            AlignedRead{sector_offset, read_len_for_node, (void*)(sector_buffers[buf_idx].get())}
        );
        buf_idx++;
    }
    return aligned_read_reqs;
}

void FileIndexReader::copy_sector_data_to_read_reqs(std::vector<AlignedRead>& aligned_read_reqs, 
                                                     std::vector<ReadReq>& read_reqs) {
    // Rebuild sector_to_node_indices map
    std::map<size_t, std::vector<uint64_t>> sector_to_node_indices;
    for(const auto& req : read_reqs){
        size_t sector_offset = offset_calc(req.key);
        sector_to_node_indices[sector_offset].push_back(req.key);
    }

    // Create a map from node_id to ReadReq for quick lookup
    std::map<uint64_t, ReadReq*> node_to_req;
    for(auto& req : read_reqs){
        node_to_req[req.key] = &req;
    }

    // Iterate over each sector and copy node data to original requests
    size_t buf_idx = 0;
    for(const auto& [sector_offset, node_indices] : sector_to_node_indices){
        char* buf = static_cast<char*>(aligned_read_reqs[buf_idx].buf);
        size_t size = aligned_read_reqs[buf_idx].len;
        
        for(const auto& node_id : node_indices){
            char* node_data = node_offset_calc(buf, node_id);
            ReadReq* req = node_to_req[node_id];
            
            // Calculate safe copy length to avoid reading past sector buffer
            size_t offset_in_sector = node_data - buf;
            size_t max_copy_len = size - offset_in_sector;
            if (max_copy_len < req->len) {
                throw std::runtime_error("[file_index_reader] read request length exceeds sector buffer bounds");
            }            
            memcpy(req->buf, node_data, req->len);
        }
        buf_idx++;
    }
}

void FileIndexReader::read(std::vector<ReadReq>& read_reqs) {
    std::vector<AlignedRead> aligned_read_reqs = convert_to_aligned_read(read_reqs);
    auto ctx = linux_aligned_file_reader->get_ctx();
    linux_aligned_file_reader->read(aligned_read_reqs, ctx, false);
    linux_aligned_file_reader->put_ctx(ctx);
    copy_sector_data_to_read_reqs(aligned_read_reqs, read_reqs);
}

void FileIndexReader::submit_req(std::vector<ReadReq> &read_reqs) {
    // Save original read requests for copying data back later
    saved_read_reqs = read_reqs;
    saved_aligned_reads = convert_to_aligned_read(saved_read_reqs);
    saved_ctx = linux_aligned_file_reader->get_ctx();
    // Use the actual number of aligned reads submitted, not the original read_reqs count
    // (multiple read_reqs can map to the same sector)
    saved_n_ops = saved_aligned_reads.size();
    linux_aligned_file_reader->submit_req(saved_ctx, saved_aligned_reads);
}

void FileIndexReader::get_submitted_req() {
    linux_aligned_file_reader->get_submitted_req(saved_ctx, saved_n_ops);
    linux_aligned_file_reader->put_ctx(saved_ctx);
    // Copy data from aligned sector buffers back to original request buffers
    copy_sector_data_to_read_reqs(saved_aligned_reads, saved_read_reqs);
}

