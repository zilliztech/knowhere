#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <faiss/impl/io.h>
#include <faiss/impl/maybe_owned_vector.h>

namespace faiss {

// holds a memory-mapped region over a file
struct MmappedFileMappingOwner : public MappingOwner {
    MmappedFileMappingOwner(const std::string& filename);
    MmappedFileMappingOwner(FILE* f);
    ~MmappedFileMappingOwner();

    void* data() const;
    size_t size() const;

    struct PImpl;
    std::unique_ptr<PImpl> p_impl;
};

// a deserializer that supports memory-mapped files
struct MappedFileIOReader : IOReader {
    std::shared_ptr<MmappedFileMappingOwner> mmap_owner;

    size_t pos = 0;

    MappedFileIOReader(const std::shared_ptr<MmappedFileMappingOwner>& owner);

    // perform a copy
    size_t operator()(void* ptr, size_t size, size_t nitems) override;
    // perform a quasi-read that returns a mmapped address, owned by mmap_owner,
    //   and updates the position
    size_t mmap(void** ptr, size_t size, size_t nitems);

    int filedescriptor() override;
};

}