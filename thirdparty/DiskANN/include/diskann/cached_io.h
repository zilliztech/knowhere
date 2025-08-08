// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "ann_exception.h"
#include "knowhere/log.h"
#include "logger.h"

// sequential cached reads

class cached_ifstream {
public:

    cached_ifstream() {
    }

    cached_ifstream(const std::string& filename, uint64_t cacheSize)
    : cache_size(cacheSize), cur_off(0) {
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        this->open(filename, cache_size);
    }

    ~cached_ifstream() {
        reader.close();
    }

    void open(const std::string& filename, uint64_t cacheSize) {
        this->cur_off = 0;

        try {
            reader.open(filename, std::ios::binary | std::ios::ate);
            fsize = reader.tellg();
            reader.seekg(0, std::ios::beg);
            assert(reader.is_open());
            assert(cacheSize > 0);
            cacheSize = (std::min)(cacheSize, fsize);
            this->cache_size = cacheSize;
            cache_buf = std::make_unique<char[]>(cacheSize);
            reader.read(cache_buf.get(), cacheSize);
            LOG_KNOWHERE_DEBUG_ << "Opened: " << filename.c_str()
                                << ", size: " << fsize
                                << ", cache_size: " << cacheSize;
        } catch (std::system_error& e) {
            throw diskann::FileException(filename, e, __FUNCSIG__, __FILE__,
                    __LINE__);
        }
    }

    size_t get_file_size() {
        return fsize;
    }

    void read(char* read_buf, uint64_t n_bytes) {
        assert(cache_buf != nullptr);
        assert(read_buf != nullptr);

        if (n_bytes <= (cache_size - cur_off)) {
            // case 1: cache contains all data
            memcpy(read_buf, cache_buf.get() + cur_off, n_bytes);
            cur_off += n_bytes;
        } else {
            // case 2: cache contains some data
            uint64_t cached_bytes = cache_size - cur_off;
            if (n_bytes - cached_bytes > fsize - reader.tellg()) {
                std::stringstream stream;
                stream << "Reading beyond end of file" << std::endl;
                stream << "n_bytes: " << n_bytes << " cached_bytes: " << cached_bytes
                       << " fsize: " << fsize << " current pos:" << reader.tellg()
                       << std::endl;
                diskann::cout << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            memcpy(read_buf, cache_buf.get() + cur_off, cached_bytes);

            // go to disk and fetch more data
            reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
            // reset cur off
            cur_off = cache_size;

            uint64_t size_left = fsize - reader.tellg();

            if (size_left >= cache_size) {
                reader.read(cache_buf.get(), cache_size);
                cur_off = 0;
            }
            // note that if size_left < cache_size, then cur_off = cache_size, so
            // subsequent reads will all be directly from file
        }
    }

    void seek(uint64_t pos) {
        reader.seekg(pos, reader.beg);
        /* invalidate */
        cur_off = cache_size;
    }

    bool set_cache_size(uint64_t cacheSize) {
        uint64_t new_cache_size = (std::min)(cacheSize, fsize);
        char *new_cache_buf = new char[new_cache_size];
        if (new_cache_buf != nullptr) {
            if (cache_buf != nullptr) {
                auto ptr = cache_buf.release();
                delete[] ptr;
            }
            cache_size = new_cache_size;
            cache_buf = std::unique_ptr<char[] >(new_cache_buf);
            /* invalidate */
            cur_off = cache_size;
            return true;
        }
        return false;
    }
private:
    // underlying ifstream
    std::ifstream reader;
    // # bytes to cache in one shot read
    uint64_t cache_size = 0;
    // underlying buf for cache
    std::unique_ptr<char[] > cache_buf = nullptr;
    // offset into cache_buf for cur_pos
    uint64_t cur_off = 0;
    // file size
    uint64_t fsize = 0;
};

// sequential cached writes

class cached_ofstream {
public:

    cached_ofstream(const std::string& filename, uint64_t cache_size)
    : cache_size(cache_size), cur_off(0) {
        writer.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try {
            writer.open(filename, std::ios::binary);
            assert(writer.is_open());
            assert(cache_size > 0);
            cache_buf = std::make_unique<char[]>(cache_size);
            LOG_KNOWHERE_DEBUG_ << "Opened: " << filename.c_str()
                    << ", cache_size: " << cache_size;
        } catch (std::system_error& e) {
            throw diskann::FileException(filename, e, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    ~cached_ofstream() {
        // dump any remaining data in memory
        if (cur_off > 0) {
            this->flush_cache();
        }
        if (writer.is_open())
            writer.close();
        LOG_KNOWHERE_DEBUG_ << "Finished writing " << fsize << "B";
    }

    size_t get_file_size() {
        return fsize;
    }
    // writes n_bytes from write_buf to the underlying ofstream/cache

    void write(char* write_buf, uint64_t n_bytes) {
        assert(cache_buf != nullptr);
        if (n_bytes <= (cache_size - cur_off)) {
            // case 1: cache can take all data
            memcpy(cache_buf.get() + cur_off, write_buf, n_bytes);
            cur_off += n_bytes;
        } else {
            // case 2: cache cant take all data
            // go to disk and write existing cache data
            writer.write(cache_buf.get(), cur_off);
            fsize += cur_off;
            // write the new data to disk
            writer.write(write_buf, n_bytes);
            fsize += n_bytes;
            // memset all cache data and reset cur_off
            memset(cache_buf.get(), 0, cache_size);
            cur_off = 0;
        }
    }

    void flush_cache() {
        assert(cache_buf != nullptr);
        writer.write(cache_buf.get(), cur_off);
        fsize += cur_off;
        memset(cache_buf.get(), 0, cache_size);
        cur_off = 0;
    }

    void reset() {
        flush_cache();
        writer.seekp(0);
    }

    void close(){
          // dump any remaining data in memory
          if (cur_off > 0)
          {
             this->flush_cache();
          }

          cache_buf = nullptr;
          if (writer.is_open())
             writer.close();
          diskann::cout << "Finished writing " << fsize << "B" << std::endl;
    }
private:
    // underlying ofstream
    std::ofstream writer;
    // # bytes to cache for one shot write
    uint64_t cache_size = 0;
    // underlying buf for cache
    std::unique_ptr<char[] > cache_buf = nullptr;
    // offset into cache_buf for cur_pos
    uint64_t cur_off = 0;

    // file size
    uint64_t fsize = 0;
};
