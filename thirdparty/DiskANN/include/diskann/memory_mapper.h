// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>

namespace diskann {
  class MemoryMapper {
   private:
    int _fd;
    char*       _buf;
    size_t      _fileSize;
    const char* _fileName;

   public:
    MemoryMapper(const char* filename);
    MemoryMapper(const std::string& filename);
    char*  getBuf();
    size_t getFileSize();

    ~MemoryMapper();
  };
}  // namespace diskann
