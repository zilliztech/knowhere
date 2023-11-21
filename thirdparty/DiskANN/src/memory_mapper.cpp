// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "diskann/logger.h"
#include "diskann/memory_mapper.h"
#include <iostream>
#include <sstream>

using namespace diskann;

MemoryMapper::MemoryMapper(const std::string& filename)
    : MemoryMapper(filename.c_str()) {
}

MemoryMapper::MemoryMapper(const char* filename) {
  _fd = open(filename, O_RDONLY);
  if (_fd <= 0) {
    std::cerr << "Inner vertices file not found" << std::endl;
    return;
  }
  struct stat sb;
  if (fstat(_fd, &sb) != 0) {
    std::cerr << "Inner vertices file not dound. " << std::endl;
    return;
  }
  _fileSize = sb.st_size;
  diskann::cout << "File Size: " << _fileSize << std::endl;
  _buf = (char*) mmap(NULL, _fileSize, PROT_READ, MAP_PRIVATE, _fd, 0);
}
char* MemoryMapper::getBuf() {
  return _buf;
}

size_t MemoryMapper::getFileSize() {
  return _fileSize;
}

MemoryMapper::~MemoryMapper() {
  if (munmap(_buf, _fileSize) != 0)
    std::cerr << "ERROR unmapping. CHECK!" << std::endl;
  close(_fd);
}
