// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "diskann/logger.h"
#include "diskann/memory_mapper.h"
#include <iostream>
#include <sstream>
#include <assert.h>
#include <string.h>
#include <errno.h>

using namespace diskann;

MemoryMapper::MemoryMapper(const std::string& filename)
    : MemoryMapper(filename.c_str()) {
}

MemoryMapper::MemoryMapper(const char* filename) {
  _fd = open(filename, O_RDONLY);
  assert(_fd >0);

  struct stat sb;
  int err = fstat(_fd, &sb);
  assert(err == 0);
  _fileSize = sb.st_size;
  _buf = (char*) mmap(NULL, _fileSize, PROT_READ, MAP_PRIVATE, _fd, 0);
  if(_buf == MAP_FAILED) {
	  cout << "mmap failed: " << strerror(errno) << std::endl;
	  throw errno;
  }
}
char* MemoryMapper::getBuf() {
  return _buf;
}

size_t MemoryMapper::getFileSize() {
  return _fileSize;
}

MemoryMapper::~MemoryMapper() {
  if (munmap(_buf, _fileSize) != 0){
	std::cerr << "ERROR unmapping. CHECK!" << std::endl;
  }
  close(_fd);
}
