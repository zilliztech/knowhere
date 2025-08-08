// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <mutex>
#include <memory>
#include <cstring>
#include <cstdio>
#include "diskann/logger_impl.h"

namespace diskann {

  // Global logger objects (unchanged)
  ANNStreamBuf coutBuff(stdout);
  ANNStreamBuf cerrBuff(stderr);
  std::basic_ostream<char> cout(&coutBuff);
  std::basic_ostream<char> cerr(&cerrBuff);

  ANNStreamBuf::ANNStreamBuf(FILE* fp) {
    if (fp == nullptr) {
      throw diskann::ANNException(
          "File pointer passed to ANNStreamBuf() cannot be null", -1);
    }
    if (fp != stdout && fp != stderr) {
      throw diskann::ANNException(
          "The custom logger only supports stdout and stderr.", -1);
    }

    _fp = fp;
    _logLevel = (_fp == stdout) ? ANNIndex::LogLevel::LL_Info
                                : ANNIndex::LogLevel::LL_Error;

    _buf = std::make_unique<char[]>(BUFFER_SIZE);
    std::memset(_buf.get(), 0, BUFFER_SIZE);
    setp(_buf.get(), _buf.get() + BUFFER_SIZE);
  }

  ANNStreamBuf::~ANNStreamBuf() {
    sync();  // flush remaining data
    _fp = nullptr;  // don't close stdout/stderr
  }

  int ANNStreamBuf::overflow(int c) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (c != EOF) {
      *pptr() = static_cast<char>(c);
      pbump(1);
    }

    flush();  // flush the buffer
    return c;
  }

  int ANNStreamBuf::sync() {
    std::lock_guard<std::mutex> lock(_mutex);
    flush();
    return 0;
  }

  int ANNStreamBuf::underflow() {
    throw diskann::ANNException("Attempt to read from write-only streambuf", -1);
  }

  int ANNStreamBuf::flush() {
    int num = static_cast<int>(pptr() - pbase());
    if (num > 0) {
      logImpl(pbase(), num);
      pbump(-num);  // reset buffer pointer
    }
    return num;
  }

  void ANNStreamBuf::logImpl(char* str, int num) {
    fwrite(str, sizeof(char), num, _fp);
    fflush(_fp);
  }

}  // namespace diskann
