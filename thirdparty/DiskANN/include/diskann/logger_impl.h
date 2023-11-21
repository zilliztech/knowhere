// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <sstream>
#include <mutex>

#include "ann_exception.h"

namespace ANNIndex {
  enum LogLevel {
    LL_Debug = 0,
    LL_Info,
    LL_Status,
    LL_Warning,
    LL_Error,
    LL_Assert,
    LL_Count
  };
};

namespace diskann {
  class ANNStreamBuf : public std::basic_streambuf<char> {
   public:
    explicit ANNStreamBuf(FILE* fp);
    ~ANNStreamBuf();

    bool is_open() const {
      return true;  // because stdout and stderr are always open.
    }
    void        close();
    virtual int underflow();
    virtual int overflow(int c);
    virtual int sync();

   private:
    FILE*              _fp;
    char*              _buf;
    int                _bufIndex;
    std::mutex         _mutex;
    ANNIndex::LogLevel _logLevel;

    int  flush();
    void logImpl(char* str, int numchars);

    static const int BUFFER_SIZE = 0;

    ANNStreamBuf(const ANNStreamBuf&);
    ANNStreamBuf& operator=(const ANNStreamBuf&);
  };
}  // namespace diskann
