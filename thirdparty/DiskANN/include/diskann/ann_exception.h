// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <string>
#include <stdexcept>
#include <system_error>

#define __FUNCSIG__ __PRETTY_FUNCTION__

namespace diskann {

  class ANNException : public std::runtime_error {
   public:
    ANNException(const std::string& message, int errorCode);
    ANNException(const std::string& message, int errorCode,
                 const std::string& funcSig, const std::string& fileName,
                 unsigned int lineNum);

   private:
    int _errorCode;
  };

  class FileException : public ANNException {
   public:
    FileException(const std::string& filename, std::system_error& e,
                  const std::string& funcSig, const std::string& fileName,
                  unsigned int lineNum);
  };
}  // namespace diskann
