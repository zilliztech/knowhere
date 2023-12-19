// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>
#include <any>

namespace diskann {

  class Parameters {
   public:
    Parameters() {
      Set<unsigned>("num_threads", 0);
    }

    template<typename ParamType>
    inline void Set(const std::string &name, const ParamType &value) {
      params[name] = static_cast<ParamType>(value);
    }

    template<typename ParamType>
    inline ParamType Get(const std::string &name) const {
      auto item = params.find(name);
      if (item == params.end()) {
        throw std::invalid_argument("Invalid parameter name.");
      } else {
        return std::any_cast<ParamType>(item->second);
      }
    }

    template<typename ParamType>
    inline ParamType Get(const std::string &name,
                         const ParamType &  default_value) {
      try {
        return Get<ParamType>(name);
      } catch (std::invalid_argument e) {
        return default_value;
      }
    }

   private:
    std::unordered_map<std::string, std::any> params;

    Parameters(const Parameters &);
    Parameters &operator=(const Parameters &);
  };
}  // namespace diskann
