#pragma once
#include <cstdint>
#include "simd/hook.h"
#include "diskann/utils.h"
#include "knowhere/operands.h"
namespace diskann {

  template<typename T>
  using DISTFUN = std::function<float(const T *, const T *, size_t)>;

  template<typename T>
  DISTFUN<T> get_distance_function(Metric m);

  template<typename T>
  float norm_l2sqr(const T *x, size_t size);
}  // namespace diskann
