#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma once

// disable HNSW SIMD implementation, using faiss instead
#if 0
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>

#include <stdexcept>

#include "cpu_x86.h"
void
cpu_x86::cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
__int64
xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#else
#include <cpuid.h>
#include <stdint.h>
#include <x86intrin.h>
void
cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
uint64_t
xgetbv(unsigned int index) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK 0

bool
AVXCapable() {
    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

bool
AVX512Capable() {
    if (!AVXCapable())
        return false;

    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000007) {  //  AVX512 Foundation
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}
#endif

#include <string.h>

#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <vector>

#include "io/memory_io.h"
#include "neighbor.h"

#include "knowhere/bitsetview.h"
#include "knowhere/feder/HNSW.h"
#include "knowhere/object.h"

namespace hnswlib {
typedef int64_t labeltype;

template <typename T>
class pairGreater {
 public:
    bool
    operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

template <typename DistanceType>
using DISTFUNC = DistanceType (*)(const void*, const void*, const void*);

template <typename DistanceType>
class SpaceInterface {
 public:
    // virtual void search(void *);
    virtual size_t
    get_data_size() = 0;

    virtual DISTFUNC<DistanceType>
    get_dist_func() = 0;

    virtual DISTFUNC<float>
    get_dist_func_sq() {
        throw std::runtime_error("Not implemented\n");
    }

    virtual void*
    get_dist_func_param() = 0;

    virtual ~SpaceInterface() {
    }
};

struct SearchParam {
    size_t ef_;
    bool for_tuning;
};

struct IteratorWorkspace {
    IteratorWorkspace(std::unique_ptr<int8_t[]> query_data_sq, const size_t num_elements, const size_t ef,
                      const bool for_tuning, std::unique_ptr<int8_t[]> raw_query_data,
                      const knowhere::BitsetView& bitset, float accumulative_alpha)
        : query_data(query_data_sq ? (const void*)(query_data_sq.get()) : (const void*)(raw_query_data.get())),
          query_data_sq(std::move(query_data_sq)),
          visited(num_elements),
          ef(ef),
          param(std::make_unique<SearchParam>()),
          raw_query_data(std::move(raw_query_data)),
          bitset(bitset),
          accumulative_alpha(accumulative_alpha) {
        param->ef_ = 0;
        param->for_tuning = for_tuning;
    }
    const void* query_data;

    // NEVER ACCESS THIS DIRECTLY! USE query_data instead.
    std::unique_ptr<int8_t[]> query_data_sq;

    bool initial_search_done = false;
    // TODO test for memory usage of this heap and add a metric monitoring it.
    IteratorMinHeap to_visit;
    // Since iterators do not occupy a thread during the entire lifecycle of an
    // iteration request, we cannot use the visited list in the shared visited list pool,
    // thus creating a new visited list for every new iteration request.
    std::vector<bool> visited;
    std::vector<knowhere::DistId> dists;
    const size_t ef;
    std::unique_ptr<SearchParam> param;
    // though named raw_query_vector, it is normalized for cosine metric. used
    // only for refinement when quantization is enabled.
    std::unique_ptr<int8_t[]> raw_query_data;
    const knowhere::BitsetView bitset;
    float accumulative_alpha;
};

template <typename dist_t>
class AlgorithmInterface {
 public:
    virtual void
    addPoint(const void* datapoint, labeltype label) = 0;

    virtual std::vector<std::pair<dist_t, labeltype>>
    searchKnnBF(const void*, size_t, const knowhere::BitsetView) const = 0;

    virtual std::vector<std::pair<dist_t, labeltype>>
    searchKnn(const void*, size_t, const knowhere::BitsetView, const SearchParam*,
              const knowhere::feder::hnsw::FederResultUniq&) const = 0;

    virtual std::unique_ptr<IteratorWorkspace>
    getIteratorWorkspace(const void*, const size_t, const bool, const knowhere::BitsetView&) const = 0;

    virtual void
    getIteratorNextBatch(IteratorWorkspace*, const knowhere::feder::hnsw::FederResultUniq&) const = 0;

    virtual std::vector<std::pair<dist_t, labeltype>>
    searchRangeBF(const void*, float, const knowhere::BitsetView) const = 0;

    virtual std::vector<std::pair<dist_t, labeltype>>
    searchRange(const void*, float, const knowhere::BitsetView, const SearchParam*,
                const knowhere::feder::hnsw::FederResultUniq&) const = 0;

    // Return k nearest neighbor in the order of closer fist
    virtual std::vector<std::pair<dist_t, labeltype>>
    searchKnnCloserFirst(void* query_data, size_t k, const knowhere::BitsetView) const;

    virtual void
    saveIndex(knowhere::MemoryIOWriter& output) = 0;
    virtual ~AlgorithmInterface() {
    }
};

template <typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(void* query_data, size_t k, const knowhere::BitsetView bitset) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    return searchKnn(query_data, k, bitset, nullptr, nullptr);
}
}  // namespace hnswlib

#include "hnswalg.h"
#include "space_cosine.h"
#include "space_hamming.h"
#include "space_ip.h"
#include "space_jaccard.h"
#include "space_l2.h"
#pragma GCC diagnostic pop
