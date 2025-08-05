// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <stdint.h>

/* enable all-in-storage features */

namespace diskann
{
namespace defaults
{

const float ALPHA = 1.2f;
const uint32_t NUM_THREADS = 0;
const uint32_t MAX_OCCLUSION_SIZE = 750;
const uint32_t FILTER_LIST_SIZE = 0;
const uint32_t NUM_FROZEN_POINTS_STATIC = 0;
const uint32_t NUM_FROZEN_POINTS_DYNAMIC = 1;
const uint32_t NUM_PQ_CENTROIDS = 256;
const uint32_t MAX_PQ_CHUNKS = 512;
const uint32_t DEFAULT_INLINE_PQ = -1;
const bool     DEFAULT_REARRANGE = true;
const uint32_t MIN_NUM_ENTRY_POINTS = 0;
const uint32_t MAX_NUM_ENTRY_POINTS = 1000;
const uint32_t DEFAULT_NUM_ENTRY_POINTS = 100;
const uint32_t DEFAULT_PQ_CACHE_SIZE = 0;
const uint32_t MIN_PQ_CACHE_SIZE = 0;
const uint32_t MAX_PQ_CACHE_SIZE = 1 << 30 /* 1g */;
const uint32_t DEFAULT_PQ_READ_PAGE_CACHE_SIZE = 5242880 /* 5m */;
const uint32_t MIN_PQ_READ_PAGE_CACHE_SIZE = 0;
const uint32_t MAX_PQ_READ_PAGE_CACHE_SIZE = 33554432 /* 32m */;
const uint32_t DEFAULT_AISAQ_VECTORS_BEAMWIDTH = 1;
const uint32_t DEFAULT_AISAQ_BEAMWIDTH = 1;
const uint32_t MAX_AISAQ_VECTORS_BEAMWIDTH = 4;
const uint32_t MAX_AISAQ_BEAMWIDTH = 16;
const uint32_t MAX_AISAQ_MAX_DEGREE = 512;
const uint32_t MAX_AISAQ_SEARCH_LIST_SIZE = 512;

// In-mem index related limits
const float GRAPH_SLACK_FACTOR = 1.3;

// SSD Index related limits
const uint64_t MAX_GRAPH_DEGREE = 512;
const uint64_t SECTOR_LEN = 4096;
const uint64_t MAX_N_SECTOR_READS = 1024;


// following constants should always be specified, but are useful as a
// sensible default at cli / python boundaries
const uint32_t MAX_DEGREE = 64;
const uint32_t BUILD_LIST_SIZE = 100;
const uint32_t SATURATE_GRAPH = false;
const uint32_t SEARCH_LIST_SIZE = 100;
} // namespace defaults
} // namespace diskann
