// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef SPARSE_DSP_INDEX_H
#define SPARSE_DSP_INDEX_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <queue>
#include <vector>

#include "index/sparse/sparse_inverted_index.h"

namespace knowhere::sparse {

// DSP (Dynamic Superblock Pruning) index for fast sparse vector search.
//
// Following the SIGIR'25 DSP paper and reference implementation:
// - u8 quantized block max scores with dual dense/sparse format per dimension
// - u16 upper bound accumulators with AVX-512 SIMD for dense dimensions
// - Counting sort (bucket sort) for block ordering by upper bound
// - Forward index with two-pointer merge scoring
// - Two-level hierarchy: superblocks for coarse pruning, subblocks for scoring
//
// Inherits storage, serialization, and utility code from InvertedIndex<DAAT_MAXSCORE>.
template <typename DType, typename QType, bool mmapped = false>
class DspIndex : public InvertedIndex<DType, QType, InvertedIndexAlgo::DAAT_MAXSCORE, mmapped> {
    using Base = InvertedIndex<DType, QType, InvertedIndexAlgo::DAAT_MAXSCORE, mmapped>;

 public:
    static constexpr uint32_t kSubblockSize = 8;
    static constexpr uint32_t kSuperblockSize = 512;
    static constexpr uint32_t kStride = kSuperblockSize / kSubblockSize;  // 64
    static constexpr uint32_t kSimdWidth = 32;                            // AVX-512 processes 32 u16 values

    explicit DspIndex(SparseMetricType metric_type) : Base(metric_type) {
    }

    Status
    Add(const SparseRow<DType>* data, size_t rows, int64_t dim) override {
        RETURN_IF_ERROR(Base::Add(data, rows, dim));
        build_dsp_metadata();
        return Status::success;
    }

    Status
    Deserialize(MemoryIOReader& reader) override {
        RETURN_IF_ERROR(Base::Deserialize(reader));
        build_dsp_metadata();
        return Status::success;
    }

    Status
    DeserializeV0(MemoryIOReader& reader, int map_flags, const std::string& fn) override {
        RETURN_IF_ERROR(Base::DeserializeV0(reader, map_flags, fn));
        build_dsp_metadata();
        return Status::success;
    }

    void
    Search(const SparseRow<DType>& query, size_t k, float* distances, label_t* labels, const BitsetView& bitset,
           const DocValueComputer<float>& computer, InvertedIndexApproxSearchParams& approx_params) const override {
        std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
        std::fill(labels, labels + k, -1);
        if (query.size() == 0) {
            return;
        }

        auto q_vec = this->parse_query(query, approx_params.drop_ratio_search);
        if (q_vec.empty()) {
            return;
        }

        const size_t heap_capacity = k * approx_params.refine_factor;
        MaxMinHeap<float> heap(heap_capacity);
        search_dsp(q_vec, heap, heap_capacity, bitset, computer, approx_params.dim_max_score_ratio,
                   approx_params.dsp_mu, approx_params.dsp_eta);

        if (approx_params.refine_factor == 1) {
            this->collect_result(heap, distances, labels);
        } else {
            this->refine_and_collect(query, heap, k, distances, labels, computer, approx_params);
        }
    }

 private:
    // ========================================================================
    // Per-dimension block max scores (u8 quantized)
    // ========================================================================
    struct DimBlockMax {
        std::vector<uint32_t> block_ids;  // empty if dense format
        std::vector<uint8_t> max_scores;  // dense: n_sb_padded_ entries; sparse: parallel to block_ids
        uint8_t kth[4] = {0, 0, 0, 0};    // kth-largest u8 score at k=10,100,1000,10000
        bool
        is_dense() const {
            return block_ids.empty() && !max_scores.empty();
        }
    };
    std::vector<DimBlockMax> dim_block_max_;

    // ========================================================================
    // Superblock max + ASC (sparse CSR format, float — used for coarse pruning)
    // ========================================================================
    std::vector<uint32_t> spb_dim_offsets_;
    std::vector<uint32_t> spb_block_ids_;
    std::vector<float> spb_max_vals_;
    std::vector<float> spb_asc_vals_;  // Average Segment Contribution per (dim, superblock)

    // ========================================================================
    // Forward index (flat layout for cache-friendly scoring)
    // ========================================================================
    std::vector<uint32_t> fwd_block_term_offsets_;  // [n_subblocks_ + 1]
    std::vector<uint32_t> fwd_term_ids_;            // inner dim IDs, sorted per block
    std::vector<uint32_t> fwd_term_entry_offsets_;  // [total_terms + 1]
    std::vector<uint8_t> fwd_doc_offsets_;          // doc position within subblock (0..7)
    std::vector<float> fwd_scores_;                 // pre-computed contribution scores

    uint32_t n_subblocks_ = 0;
    uint32_t n_superblocks_ = 0;
    uint32_t n_sb_padded_ = 0;  // n_subblocks_ rounded up to kSimdWidth

    // Dense threshold: if a posting list has non-zero blocks in more than
    // this fraction of total subblocks, store as dense u8 array.
    static constexpr float kDenseThreshold = 0.125f;  // 12.5%

    // ASC (Average Segment Contribution): divide each superblock into segments.
    // Track max score per segment, store mean of non-zero segment maxima.
    static constexpr uint32_t kNumSegments = 8;
    static constexpr uint32_t kSegmentSize = kSuperblockSize / kNumSegments;  // 64 docs

    // ========================================================================
    // Build DSP metadata from inverted index
    // ========================================================================
    void
    build_dsp_metadata() {
        if (this->n_rows_internal_ == 0 || this->nr_inner_dims_ == 0) {
            return;
        }

        n_subblocks_ = (this->n_rows_internal_ + kSubblockSize - 1) / kSubblockSize;
        n_superblocks_ = (this->n_rows_internal_ + kSuperblockSize - 1) / kSuperblockSize;
        // Pad to multiple of kStride (64) so SIMD loops on the last superblock don't overflow
        n_sb_padded_ = (n_subblocks_ + kStride - 1) / kStride * kStride;

        const uint32_t nr_dims = this->nr_inner_dims_;
        const bool is_bm25 = this->metric_type_ == SparseMetricType::METRIC_BM25;

        // ---- Phase 1: Scan posting lists, compute block max + per-doc forward index ----
        // Per-doc forward index: (inner_dim, score) pairs appended per doc.
        // This avoids a giant "collectors" array — follows reference implementation pattern.
        struct DocFwdEntry {
            uint32_t inner_dim;
            float score;
        };
        std::vector<std::vector<DocFwdEntry>> per_doc_fwd(this->n_rows_internal_);

        // Temporary dense float block max per dim (reused)
        std::vector<float> tmp_sb_max(n_subblocks_, 0.0f);
        std::vector<uint8_t> sb_touched(n_subblocks_, 0);
        std::vector<uint32_t> touched_list;
        touched_list.reserve(n_subblocks_);

        // Superblock max: collect per dim
        std::vector<float> tmp_spb_max(n_superblocks_, 0.0f);
        std::vector<uint8_t> spb_touched(n_superblocks_, 0);
        std::vector<uint32_t> spb_touched_list;
        spb_touched_list.reserve(n_superblocks_);

        // Segment max: for ASC computation (8 segments per superblock)
        std::vector<float> tmp_seg_max(n_superblocks_ * kNumSegments, 0.0f);

        // Superblock CSR builders
        struct SpbEntry {
            uint32_t block_id;
            float max_score;
            float asc;  // Average Segment Contribution
        };
        std::vector<std::vector<SpbEntry>> per_dim_spb(nr_dims);

        // Per-dim block max result
        dim_block_max_.resize(nr_dims);

        // kth score tracking: 4 min-heaps per dim for k=10,100,1000,10000
        static constexpr uint32_t kKthSizes[4] = {10, 100, 1000, 10000};
        using KthHeap = std::priority_queue<float, std::vector<float>, std::greater<float>>;

        for (uint32_t d = 0; d < nr_dims; ++d) {
            const auto& plist_ids = this->inverted_index_ids_spans_[d];
            const auto& plist_vals = this->inverted_index_vals_spans_[d];
            const float max_score_d = this->max_score_in_dim_spans_[d];

            // Skip empty or zero-max-score dims
            if (plist_ids.size() == 0 || max_score_d <= 0.0f) {
                continue;
            }

            const float inv_max_score = 255.0f / max_score_d;

            KthHeap kth_heaps[4];

            for (size_t i = 0; i < plist_ids.size(); ++i) {
                const uint32_t doc_id = plist_ids[i];
                const QType val = plist_vals[i];

                float score;
                if (is_bm25) {
                    score = this->bm25_params_->max_score_computer(val, this->bm25_params_->row_sums_spans_[doc_id]);
                } else {
                    score = static_cast<float>(val);
                }

                // Update kth heaps
                for (int h = 0; h < 4; ++h) {
                    if (kth_heaps[h].size() < kKthSizes[h]) {
                        kth_heaps[h].push(score);
                    } else if (score > kth_heaps[h].top()) {
                        kth_heaps[h].pop();
                        kth_heaps[h].push(score);
                    }
                }

                const uint32_t sb = doc_id / kSubblockSize;
                const uint32_t spb = doc_id / kSuperblockSize;

                // Track subblock max
                if (!sb_touched[sb]) {
                    touched_list.push_back(sb);
                    sb_touched[sb] = 1;
                }
                tmp_sb_max[sb] = std::max(tmp_sb_max[sb], score);

                // Track superblock max
                if (!spb_touched[spb]) {
                    spb_touched_list.push_back(spb);
                    spb_touched[spb] = 1;
                }
                tmp_spb_max[spb] = std::max(tmp_spb_max[spb], score);

                // Track segment max (for ASC)
                const uint32_t seg = doc_id / kSegmentSize;
                tmp_seg_max[seg] = std::max(tmp_seg_max[seg], score);

                // Append to per-doc forward index
                per_doc_fwd[doc_id].push_back({d, score});
            }

            // ---- Store kth scores as u8 ----
            auto& bm = dim_block_max_[d];
            for (int h = 0; h < 4; ++h) {
                if (!kth_heaps[h].empty()) {
                    float kth_f = kth_heaps[h].top();
                    bm.kth[h] = static_cast<uint8_t>(std::min(255.0f, std::floor(kth_f * inv_max_score)));
                }
            }

            // ---- Decide dense vs sparse, quantize block max to u8 ----
            const uint32_t nnz_blocks = touched_list.size();
            if (nnz_blocks > static_cast<uint32_t>(n_subblocks_ * kDenseThreshold)) {
                // Dense: allocate padded array, zero-init (padding stays zero)
                bm.max_scores.resize(n_sb_padded_, 0);
                for (uint32_t sb : touched_list) {
                    bm.max_scores[sb] =
                        static_cast<uint8_t>(std::min(255.0f, std::ceil(tmp_sb_max[sb] * inv_max_score)));
                }
                // block_ids left empty → is_dense() returns true
            } else {
                // Sparse: sorted (block_id, u8) pairs
                std::sort(touched_list.begin(), touched_list.end());
                bm.block_ids.resize(nnz_blocks);
                bm.max_scores.resize(nnz_blocks);
                for (uint32_t i = 0; i < nnz_blocks; ++i) {
                    uint32_t sb = touched_list[i];
                    bm.block_ids[i] = sb;
                    bm.max_scores[i] =
                        static_cast<uint8_t>(std::min(255.0f, std::ceil(tmp_sb_max[sb] * inv_max_score)));
                }
            }

            // ---- Collect superblock max + ASC into CSR ----
            std::sort(spb_touched_list.begin(), spb_touched_list.end());
            per_dim_spb[d].reserve(spb_touched_list.size());
            for (uint32_t spb : spb_touched_list) {
                // Compute ASC: mean of non-zero segment maxima within this superblock
                float seg_sum = 0.0f;
                uint32_t seg_count = 0;
                for (uint32_t s = 0; s < kNumSegments; ++s) {
                    float seg_max = tmp_seg_max[spb * kNumSegments + s];
                    if (seg_max > 0.0f) {
                        seg_sum += seg_max;
                        seg_count++;
                    }
                }
                float asc = (seg_count > 0) ? (seg_sum / seg_count) : 0.0f;
                per_dim_spb[d].push_back({spb, tmp_spb_max[spb], asc});
            }

            // ---- Reset temp arrays ----
            for (uint32_t sb : touched_list) {
                tmp_sb_max[sb] = 0.0f;
                sb_touched[sb] = 0;
            }
            touched_list.clear();
            for (uint32_t spb : spb_touched_list) {
                tmp_spb_max[spb] = 0.0f;
                spb_touched[spb] = 0;
                // Reset segment tracking for this superblock
                for (uint32_t s = 0; s < kNumSegments; ++s) {
                    tmp_seg_max[spb * kNumSegments + s] = 0.0f;
                }
            }
            spb_touched_list.clear();
        }

        // ---- Phase 2: Build superblock CSR ----
        {
            uint32_t total_spb = 0;
            spb_dim_offsets_.resize(nr_dims + 1);
            for (uint32_t d = 0; d < nr_dims; ++d) {
                spb_dim_offsets_[d] = total_spb;
                total_spb += per_dim_spb[d].size();
            }
            spb_dim_offsets_[nr_dims] = total_spb;

            spb_block_ids_.resize(total_spb);
            spb_max_vals_.resize(total_spb);
            spb_asc_vals_.resize(total_spb);
            for (uint32_t d = 0; d < nr_dims; ++d) {
                uint32_t off = spb_dim_offsets_[d];
                for (const auto& e : per_dim_spb[d]) {
                    spb_block_ids_[off] = e.block_id;
                    spb_max_vals_[off] = e.max_score;
                    spb_asc_vals_[off] = e.asc;
                    ++off;
                }
            }
        }

        // ---- Phase 3: Build flat forward index from per-doc data ----
        // Process one subblock at a time: collect entries from its docs, sort by dim,
        // then emit flat forward index arrays. Frees per-doc data as we go.
        {
            // Sort each doc's entries by dim (needed for two-pointer merge during search)
            for (uint32_t doc = 0; doc < this->n_rows_internal_; ++doc) {
                auto& entries = per_doc_fwd[doc];
                if (entries.size() > 1) {
                    std::sort(entries.begin(), entries.end(),
                              [](const DocFwdEntry& a, const DocFwdEntry& b) { return a.inner_dim < b.inner_dim; });
                }
            }

            // Two-pass: first count, then fill
            uint32_t total_terms = 0;
            uint32_t total_entries = 0;

            // Temporary buffer for collecting block entries
            struct BlockEntry {
                uint32_t inner_dim;
                uint8_t doc_offset;
                float score;
            };
            std::vector<BlockEntry> block_buf;
            block_buf.reserve(1024);

            // Pass 1: count terms and entries per subblock
            for (uint32_t sb = 0; sb < n_subblocks_; ++sb) {
                block_buf.clear();
                const uint32_t doc_start = sb * kSubblockSize;
                const uint32_t doc_end =
                    std::min(doc_start + kSubblockSize, static_cast<uint32_t>(this->n_rows_internal_));
                for (uint32_t doc = doc_start; doc < doc_end; ++doc) {
                    const uint8_t doc_off = static_cast<uint8_t>(doc - doc_start);
                    for (const auto& e : per_doc_fwd[doc]) {
                        block_buf.push_back({e.inner_dim, doc_off, e.score});
                    }
                }
                if (block_buf.empty())
                    continue;
                std::sort(block_buf.begin(), block_buf.end(), [](const BlockEntry& a, const BlockEntry& b) {
                    return a.inner_dim < b.inner_dim || (a.inner_dim == b.inner_dim && a.doc_offset < b.doc_offset);
                });
                total_entries += block_buf.size();
                total_terms++;
                for (size_t i = 1; i < block_buf.size(); ++i) {
                    if (block_buf[i].inner_dim != block_buf[i - 1].inner_dim) {
                        total_terms++;
                    }
                }
            }

            fwd_block_term_offsets_.resize(n_subblocks_ + 1);
            fwd_term_ids_.resize(total_terms);
            fwd_term_entry_offsets_.resize(total_terms + 1);
            fwd_doc_offsets_.resize(total_entries);
            fwd_scores_.resize(total_entries);

            uint32_t term_pos = 0;
            uint32_t entry_pos = 0;

            // Pass 2: fill flat arrays and free per-doc data
            for (uint32_t sb = 0; sb < n_subblocks_; ++sb) {
                fwd_block_term_offsets_[sb] = term_pos;
                block_buf.clear();
                const uint32_t doc_start = sb * kSubblockSize;
                const uint32_t doc_end =
                    std::min(doc_start + kSubblockSize, static_cast<uint32_t>(this->n_rows_internal_));
                for (uint32_t doc = doc_start; doc < doc_end; ++doc) {
                    const uint8_t doc_off = static_cast<uint8_t>(doc - doc_start);
                    for (const auto& e : per_doc_fwd[doc]) {
                        block_buf.push_back({e.inner_dim, doc_off, e.score});
                    }
                    // Free this doc's per-doc data
                    per_doc_fwd[doc].clear();
                    per_doc_fwd[doc].shrink_to_fit();
                }
                if (block_buf.empty())
                    continue;
                std::sort(block_buf.begin(), block_buf.end(), [](const BlockEntry& a, const BlockEntry& b) {
                    return a.inner_dim < b.inner_dim || (a.inner_dim == b.inner_dim && a.doc_offset < b.doc_offset);
                });

                fwd_term_ids_[term_pos] = block_buf[0].inner_dim;
                fwd_term_entry_offsets_[term_pos] = entry_pos;

                for (size_t i = 0; i < block_buf.size(); ++i) {
                    if (i > 0 && block_buf[i].inner_dim != block_buf[i - 1].inner_dim) {
                        term_pos++;
                        fwd_term_ids_[term_pos] = block_buf[i].inner_dim;
                        fwd_term_entry_offsets_[term_pos] = entry_pos;
                    }
                    fwd_doc_offsets_[entry_pos] = block_buf[i].doc_offset;
                    fwd_scores_[entry_pos] = block_buf[i].score;
                    entry_pos++;
                }
                term_pos++;
            }
            fwd_block_term_offsets_[n_subblocks_] = term_pos;
            fwd_term_entry_offsets_[total_terms] = entry_pos;
        }

        // Free inverted index posting lists — DSP search uses only the forward index
        // and max_score_in_dim. Follows reference implementation which discards posting
        // lists after building block max + forward index metadata.
        // Only possible for non-mmapped mode (mmapped data is externally owned).
        if constexpr (!mmapped) {
            for (size_t i = 0; i < this->inverted_index_ids_.size(); ++i) {
                this->inverted_index_ids_[i].clear();
                this->inverted_index_ids_[i].shrink_to_fit();
                this->inverted_index_vals_[i].clear();
                this->inverted_index_vals_[i].shrink_to_fit();
            }
        }
        this->inverted_index_ids_spans_.clear();
        this->inverted_index_vals_spans_.clear();
    }

    // ========================================================================
    // DSP Search
    // ========================================================================
    template <typename DocIdFilter>
    void
    search_dsp(const std::vector<std::pair<size_t, DType>>& q_vec, MaxMinHeap<float>& heap, size_t heap_capacity,
               DocIdFilter& filter, const DocValueComputer<float>& computer, float dim_max_score_ratio, float mu,
               float eta) const {
#ifdef SEEK_INSTRUMENTATION
        auto t0 = std::chrono::high_resolution_clock::now();
#define DSP_TIMER_MARK(var) auto var = std::chrono::high_resolution_clock::now()
#define DSP_TIMER_US(a, b) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count()
#else
#define DSP_TIMER_MARK(var) (void)0
#define DSP_TIMER_US(a, b) 0
#endif
        // ---- Step 0: Prepare sorted query ----
        struct QueryTerm {
            uint32_t inner_dim;
            float weight;
            uint8_t u8_weight;
        };
        std::vector<QueryTerm> query(q_vec.size());
        for (size_t i = 0; i < q_vec.size(); ++i) {
            query[i].inner_dim = static_cast<uint32_t>(q_vec[i].first);
            query[i].weight = static_cast<float>(q_vec[i].second);
        }
        // Sort by inner_dim for two-pointer merge
        std::sort(query.begin(), query.end(), [](const auto& a, const auto& b) { return a.inner_dim < b.inner_dim; });
        const size_t n_query_terms = query.size();

        // ---- Step 1: Compute u8 query weights and scale factor ----
        // Note: dim_max_score_ratio is NOT included in S. It cancels in u8 weights
        // (ratio appears in both numerator and S denominator). Excluding it from S
        // gives a larger score_scale, which means a tighter u16 threshold for pruning.
        float S = 0.0f;
        for (const auto& qt : query) {
            S += qt.weight * this->max_score_in_dim_spans_[qt.inner_dim];
        }
        if (S <= 0.0f)
            return;

        const float inv_S = 255.0f / S;
        for (auto& qt : query) {
            float w = qt.weight * this->max_score_in_dim_spans_[qt.inner_dim] * inv_S;
            // Use ceil to ensure u16 UB >= true float UB * scale (no false negatives)
            uint8_t u8w = static_cast<uint8_t>(std::min(255.0f, std::max(1.0f, std::ceil(w))));
            qt.u8_weight = u8w;
        }
        const float score_scale = 65025.0f / S;

        // ---- Step 2: Initialize threshold from kth scores ----
        float float_threshold = 0.0f;
#ifndef DSP_DISABLE_KTH_INIT
        {
            // Select kth bucket based on k
            int kth_bucket = (heap_capacity > 10) + (heap_capacity > 100) + (heap_capacity > 1000);
            for (const auto& qt : query) {
                const auto& bm = dim_block_max_[qt.inner_dim];
                uint8_t kth_u8 = bm.kth[kth_bucket];
                if (kth_u8 == 0)
                    continue;
                float kth_float = kth_u8 / 255.0f * this->max_score_in_dim_spans_[qt.inner_dim];
                float term_thresh = qt.weight * kth_float;
                float_threshold = std::max(float_threshold, term_thresh);
            }
        }
#endif
        uint16_t u16_threshold = static_cast<uint16_t>(std::min(65535.0f, float_threshold * score_scale));

        DSP_TIMER_MARK(t1);
        // ---- Step 3: Superblock pruning with ASC (sparse float) ----
        // Compute both max UBs and ASC UBs per superblock
        std::vector<float> superblock_ub(n_superblocks_, 0.0f);
        std::vector<float> superblock_asc(n_superblocks_, 0.0f);
        for (const auto& qt : query) {
            const float qw = qt.weight;
            const uint32_t start = spb_dim_offsets_[qt.inner_dim];
            const uint32_t end = spb_dim_offsets_[qt.inner_dim + 1];
            for (uint32_t i = start; i < end; ++i) {
                superblock_ub[spb_block_ids_[i]] += qw * spb_max_vals_[i];
                superblock_asc[spb_block_ids_[i]] += qw * spb_asc_vals_[i];
            }
        }

        // Dual threshold check: keep if max exceeds mu-threshold OR asc exceeds eta-threshold
        const float mu_threshold = (mu > 0.0f) ? float_threshold / mu : float_threshold;
        const float eta_threshold = (eta > 0.0f) ? float_threshold / eta : float_threshold;

        // Collect surviving superblocks + build bitset
        std::vector<uint32_t> surviving_spb;
        surviving_spb.reserve(n_superblocks_);
        std::vector<uint8_t> spb_alive(n_superblocks_, 0);
        for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
            if (superblock_ub[spb] > mu_threshold || superblock_asc[spb] > eta_threshold) {
                surviving_spb.push_back(spb);
                spb_alive[spb] = 1;
            }
        }
        if (surviving_spb.empty())
            return;

        DSP_TIMER_MARK(t2);
        // ---- Step 4: Subblock UB computation (u8×u8→u16 with SIMD for dense) ----
        std::vector<uint16_t> block_ub(n_sb_padded_, 0);

        for (const auto& qt : query) {
            const auto& bm = dim_block_max_[qt.inner_dim];
            if (bm.max_scores.empty())
                continue;

            if (bm.is_dense()) {
                // Dense: process only surviving superblocks' subblock ranges
                for (uint32_t spb : surviving_spb) {
                    const uint32_t sb_start = spb * kStride;
                    // Each superblock = kStride subblocks = 64, process with SIMD (2 iterations of 32)
                    accumulate_block_ub_dispatch(block_ub.data() + sb_start, bm.max_scores.data() + sb_start,
                                                 static_cast<uint16_t>(qt.u8_weight), kStride);
                }
            } else {
                // Sparse: scalar scatter with superblock filtering
                const uint16_t u16w = static_cast<uint16_t>(qt.u8_weight);
                for (size_t i = 0; i < bm.block_ids.size(); ++i) {
                    const uint32_t sb = bm.block_ids[i];
                    if (!spb_alive[sb / kStride])
                        continue;
                    uint32_t prod = u16w * bm.max_scores[i];
                    uint32_t sum = static_cast<uint32_t>(block_ub[sb]) + prod;
                    block_ub[sb] = static_cast<uint16_t>(sum < 65535u ? sum : 65535u);
                }
            }
        }

        DSP_TIMER_MARK(t3);
        // ---- Step 5: Collect candidates and counting sort by u16 UB ----
        // SIMD fast-path: scan 64 u16 block UBs per superblock in 2 AVX-512 loads.
        // If no block in the superblock exceeds threshold, skip entirely.
        std::vector<uint32_t> candidates;
        candidates.reserve(n_subblocks_ / 4);
        uint16_t max_ub = 0;

        for (uint32_t spb : surviving_spb) {
            const uint32_t sb_start = spb * kStride;
            // SIMD scan: check if ANY of the 64 subblocks exceeds threshold
            // block_ub is padded to n_sb_padded_ (multiple of kStride), safe to read kStride elements
            if (!scan_block_ub_any_above_dispatch(block_ub.data() + sb_start, u16_threshold, kStride)) {
                continue;
            }
            // At least one block above threshold — scalar collect
            const uint32_t sb_end = std::min(sb_start + kStride, n_subblocks_);
            for (uint32_t sb = sb_start; sb < sb_end; ++sb) {
                if (block_ub[sb] > u16_threshold) {
                    candidates.push_back(sb);
                    max_ub = std::max(max_ub, block_ub[sb]);
                }
            }
        }
        if (candidates.empty())
            return;

        // Counting sort: sort candidates descending by UB
        const uint32_t range = max_ub - u16_threshold;
        std::vector<uint32_t> counts(range + 1, 0);
        for (uint32_t sb : candidates) {
            counts[block_ub[sb] - u16_threshold - 1]++;
        }
        uint32_t pos = 0;
        for (int b = static_cast<int>(range); b >= 0; --b) {
            uint32_t c = counts[b];
            counts[b] = pos;
            pos += c;
        }
        std::vector<uint32_t> sorted_blocks(candidates.size());
        for (uint32_t sb : candidates) {
            sorted_blocks[counts[block_ub[sb] - u16_threshold - 1]++] = sb;
        }

        DSP_TIMER_MARK(t4);
        // ---- Step 6: Score blocks using forward index ----
        float scores[kSubblockSize];

#ifdef SEEK_INSTRUMENTATION
        uint64_t local_entries = 0;
        uint64_t local_blocks = 0;
        uint64_t local_docs = 0;
        g_dsp_stats.total_blocks += candidates.size();
        g_dsp_stats.queries++;
#endif

        for (size_t ci = 0; ci < sorted_blocks.size(); ++ci) {
            const uint32_t sb_id = sorted_blocks[ci];

            // Early termination: if this block's UB is at or below threshold, all remaining are too
            if (block_ub[sb_id] <= u16_threshold)
                break;

            const uint32_t block_term_start = fwd_block_term_offsets_[sb_id];
            const uint32_t block_term_end = fwd_block_term_offsets_[sb_id + 1];
            if (block_term_start == block_term_end)
                continue;

            // Prefetch next block's forward index data
            if (ci + 1 < sorted_blocks.size()) {
                const uint32_t next_sb = sorted_blocks[ci + 1];
                const uint32_t next_start = fwd_block_term_offsets_[next_sb];
                __builtin_prefetch(&fwd_term_ids_[next_start], 0, 1);
                const uint32_t next_entry_start = fwd_term_entry_offsets_[next_start];
                __builtin_prefetch(&fwd_doc_offsets_[next_entry_start], 0, 0);
                __builtin_prefetch(&fwd_scores_[next_entry_start], 0, 0);
            }

#ifdef SEEK_INSTRUMENTATION
            local_blocks++;
#endif

            std::memset(scores, 0, sizeof(scores));

            // Two-pointer merge of sorted query terms and block terms
            size_t qi = 0;
            uint32_t bi = block_term_start;

            while (qi < n_query_terms && bi < block_term_end) {
                const uint32_t q_dim = query[qi].inner_dim;
                const uint32_t b_dim = fwd_term_ids_[bi];

                if (q_dim < b_dim) {
                    ++qi;
                } else if (q_dim > b_dim) {
                    ++bi;
                } else {
                    const float q_weight = query[qi].weight;
                    const uint32_t e_start = fwd_term_entry_offsets_[bi];
                    const uint32_t e_end = fwd_term_entry_offsets_[bi + 1];

#ifdef SEEK_INSTRUMENTATION
                    local_entries += (e_end - e_start);
#endif
                    for (uint32_t j = e_start; j < e_end; ++j) {
                        scores[fwd_doc_offsets_[j]] += q_weight * fwd_scores_[j];
                    }
                    ++qi;
                    ++bi;
                }
            }

            // Push qualifying docs to heap
            const uint32_t doc_base = sb_id * kSubblockSize;
            const uint32_t doc_end = std::min(doc_base + kSubblockSize, static_cast<uint32_t>(this->n_rows_internal_));
            for (uint32_t i = 0; i < doc_end - doc_base; ++i) {
                if (scores[i] > float_threshold) {
                    const uint32_t doc_id = doc_base + i;
                    if (!filter.empty() && filter.test(doc_id)) {
                        continue;
                    }
#ifdef SEEK_INSTRUMENTATION
                    local_docs++;
#endif
                    heap.push(doc_id, scores[i]);
                    if (heap.full()) {
                        float new_thresh = heap.top().val;
                        if (new_thresh > float_threshold) {
                            float_threshold = new_thresh;
                            u16_threshold = static_cast<uint16_t>(std::min(65535.0f, float_threshold * score_scale));
                        }
                    }
                }
            }
        }  // for sorted_blocks
#ifdef SEEK_INSTRUMENTATION
        g_dsp_stats.blocks_processed += local_blocks;
        g_dsp_stats.entries_scored += local_entries;
        g_dsp_stats.docs_pushed += local_docs;

        DSP_TIMER_MARK(t5);
        // Accumulate per-phase timings using thread-safe atomics
        static std::atomic<uint64_t> us_prep{0}, us_spb{0}, us_ub{0}, us_sort{0}, us_score{0};
        static std::atomic<uint64_t> timer_queries{0};
        us_prep += DSP_TIMER_US(t0, t1);
        us_spb += DSP_TIMER_US(t1, t2);
        us_ub += DSP_TIMER_US(t2, t3);
        us_sort += DSP_TIMER_US(t3, t4);
        us_score += DSP_TIMER_US(t4, t5);
        auto tq = timer_queries.fetch_add(1);
        if (tq > 0 && tq % 6403 == 0) {
            printf("\n[DSP Phase Timing (cumulative %lu queries)]\n", (unsigned long)(tq + 1));
            printf("  prep:  %lu us (%.1f%%)\n", (unsigned long)us_prep.load(),
                   100.0 * us_prep / (us_prep + us_spb + us_ub + us_sort + us_score));
            printf("  spb:   %lu us (%.1f%%)\n", (unsigned long)us_spb.load(),
                   100.0 * us_spb / (us_prep + us_spb + us_ub + us_sort + us_score));
            printf("  ub:    %lu us (%.1f%%)\n", (unsigned long)us_ub.load(),
                   100.0 * us_ub / (us_prep + us_spb + us_ub + us_sort + us_score));
            printf("  sort:  %lu us (%.1f%%)\n", (unsigned long)us_sort.load(),
                   100.0 * us_sort / (us_prep + us_spb + us_ub + us_sort + us_score));
            printf("  score: %lu us (%.1f%%)\n", (unsigned long)us_score.load(),
                   100.0 * us_score / (us_prep + us_spb + us_ub + us_sort + us_score));
        }
#endif
    }
};

}  // namespace knowhere::sparse

#endif  // SPARSE_DSP_INDEX_H
