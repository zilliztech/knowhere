#pragma once

#include <cstdint>
#include <vector>

namespace knowhere::sparse::inverted {

enum class IndexScorerType { UNKNOWN, IP, BM25 };

struct IndexScorerConfig {
    IndexScorerType scorer_type;

    union {
        struct {
            float k1;
            float b;
            float avgdl;
        } bm25;
    } scorer_params;
};

// Query-time scorers are small concrete functors used in hot loops.
// Build-time scorers keep the existing virtual interface for index construction
// and metadata generation paths that are outside query scoring hot paths.
struct IPDimScorer {
    float qval;

    template <typename RType>
    [[nodiscard]] float
    operator()(uint32_t /*rid*/, RType rval) const noexcept {
        return qval * static_cast<float>(rval);
    }
};

struct BM25DimScorer {
    float qval;
    float p1;
    float p2;
    float p3;
    const float* row_sums;

    template <typename RType>
    [[nodiscard]] float
    operator()(uint32_t rid, RType rval) const noexcept {
        const float tf = static_cast<float>(rval);
        return qval * p1 * tf / (tf + p2 + p3 * row_sums[rid]);
    }
};

struct IPQueryScorer {
    static constexpr auto scorer_type = IndexScorerType::IP;

    [[nodiscard]] IPDimScorer
    dim_scorer(float qval) const noexcept {
        return {qval};
    }
};

struct BM25QueryScorer {
    static constexpr auto scorer_type = IndexScorerType::BM25;

    float p1;
    float p2;
    float p3;
    const float* row_sums;

    explicit BM25QueryScorer(float k1, float b, float avgdl, const std::vector<float>& row_sums_in)
        // row_sums stays valid as long as the underlying vector is not reallocated.
        : p1(k1 + 1), p2(k1 * (1 - b)), p3(k1 * b / avgdl), row_sums(row_sums_in.data()) {
    }

    [[nodiscard]] BM25DimScorer
    dim_scorer(float qval) const noexcept {
        return {qval, p1, p2, p3, row_sums};
    }
};

/** Index scorer construct scorers for dimensions in the index. */
class IndexScorer {
 public:
    IndexScorer() = default;
    IndexScorer(const IndexScorer&) = default;
    IndexScorer(IndexScorer&&) noexcept = default;
    IndexScorer&
    operator=(const IndexScorer&) = delete;
    IndexScorer&
    operator=(IndexScorer&&) noexcept = delete;
    virtual ~IndexScorer() = default;
    [[nodiscard]] virtual float
    vec_score(uint32_t rid, float rval) const = 0;

    [[nodiscard]] const IndexScorerConfig&
    config() const {
        return config_;
    }

 protected:
    IndexScorerConfig config_;
};

/** Index scorer for IP. */
struct IPIndexScorer : public IndexScorer {
 public:
    explicit IPIndexScorer() {
        config_.scorer_type = IndexScorerType::IP;
    }

    [[nodiscard]] float
    vec_score(uint32_t rid, float rval) const override {
        return rval;
    }
};

/** Index scorer for BM25. */
struct BM25IndexScorer : public IndexScorer {
 public:
    explicit BM25IndexScorer(const float k1, const float b, const float avgdl, const std::vector<float>& row_sums)
        : p1_(k1 + 1), p2_(k1 * (1 - b)), p3_(k1 * b / avgdl), row_sums_(row_sums) {
        config_.scorer_type = IndexScorerType::BM25;
        config_.scorer_params.bm25.k1 = k1;
        config_.scorer_params.bm25.b = b;
        config_.scorer_params.bm25.avgdl = avgdl;
    }

    ~BM25IndexScorer() override = default;

    [[nodiscard]] float
    vec_score(uint32_t rid, float rval) const override {
        return p1_ * rval / (rval + p2_ + p3_ * row_sums_[rid]);
    }

    [[nodiscard]] const std::vector<float>&
    row_sums() const {
        return row_sums_;
    }

 protected:
    const float p1_;
    const float p2_;
    const float p3_;
    const std::vector<float>& row_sums_;
};
}  // namespace knowhere::sparse::inverted
