#pragma once

#include <cstdint>
#include <functional>

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

/**
 * DimScorer is a function that computes a relevance score for a vector.
 * @param uint32_t The vector ID
 * @param float The vector's dimension value, such as term frequency in BM25 or value in IP
 * @return float The computed relevance score
 */
using DimScorer = std::function<float(uint32_t, float)>;

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
    [[nodiscard]] virtual DimScorer
    dim_scorer(float qval) const = 0;
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

    [[nodiscard]] DimScorer
    dim_scorer(float qval) const override {
        return [qval](uint32_t rid, float rval) { return qval * rval; };
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

    // In senario of BM25, qval is IDF value, rval is TF value
    [[nodiscard]] DimScorer
    dim_scorer(float qval) const override {
        return
            [&, qval](uint32_t rid, uint32_t rval) { return qval * p1_ * rval / (rval + p2_ + p3_ * row_sums_[rid]); };
    }

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
