#pragma once

#include <cstdint>
#include <functional>

namespace knowhere::sparse::pisa {

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
};

/** Index scorer for IP. */
struct IPIndexScorer : public IndexScorer {
 public:
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
        : k1_(k1), b_(b), avgdl_(avgdl), row_sums_(row_sums) {
    }
    ~BM25IndexScorer() override = default;

    // In senario of BM25, qval is IDF value, rval is TF value
    [[nodiscard]] DimScorer
    dim_scorer(float qval) const override {
        return [&, qval](uint32_t rid, uint32_t rval) {
            return qval * (k1_ + 1) * rval / (rval + k1_ * (1 - b_ + b_ * row_sums_[rid] / avgdl_));
        };
    }

    [[nodiscard]] float
    vec_score(uint32_t rid, float rval) const override {
        return (k1_ + 1) * rval / (rval + k1_ * (1 - b_ + b_ * row_sums_[rid] / avgdl_));
    }

 protected:
    const float k1_;
    const float b_;
    const float avgdl_;
    const std::vector<float>& row_sums_;
};
}  // namespace knowhere::sparse::pisa
