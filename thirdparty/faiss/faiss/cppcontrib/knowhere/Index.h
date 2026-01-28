#pragma once

#include <faiss/Index.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

struct HasIsCosine2 {
    /// both IP and COSINE are regarded as INNER_PRODUCT in faiss
    bool is_cosine = false;
};

static inline bool whether_index_has_cosine_enabled(const faiss::Index* index) {
    const HasIsCosine2* has_cosine = dynamic_cast<const HasIsCosine2*>(index);
    return (has_cosine != nullptr && has_cosine->is_cosine);
}

// a special one to track the root of our changes.
// must be removed once we're done
struct Index : faiss::Index, HasIsCosine2 {
    explicit Index(idx_t d = 0, MetricType metric = METRIC_L2) : faiss::Index(d, metric) {}
};

}
}
}
