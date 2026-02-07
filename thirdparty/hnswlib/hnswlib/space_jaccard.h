#pragma once

#include <faiss/cppcontrib/knowhere/utils/binary_distances.h>

#include "hnswlib.h"

namespace hnswlib {

static float
Jaccard(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return faiss::cppcontrib::knowhere::bvec_jaccard((const uint8_t*)pVect1v, (const uint8_t*)pVect2v, *((size_t*)qty_ptr) / 8);
}

class JaccardSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    JaccardSpace(size_t dim) {
        fstdistfunc_ = Jaccard;
        dim_ = dim;
        data_size_ = (dim + 7)/ 8;
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<float>
    get_dist_func() {
        return fstdistfunc_;
    }

    void*
    get_dist_func_param() {
        return &dim_;
    }

    ~JaccardSpace() {
    }
};

}  // namespace hnswlib
