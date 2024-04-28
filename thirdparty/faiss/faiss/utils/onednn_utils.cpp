#ifdef FAISS_WITH_DNNL
#include "onednn_utils.h"

thread_local faiss::inner_product_desc inner_product_desc_t;

void faiss::comput_f32bf16f32_inner_product(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    float* in_f32_1, float* in_f32_2, float** out_f32) {
    inner_product_desc_t.init(xrow, xcol, yrow, ycol, in_f32_1, in_f32_2);
    inner_product_desc_t.execut(out_f32);
}
#endif
