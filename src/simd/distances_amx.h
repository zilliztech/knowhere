#pragma once
 
#include <cstddef>
#include <cstdint>
 
#include "knowhere/operands.h"
#define USE_AMX
namespace faiss {
    float bf16_vec_inner_product_amx_ref(void **p_bVect1v, void *p_qVect2v, void *dim_ptr, 
        size_t b_Size, size_t q_Size, float * results_amx);
}