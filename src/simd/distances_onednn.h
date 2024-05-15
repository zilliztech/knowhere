/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* All distance functions for L2 and IP distances.
 * The actual functions are implemented in distances.cpp and distances_simd.cpp
 */

#pragma once
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <atomic>
#include <memory>
#include <pthread.h>
#include "oneapi/dnnl/dnnl.hpp"
#include <faiss/impl/ResultHandler.h>
#include <mm_malloc.h>
#include <unistd.h>

namespace faiss {
static dnnl::engine cpu_engine;
static dnnl::stream engine_stream;
static bool is_onednn_init = false;
static std::mutex init_mutex;

enum DNNL_STATE {
    DNNL_UNSUPPORTED = false,
    DNNL_SUPPORTED = true,
    DNNL_UNKOWN = 99
};
static DNNL_STATE dnnl_state = DNNL_STATE::DNNL_UNKOWN;
static bool is_dnnl_enabled() {
    if (dnnl_state == DNNL_STATE::DNNL_UNKOWN) [[unlikely]] {
        char* env = getenv("DNNL_ENABLE");
        if (env != NULL && strcmp(env, "1") == 0) {
            dnnl_state = DNNL_STATE::DNNL_SUPPORTED;
        } else {
            dnnl_state = DNNL_STATE::DNNL_UNSUPPORTED;
        }
    }
    return dnnl_state;
}

static void init_onednn() {
	std::unique_lock<std::mutex> lock(init_mutex);

    if (is_onednn_init) {
        return;
    }

    printf("init onednn\n");

    // init onednn env
    cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    engine_stream = dnnl::stream(cpu_engine);

    is_onednn_init = true;
}

__attribute__((constructor))
static void library_load() {
    // this function will be called as long as the faiss lib is loaded
    // printf("Library loaded.\n");
    if (is_dnnl_enabled()) {
        init_onednn();
    }
}

/**
 * @brief Compute float32 matrix inner product with bf16 intermediate results to accelerate
 * @details The main idea is:
 * 1. Define float32 memory layout for input and output
 * 2. Create low precision bf16 memory descriptors as inner product input
 * 3. Generate inner product primitive descriptor
 * 4. Execute float32 => (reorder) => bf16 => (inner product) => float32
 *    chain operation, isolate different precision data, accelerate inner product
 * 5. Pipeline execution via streams for asynchronous scheduling
 *
 * @param xrow Row number of input matrix X
 * @param xcol Column number of input matrix X
 * @param yrow Row number of weight matrix Y
 * @param ycol Column number of weight matrix Y
 * @param in_f32_1 Input matrix pointer in float32 type
 * @param in_f32_2 Weight matrix pointer in float32 type
 * @param out_f32 Output matrix pointer for result in float32 type
 * @return None
 */

enum BASE_DATA_STATE {
    INIT,
    MODIFIED,
    PREPARE,
    READY
};

class BaseData {
 public:
    static std::atomic<BASE_DATA_STATE>& getState() {
        static std::atomic<BASE_DATA_STATE> is_base_changed(BASE_DATA_STATE::INIT);
        return is_base_changed;
    }
};

static dnnl::memory bf16_mem2;
static pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

struct inner_product_desc {
    uint32_t xrow;
    uint32_t xcol;
    uint32_t yrow;
    uint32_t ycol;
    float* in_f32_1;
    float* in_f32_2;
    float* out_f32;
    dnnl::memory::desc f32_md1;
    dnnl::memory::desc f32_md2;
    dnnl::memory::desc f32_dst_md2;
    dnnl::memory f32_mem1;
    dnnl::memory f32_mem2;
    dnnl::memory f32_dst_mem;

    dnnl::memory::desc bf16_md1;
    dnnl::memory::desc bf16_md2;

    dnnl::inner_product_forward::primitive_desc inner_product_pd;
    dnnl::inner_product_forward inner_product_prim;

    dnnl::memory bf16_mem1;

    bool is_same(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
      float* in_f32_1, float* in_f32_2) {
      return this->xrow == xrow && this->xcol == xcol &&
             this->yrow == yrow && this->ycol == ycol &&
             this->in_f32_2 == in_f32_2;
    }

    void init(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
		float* in_f32_1, float* in_f32_2) {
        if (is_same(xrow, xcol, yrow, ycol, in_f32_1, in_f32_2)) {
            if (this->in_f32_1 != in_f32_1) {
                this->in_f32_1 = in_f32_1;
                f32_mem1 = dnnl::memory(f32_md1, cpu_engine, in_f32_1);
            }
            return;
        }

        this->xrow = xrow;
        this->xcol = xcol;
        this->yrow = yrow;
        this->ycol = ycol;
        this->in_f32_1 = in_f32_1;
        this->in_f32_2 = in_f32_2;

        f32_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
        f32_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
        f32_dst_md2 = dnnl::memory::desc({xrow, yrow}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

        f32_mem1 = dnnl::memory(f32_md1, cpu_engine, in_f32_1);
        f32_mem2 = dnnl::memory(f32_md2, cpu_engine, in_f32_2);
        f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine);

        // inner memory bf16
        bf16_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
        bf16_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);

        inner_product_pd = dnnl::inner_product_forward::primitive_desc(
              cpu_engine, dnnl::prop_kind::forward_training,
              bf16_md1, bf16_md2, f32_dst_md2);

        inner_product_prim = dnnl::inner_product_forward(inner_product_pd);

        bf16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine);

        // update state for new base data
        if (this->in_f32_2 != in_f32_2)
            BaseData::getState().store(BASE_DATA_STATE::MODIFIED);
    }

    void execut(float** out_f32) {
        dnnl::reorder(f32_mem1, bf16_mem1).execute(engine_stream, f32_mem1, bf16_mem1);
        BASE_DATA_STATE expected = BASE_DATA_STATE::MODIFIED;

        if (BaseData::getState().compare_exchange_strong(expected, BASE_DATA_STATE::PREPARE)) {
            pthread_rwlock_wrlock(&rwlock);

            bf16_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine,  DNNL_MEMORY_ALLOCATE);
            dnnl::reorder(f32_mem2, bf16_mem2).execute(engine_stream, f32_mem2, bf16_mem2);
            inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, bf16_mem1},
                                                          {DNNL_ARG_WEIGHTS, bf16_mem2},
                                                          {DNNL_ARG_DST, f32_dst_mem}});
            pthread_rwlock_unlock(&rwlock);
            BaseData::getState().store(BASE_DATA_STATE::READY);
        } else {
            while(BaseData::getState() != BASE_DATA_STATE::READY) {
                usleep(50000);
            }

            pthread_rwlock_rdlock(&rwlock);
            inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, bf16_mem1},
                                                      {DNNL_ARG_WEIGHTS, bf16_mem2},
                                                      {DNNL_ARG_DST, f32_dst_mem}});
            pthread_rwlock_unlock(&rwlock);
        }

        *out_f32 = (float*) f32_dst_mem.get_data_handle();
    }
};

void fvec_f32bf16f32_inner_product_onednn(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
                                          float* in_f32_1, float* in_f32_2, float** out_f32);

} // namespace faiss

extern thread_local faiss::inner_product_desc inner_product_desc_t;
