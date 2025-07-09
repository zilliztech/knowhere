#pragma once
 
#include <cstddef>
#include <cstdint>
#include <sys/time.h>
#include <sys/syscall.h> 
#include <unistd.h>
 
#include "knowhere/operands.h"
#include "knowhere/expected.h"
#define USE_AMX
namespace faiss {

#define XFEATURE_XTILECFG           17
#define XFEATURE_XTILEDATA          18
#define XFEATURE_MASK_XTILECFG      (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA     (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE         (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM         0x1022
#define ARCH_REQ_XCOMP_PERM         0x1023   


static knowhere::Status enable_amx() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) {
        std::cout << "SYS_arch_prctl(READ) error" << std::endl;
        return knowhere::Status::success;
    }
    if (bitmask & XFEATURE_MASK_XTILEDATA) {
        return knowhere::Status::internal_error;
    }
    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status) {
        return knowhere::Status::internal_error;;
    }
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) {
        return knowhere::Status::internal_error;;
    }
    return knowhere::Status::success;
}
float bf16_vec_inner_product_amx_ref(void **p_bVect1v, void *p_qVect2v, void *dim_ptr, 
        size_t b_Size, size_t q_Size, float * results_amx);
}