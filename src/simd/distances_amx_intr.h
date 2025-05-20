
#pragma once
#include <stdlib.h>
#include <mutex>
#include <shared_mutex>
#include <cstring> 
#include <immintrin.h> 
#include <sys/time.h>
#include <sys/syscall.h> 
#include <unistd.h>
#include <iostream>



// static bool is_onednn_init = false;
// static std::mutex init_mutex;

#define u64 unsigned long long
#define u8  unsigned char
#define u16 unsigned short int

#define XFEATURE_XTILECFG           17
#define XFEATURE_XTILEDATA          18
#define XFEATURE_MASK_XTILECFG      (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA     (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE         (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM         0x1022
#define ARCH_REQ_XCOMP_PERM         0x1023   

using namespace std;
static uint64_t get_amx_xcr_reg(void) {
    // 调用 _xgetbv 内建函数，参数0表示读取XCR0
    return _xgetbv(0);
}
static int enable_amx() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) {
        std::cout << "SYS_arch_prctl(READ) error" << std::endl;
        return 0;
    }
    if (bitmask & XFEATURE_MASK_XTILEDATA) {
        return 1;
    }
    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status) {
        std::cout << "SYS_arch_prctl(WRITE) error" << std::endl;
        return 0;
    }
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) {
        std::cout << "SYS_arch_prctl(READ) error" << std::endl;
        return 0;
    }
    return 1;
}

// static bool is_config = false;

static void amx_bf16_mul(void *cfg, void *ma, void *mb, int64_t a_stride, int64_t b_stride, void *mc) {
/*     if (cfg != NULL) {
        _tile_loadconfig((void *)cfg);
    } */

    // 使用 AMX 指令进行矩阵乘法
    // 注意：这里的函数名是假设性的，实际使用时需要根据 oneAPI 文档来确定
    _tile_loadd(0,ma, a_stride);
    _tile_loadd(1,mb, b_stride);
    _tile_dpbf16ps(2,0,1);
    //_tile_stored(2, mc, b_stride);
    //_tile_release();


    return ;  // 返回 mc 的地址
}