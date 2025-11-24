
// -*- c++ -*-

#include <iostream>
#include <mutex>

#include <faiss/FaissHook.h>
#include <faiss/impl/ScalarQuantizerDC.h>
#include <faiss/impl/ScalarQuantizerDC_avx.h>
#include <faiss/impl/ScalarQuantizerDC_avx512.h>
#include <faiss/impl/ScalarQuantizerDC_neon.h>
#include <faiss/impl/ScalarQuantizerDC_rvv.h>
namespace faiss {

sq_get_distance_computer_func_ptr sq_get_distance_computer =
        sq_get_distance_computer_ref;
sq_sel_quantizer_func_ptr sq_sel_quantizer = sq_select_quantizer_ref;
sq_sel_inv_list_scanner_func_ptr sq_sel_inv_list_scanner =
        sq_select_inverted_list_scanner_ref;

// Note: The Hamming computer implementation is selected at compile time
// based on the instruction set in `hamdis-inl.h`, not by runtime hook.
sq_get_distance_computer_func_ptr sq_get_hamming_distance_computer =
        sq_get_hamming_distance_computer_ref;

// Note: The Jaccard distance computer uses `__builtin_popcount` for 
// computation. This function is efficiently implemented by the
// compiler and automatically utilizes the best available instruction set.
// Therefore, there is no need to manually adjust or hook the Jaccard computer
// for different SIMD instruction sets.
sq_get_distance_computer_func_ptr sq_get_jaccard_distance_computer =
        sq_get_jaccard_distance_computer_ref;

void sq_hook() {
    // SQ8 always hook best SIMD
#ifdef __x86_64__
    static std::mutex hook_mutex;
    std::lock_guard<std::mutex> lock(hook_mutex);

    if (use_avx512 && cpu_support_avx512()) {
        /* for IVFSQ */
        sq_get_distance_computer = sq_get_distance_computer_avx512;
        sq_sel_quantizer = sq_select_quantizer_avx512;
        sq_sel_inv_list_scanner = sq_select_inverted_list_scanner_avx512;
    } else if (use_avx2 && cpu_support_avx2()) {
        /* for IVFSQ */
        sq_get_distance_computer = sq_get_distance_computer_avx;
        sq_sel_quantizer = sq_select_quantizer_avx;
        sq_sel_inv_list_scanner = sq_select_inverted_list_scanner_avx;
    } else if (use_sse4_2 && cpu_support_sse4_2()) {
        /* for IVFSQ */
        sq_get_distance_computer = sq_get_distance_computer_ref;
        sq_sel_quantizer = sq_select_quantizer_ref;
        sq_sel_inv_list_scanner = sq_select_inverted_list_scanner_ref;
    }
#endif

#if defined(__ARM_NEON)
    static std::mutex hook_mutex;
    std::lock_guard<std::mutex> lock(hook_mutex);

    /* for IVFSQ */
    sq_get_distance_computer = sq_get_distance_computer_neon;
    sq_sel_quantizer = sq_select_quantizer_neon;
    sq_sel_inv_list_scanner = sq_select_inverted_list_scanner_neon;
#endif

#if defined(__riscv_vector)
    static std::mutex hook_mutex;
    std::lock_guard<std::mutex> lock(hook_mutex);
    /* for IVFSQ */
    sq_get_distance_computer = sq_get_distance_computer_rvv;
    sq_sel_quantizer = sq_select_quantizer_rvv;
    sq_sel_inv_list_scanner = sq_select_inverted_list_scanner_rvv;
#endif
}


} // namespace faiss

