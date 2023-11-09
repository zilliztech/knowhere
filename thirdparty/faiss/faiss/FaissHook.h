
// -*- c++ -*-

#pragma once

#include <string>

#include <faiss/MetricType.h>
#include <faiss/impl/ScalarQuantizer.h>
#include "simd/hook.h"
namespace faiss {

// todo aguzhva: replace FaissHook.h with simd/hook.h

struct IDSelector;

typedef ScalarQuantizer::SQDistanceComputer* (*sq_get_distance_computer_func_ptr)(
        MetricType,
        ScalarQuantizer::QuantizerType,
        size_t,
        const std::vector<float>&);
typedef ScalarQuantizer::SQuantizer* (*sq_sel_quantizer_func_ptr)(
        ScalarQuantizer::QuantizerType,
        size_t,
        const std::vector<float>&);
typedef InvertedListScanner* (*sq_sel_inv_list_scanner_func_ptr)(
        MetricType,
        const ScalarQuantizer*,
        const Index*,
        size_t,
        bool,
        const IDSelector*,
        bool);

extern sq_get_distance_computer_func_ptr sq_get_distance_computer;
extern sq_sel_quantizer_func_ptr sq_sel_quantizer;
extern sq_sel_inv_list_scanner_func_ptr sq_sel_inv_list_scanner;
void sq_hook();
} // namespace faiss
