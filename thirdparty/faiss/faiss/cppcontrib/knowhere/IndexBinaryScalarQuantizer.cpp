// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

#include <faiss/cppcontrib/knowhere/IndexBinaryScalarQuantizer.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/cppcontrib/knowhere/utils/hamming_distance/hamdis-inl.h>
#include <faiss/cppcontrib/knowhere/utils/jaccard-inl.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

namespace {

// Adapter that binds one of baseline's HammingComputerN / JaccardComputerN
// size-specialized primitives into the FlatCodesDistanceComputer interface.
template <class BinaryComputer>
struct BinaryFlatCodesDC : faiss::FlatCodesDistanceComputer {
    BinaryComputer binary_computer;
    std::vector<uint8_t> tmp;

    BinaryFlatCodesDC(const uint8_t* codes_in, size_t code_size_in)
            : FlatCodesDistanceComputer(codes_in, code_size_in),
              tmp(code_size_in) {}

    void set_query(const float* x) final {
        // Legacy convention: each float is an integer in [0, 255]; cast
        // to uint8 to recover the bit-packed query byte. Same pattern
        // used by the fork's BinarySQDistanceComputerWrapper.
        for (size_t i = 0; i < code_size; ++i) {
            tmp[i] = static_cast<uint8_t>(x[i]);
        }
        binary_computer.set(tmp.data(), code_size);
    }

    float distance_to_code(const uint8_t* code) final {
        return binary_computer.compute(code);
    }

    float symmetric_dis(idx_t i, idx_t j) final {
        BinaryComputer temp;
        temp.set(codes + i * code_size, code_size);
        return temp.compute(codes + j * code_size);
    }
};

faiss::FlatCodesDistanceComputer*
make_hamming_dc(const uint8_t* codes, size_t code_size) {
    switch (code_size) {
        case 4:
            return new BinaryFlatCodesDC<HammingComputer4>(codes, code_size);
        case 8:
            return new BinaryFlatCodesDC<HammingComputer8>(codes, code_size);
        case 16:
            return new BinaryFlatCodesDC<HammingComputer16>(codes, code_size);
        case 20:
            return new BinaryFlatCodesDC<HammingComputer20>(codes, code_size);
        case 32:
            return new BinaryFlatCodesDC<HammingComputer32>(codes, code_size);
        case 64:
            return new BinaryFlatCodesDC<HammingComputer64>(codes, code_size);
        default:
            return new BinaryFlatCodesDC<HammingComputerDefault>(
                    codes, code_size);
    }
}

faiss::FlatCodesDistanceComputer*
make_jaccard_dc(const uint8_t* codes, size_t code_size) {
    switch (code_size) {
        case 8:
            return new BinaryFlatCodesDC<JaccardComputer8>(codes, code_size);
        case 16:
            return new BinaryFlatCodesDC<JaccardComputer16>(codes, code_size);
        case 32:
            return new BinaryFlatCodesDC<JaccardComputer32>(codes, code_size);
        case 64:
            return new BinaryFlatCodesDC<JaccardComputer64>(codes, code_size);
        case 128:
            return new BinaryFlatCodesDC<JaccardComputer128>(
                    codes, code_size);
        case 256:
            return new BinaryFlatCodesDC<JaccardComputer256>(
                    codes, code_size);
        case 512:
            return new BinaryFlatCodesDC<JaccardComputer512>(
                    codes, code_size);
        default:
            return new BinaryFlatCodesDC<JaccardComputerDefault>(
                    codes, code_size);
    }
}

} // namespace

IndexBinaryScalarQuantizer::IndexBinaryScalarQuantizer() : IndexFlatCodes() {}

IndexBinaryScalarQuantizer::IndexBinaryScalarQuantizer(int d, MetricType metric)
        : IndexFlatCodes(static_cast<size_t>((d + 7) / 8), d, metric) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_Hamming || metric == METRIC_Jaccard ||
                    metric == METRIC_Substructure ||
                    metric == METRIC_Superstructure,
            "IndexBinaryScalarQuantizer: unsupported metric (expected Hamming, "
            "Jaccard, Substructure, or Superstructure)");
    is_trained = true;
}

void IndexBinaryScalarQuantizer::sa_encode(
        idx_t n, const float* x, uint8_t* bytes) const {
    // Follows the legacy Quantizer1bitDirect convention byte-for-byte:
    // each vector has d floats, but only the first code_size are read;
    // each is cast to uint8 to form the code byte.
    const size_t cs = code_size;
    for (idx_t vi = 0; vi < n; ++vi) {
        const float* src = x + vi * static_cast<idx_t>(d);
        uint8_t* dst = bytes + vi * cs;
        for (size_t i = 0; i < cs; ++i) {
            dst[i] = static_cast<uint8_t>(src[i]);
        }
    }
}

void IndexBinaryScalarQuantizer::sa_decode(
        idx_t n, const uint8_t* bytes, float* x) const {
    // Mirror of sa_encode. Output stride is d (matching baseline
    // ScalarQuantizer::decode) but only the first code_size lanes of
    // each d-float slot are written. Trailing lanes are left untouched
    // by design: callers that only need the meaningful bytes (see
    // faiss_hnsw.cc GetVectorByIds, bin1 branch) allocate exactly
    // code_size floats per vector and rely on the decoder not writing
    // past that. Zero-filling the tail would overrun those buffers.
    const size_t cs = code_size;
    for (idx_t vi = 0; vi < n; ++vi) {
        float* dst = x + vi * static_cast<idx_t>(d);
        const uint8_t* src = bytes + vi * cs;
        for (size_t i = 0; i < cs; ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
    }
}

faiss::FlatCodesDistanceComputer*
IndexBinaryScalarQuantizer::get_FlatCodesDistanceComputer() const {
    switch (metric_type) {
        case METRIC_Hamming:
        case METRIC_Substructure:
        case METRIC_Superstructure:
            return make_hamming_dc(codes.data(), code_size);
        case METRIC_Jaccard:
            return make_jaccard_dc(codes.data(), code_size);
        default:
            FAISS_THROW_MSG(
                    "IndexBinaryScalarQuantizer: unsupported metric in "
                    "get_FlatCodesDistanceComputer");
    }
}

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
