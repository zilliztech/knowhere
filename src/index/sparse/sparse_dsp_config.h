// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef SPARSE_DSP_CONFIG_H
#define SPARSE_DSP_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

// Search parameters for the DSP (Dynamic Superblock Pruning) index.
//
// Mode selection (dsp_mode):
//   0 = DSP:  dual-threshold (mu, eta) superblock pruning (SIGIR'25 paper).
//             Safe with default mu=1, eta=1. Set dsp_gamma>0 for a top-gamma backstop.
//   1 = LSP/0: top-gamma superblock inclusion only, no mu/eta gates.
//             Recommended for SPLADE-family models. Simplest and generally fastest.
//             k=10:   gamma=250 (~99% recall) or gamma=500 (near-safe)
//             k=1000: gamma=1000 (~99% recall) or gamma=2000 (near-safe)
//   2 = LSP/1: LSP/0 + mu-overestimation gate (ub > theta/mu).
//   3 = LSP/2: LSP/1 + ASC gate (ub > theta/mu || asc > theta/eta).
//
// For all modes, dsp_eta also controls subblock BoundSum pruning.
// dsp_kth_init seeds the pruning threshold from per-dimension kth-largest scores (orthogonal to mode).
class SparseDspConfig : public BaseConfig {
 public:
    CFG_FLOAT drop_ratio_search;
    CFG_INT refine_factor;
    CFG_INT dsp_mode;
    CFG_FLOAT dsp_mu;
    CFG_FLOAT dsp_eta;
    CFG_INT dsp_gamma;
    CFG_BOOL dsp_kth_init;
    CFG_FLOAT dsp_kth_alpha;
    KNOHWERE_DECLARE_CONFIG(SparseDspConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(drop_ratio_search)
            .description("drop ratio for search")
            .set_default(0.0f)
            .set_range(0.0f, 1.0f, true, false)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_factor)
            .description("refine factor for approximate search")
            .set_default(1)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(dsp_mode)
            .set_range(0, 3)
            .set_default(0)
            .description(
                "superblock selection mode: "
                "0=dsp (dual-threshold mu/eta + optional top-gamma backstop), "
                "1=lsp0 (top-gamma from ub>=theta, no mu/asc gate), "
                "2=lsp1 (lsp0 + mu-overestimation gate: ub>theta/mu), "
                "3=lsp2 (lsp1 + asc gate: ub>theta/mu || asc>theta/eta)")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(dsp_mu)
            .set_range(0.1, 2.0)
            .set_default(1.0)
            .description("superblock max-based threshold relaxation factor (used by dsp/lsp1/lsp2)")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(dsp_eta)
            .set_range(0.1, 2.0)
            .set_default(1.0)
            .description(
                "threshold relaxation for superblock ASC pruning (dsp/lsp2) "
                "and subblock BoundSum pruning (all modes)")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(dsp_gamma)
            .set_range(0, 100000)
            .set_default(0)
            .description(
                "always visit top-gamma superblocks by UB score; "
                "recommended: 250-500 for k=10, 1000-2000 for k=1000 "
                "(0 = disabled)")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(dsp_kth_init)
            .set_default(true)
            .description(
                "enable kth-score threshold initialization before pruning "
                "(false = start threshold at 0, orthogonal to mode)")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(dsp_kth_alpha)
            .set_range(0.0, 1.0)
            .set_default(1.0)
            .description(
                "scale factor for kth-score threshold seed: threshold *= alpha "
                "(1.0 = full seed, 0.0 = no seed, intermediate = calibrated)")
            .for_search();
    }
};  // class SparseDspConfig

}  // namespace knowhere

#endif  // SPARSE_DSP_CONFIG_H
