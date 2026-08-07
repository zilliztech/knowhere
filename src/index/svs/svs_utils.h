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

#ifndef SVS_UTILS_H
#define SVS_UTILS_H

#ifdef KNOWHERE_WITH_SVS

#include <optional>
#include <string>

#include "faiss/svs/IndexSVSVamana.h"

namespace knowhere {

// Maps a knowhere `svs_storage_kind` config string onto the faiss enum. Shared by every SVS index node so that
// newly supported storage formats only need to be added in one place. faiss owns the enum -> SVS runtime half of
// the mapping (`faiss::to_svs_storage_kind`).
inline std::optional<faiss::SVSStorageKind>
str_to_svs_storage_kind(const std::string& s) {
    if (s == "fp32")
        return faiss::SVS_FP32;
    if (s == "fp16")
        return faiss::SVS_FP16;
    if (s == "sqi8")
        return faiss::SVS_SQI8;
    if (s == "lvq4x0")
        return faiss::SVS_LVQ4x0;
    if (s == "lvq4x4")
        return faiss::SVS_LVQ4x4;
    if (s == "lvq4x8")
        return faiss::SVS_LVQ4x8;
    if (s == "lvq8x0")
        return faiss::SVS_LVQ8x0;
    if (s == "leanvec4x4")
        return faiss::SVS_LeanVec4x4;
    if (s == "leanvec4x8")
        return faiss::SVS_LeanVec4x8;
    if (s == "leanvec8x8")
        return faiss::SVS_LeanVec8x8;
    return std::nullopt;
}

}  // namespace knowhere

#endif  // KNOWHERE_WITH_SVS

#endif  // SVS_UTILS_H
