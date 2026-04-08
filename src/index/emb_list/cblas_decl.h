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

// Minimal CBLAS declarations to avoid platform-specific header issues (e.g. missing cblas.h on macOS).
// On platforms where the system cblas.h exists and is transitively included, the include guard
// prevents duplicate definitions. On platforms where cblas.h is unavailable, this provides the
// necessary declarations.

#ifndef KNOWHERE_CBLAS_DECL_H
#define KNOWHERE_CBLAS_DECL_H

#ifdef CBLAS_H
// System cblas.h was already included — nothing to do.
#else
// Define the system include guard so that if cblas.h is included later, it will be skipped.
#define CBLAS_H

extern "C" {

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };

void
cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, int M, int N, int K,
            float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);

void
cblas_strsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side, enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
            enum CBLAS_DIAG Diag, int M, int N, float alpha, const float* A, int lda, float* B, int ldb);

void
cblas_saxpy(int N, float alpha, const float* X, int incX, float* Y, int incY);

}  // extern "C"

#endif  // CBLAS_H

#endif  // KNOWHERE_CBLAS_DECL_H
