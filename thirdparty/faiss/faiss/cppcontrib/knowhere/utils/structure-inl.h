// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#ifndef FAISS_STRUCTURE_INL_H
#define FAISS_STRUCTURE_INL_H

#include <faiss/cppcontrib/knowhere/utils/binary_distances.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

template<bool is_super>
struct StructureComputer8 {
    uint64_t a0;

    StructureComputer8() {}

    StructureComputer8(const uint8_t *a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t *a8, int code_size) {
        assert(code_size == 8);
        const uint64_t *a = (uint64_t *)a8;
        a0 = a[0];
    }

    inline bool compute(const uint8_t *b8) const {
        const uint64_t *b = (uint64_t *)b8;
        const uint64_t& r0 = is_super ? b[0] : a0;
        return (a0 & b[0]) == r0;
    }
};

template<bool is_super>
struct StructureComputer16 {
    uint64_t a0, a1;

    StructureComputer16() {}

    StructureComputer16(const uint8_t *a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t *a8, int code_size) {
        assert(code_size == 16);
        const uint64_t *a = (uint64_t *)a8;
        a0 = a[0]; a1 = a[1];
    }

    inline bool compute(const uint8_t *b8) const {
        const uint64_t *b = (uint64_t *)b8;
        const uint64_t& r0 = is_super ? b[0] : a0;
        const uint64_t& r1 = is_super ? b[1] : a1;
        return (a0 & b[0]) == r0 && (a1 & b[1]) == r1;
    }
};

template<bool is_super>
struct StructureComputer32 {
    uint64_t a0, a1, a2, a3;

    StructureComputer32() {}

    StructureComputer32(const uint8_t *a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t *a8, int code_size) {
        assert(code_size == 32);
        const uint64_t *a = (uint64_t *)a8;
        a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
    }

    inline bool compute(const uint8_t *b8) const {
        const uint64_t *b = (uint64_t *)b8;
        const uint64_t& r0 = is_super ? b[0] : a0;
        const uint64_t& r1 = is_super ? b[1] : a1;
        const uint64_t& r2 = is_super ? b[2] : a2;
        const uint64_t& r3 = is_super ? b[3] : a3;
        return (a0 & b[0]) == r0 && (a1 & b[1]) == r1 &&
                (a2 & b[2]) == r2 && (a3 & b[3]) == r3;
    }
};

template<bool is_super>
struct StructureComputer64 {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;

    StructureComputer64() {}

    StructureComputer64(const uint8_t *a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t *a8, int code_size) {
        assert(code_size == 64);
        const uint64_t *a = (uint64_t *)a8;
        a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
        a4 = a[4]; a5 = a[5]; a6 = a[6]; a7 = a[7];
    }

    inline bool compute(const uint8_t *b8) const {
        const uint64_t *b = (uint64_t *)b8;
        const uint64_t& r0 = is_super ? b[0] : a0;
        const uint64_t& r1 = is_super ? b[1] : a1;
        const uint64_t& r2 = is_super ? b[2] : a2;
        const uint64_t& r3 = is_super ? b[3] : a3;
        const uint64_t& r4 = is_super ? b[4] : a4;
        const uint64_t& r5 = is_super ? b[5] : a5;
        const uint64_t& r6 = is_super ? b[6] : a6;
        const uint64_t& r7 = is_super ? b[7] : a7;
        return (a0 & b[0]) == r0 && (a1 & b[1]) == r1 &&
                (a2 & b[2]) == r2 && (a3 & b[3]) == r3 &&
                (a4 & b[4]) == r4 && (a5 & b[5]) == r5 &&
                (a6 & b[6]) == r6 && (a7 & b[7]) == r7;
    }
};

template<bool is_super>
struct StructureComputer128 {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7,
            a8, a9, a10, a11, a12, a13, a14, a15;

    StructureComputer128() {}

    StructureComputer128(const uint8_t *a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t *au8, int code_size) {
        assert(code_size == 128);
        const uint64_t *a = (uint64_t *)au8;
        a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
        a4 = a[4]; a5 = a[5]; a6 = a[6]; a7 = a[7];
        a8 = a[8]; a9 = a[9]; a10 = a[10]; a11 = a[11];
        a12 = a[12]; a13 = a[13]; a14 = a[14]; a15 = a[15];
    }

    inline float compute(const uint8_t *b8) const {
        const uint64_t *b = (uint64_t *)b8;
        const uint64_t& r0 = is_super ? b[0] : a0;
        const uint64_t& r1 = is_super ? b[1] : a1;
        const uint64_t& r2 = is_super ? b[2] : a2;
        const uint64_t& r3 = is_super ? b[3] : a3;
        const uint64_t& r4 = is_super ? b[4] : a4;
        const uint64_t& r5 = is_super ? b[5] : a5;
        const uint64_t& r6 = is_super ? b[6] : a6;
        const uint64_t& r7 = is_super ? b[7] : a7;
        const uint64_t& r8 = is_super ? b[8] : a8;
        const uint64_t& r9 = is_super ? b[9] : a9;
        const uint64_t& r10 = is_super ? b[10] : a10;
        const uint64_t& r11 = is_super ? b[11] : a11;
        const uint64_t& r12 = is_super ? b[12] : a12;
        const uint64_t& r13 = is_super ? b[13] : a13;
        const uint64_t& r14 = is_super ? b[14] : a14;
        const uint64_t& r15 = is_super ? b[15] : a15;
        return (a0 & b[0]) == r0 && (a1 & b[1]) == r1 &&
                (a2 & b[2]) == r2 && (a3 & b[3]) == r3 &&
                (a4 & b[4]) == r4 && (a5 & b[5]) == r5 &&
                (a6 & b[6]) == r6 && (a7 & b[7]) == r7 &&
                (a8 & b[8]) == r8 && (a9 & b[9]) == r9 &&
                (a10 & b[10]) == r10 && (a11 & b[11]) == r11 &&
                (a12 & b[12]) == r12 && (a13 & b[13]) == r13 &&
                (a14 & b[14]) == r14 && (a15 & b[15]) == r15;
    }
};

template<bool is_super>
struct StructureComputer256 {
    uint64_t a0,a1,a2,a3,a4,a5,a6,a7,
        a8,a9,a10,a11,a12,a13,a14,a15,
        a16,a17,a18,a19,a20,a21,a22,a23,
        a24,a25,a26,a27,a28,a29,a30,a31;

    StructureComputer256() {}

    StructureComputer256(const uint8_t *a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t *au8, int code_size) {
        assert(code_size == 256);
        const uint64_t *a = (uint64_t *)au8;
        a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
        a4 = a[4]; a5 = a[5]; a6 = a[6]; a7 = a[7];
        a8 = a[8]; a9 = a[9]; a10 = a[10]; a11 = a[11];
        a12 = a[12]; a13 = a[13]; a14 = a[14]; a15 = a[15];
        a16 = a[16]; a17 = a[17]; a18 = a[18]; a19 = a[19];
        a20 = a[20]; a21 = a[21]; a22 = a[22]; a23 = a[23];
        a24 = a[24]; a25 = a[25]; a26 = a[26]; a27 = a[27];
        a28 = a[28]; a29 = a[29]; a30 = a[30]; a31 = a[31];
    }

    inline float compute(const uint8_t *b8) const {
        const uint64_t *b = (uint64_t *)b8;
        const uint64_t& r0 = is_super ? b[0] : a0;
        const uint64_t& r1 = is_super ? b[1] : a1;
        const uint64_t& r2 = is_super ? b[2] : a2;
        const uint64_t& r3 = is_super ? b[3] : a3;
        const uint64_t& r4 = is_super ? b[4] : a4;
        const uint64_t& r5 = is_super ? b[5] : a5;
        const uint64_t& r6 = is_super ? b[6] : a6;
        const uint64_t& r7 = is_super ? b[7] : a7;
        const uint64_t& r8 = is_super ? b[8] : a8;
        const uint64_t& r9 = is_super ? b[9] : a9;
        const uint64_t& r10 = is_super ? b[10] : a10;
        const uint64_t& r11 = is_super ? b[11] : a11;
        const uint64_t& r12 = is_super ? b[12] : a12;
        const uint64_t& r13 = is_super ? b[13] : a13;
        const uint64_t& r14 = is_super ? b[14] : a14;
        const uint64_t& r15 = is_super ? b[15] : a15;
        const uint64_t& r16 = is_super ? b[16] : a16;
        const uint64_t& r17 = is_super ? b[17] : a17;
        const uint64_t& r18 = is_super ? b[18] : a18;
        const uint64_t& r19 = is_super ? b[19] : a19;
        const uint64_t& r20 = is_super ? b[20] : a20;
        const uint64_t& r21 = is_super ? b[21] : a21;
        const uint64_t& r22 = is_super ? b[22] : a22;
        const uint64_t& r23 = is_super ? b[23] : a23;
        const uint64_t& r24 = is_super ? b[24] : a24;
        const uint64_t& r25 = is_super ? b[25] : a25;
        const uint64_t& r26 = is_super ? b[26] : a26;
        const uint64_t& r27 = is_super ? b[27] : a27;
        const uint64_t& r28 = is_super ? b[28] : a28;
        const uint64_t& r29 = is_super ? b[29] : a29;
        const uint64_t& r30 = is_super ? b[30] : a30;
        const uint64_t& r31 = is_super ? b[31] : a31;
        return (a0 & b[0]) == r0 && (a1 & b[1]) == r1 &&
                (a2 & b[2]) == r2 && (a3 & b[3]) == r3 &&
                (a4 & b[4]) == r4 && (a5 & b[5]) == r5 &&
                (a6 & b[6]) == r6 && (a7 & b[7]) == r7 &&
                (a8 & b[8]) == r8 && (a9 & b[9]) == r9 &&
                (a10 & b[10]) == r10 && (a11 & b[11]) == r11 &&
                (a12 & b[12]) == r12 && (a13 & b[13]) == r13 &&
                (a14 & b[14]) == r14 && (a15 & b[15]) == r15 &&
                (a16 & b[16]) == r16 && (a17 & b[17]) == r17 &&
                (a18 & b[18]) == r18 && (a19 & b[19]) == r19 &&
                (a20 & b[20]) == r20 && (a21 & b[21]) == r21 &&
                (a22 & b[22]) == r22 && (a23 & b[23]) == r23 &&
                (a24 & b[24]) == r24 && (a25 & b[25]) == r25 &&
                (a26 & b[26]) == r26 && (a27 & b[27]) == r27 &&
                (a28 & b[28]) == r28 && (a29 & b[29]) == r29 &&
                (a30 & b[30]) == r30 && (a31 & b[31]) == r31;
    }
};

template<bool is_super>
struct StructureComputer512 {
    uint64_t a0,a1,a2,a3,a4,a5,a6,a7,
        a8,a9,a10,a11,a12,a13,a14,a15,
        a16,a17,a18,a19,a20,a21,a22,a23,
        a24,a25,a26,a27,a28,a29,a30,a31,
        a32,a33,a34,a35,a36,a37,a38,a39,
        a40,a41,a42,a43,a44,a45,a46,a47,
        a48,a49,a50,a51,a52,a53,a54,a55,
        a56,a57,a58,a59,a60,a61,a62,a63;

    StructureComputer512() {}

    StructureComputer512(const uint8_t *a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t *au8, int code_size) {
        assert(code_size == 512);
        const uint64_t *a = (uint64_t *)au8;
        a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
        a4 = a[4]; a5 = a[5]; a6 = a[6]; a7 = a[7];
        a8 = a[8]; a9 = a[9]; a10 = a[10]; a11 = a[11];
        a12 = a[12]; a13 = a[13]; a14 = a[14]; a15 = a[15];
        a16 = a[16]; a17 = a[17]; a18 = a[18]; a19 = a[19];
        a20 = a[20]; a21 = a[21]; a22 = a[22]; a23 = a[23];
        a24 = a[24]; a25 = a[25]; a26 = a[26]; a27 = a[27];
        a28 = a[28]; a29 = a[29]; a30 = a[30]; a31 = a[31];
        a32 = a[32]; a33 = a[33]; a34 = a[34]; a35 = a[35];
        a36 = a[36]; a37 = a[37]; a38 = a[38]; a39 = a[39];
        a40 = a[40]; a41 = a[41]; a42 = a[42]; a43 = a[43];
        a44 = a[44]; a45 = a[45]; a46 = a[46]; a47 = a[47];
        a48 = a[48]; a49 = a[49]; a50 = a[50]; a51 = a[51];
        a52 = a[52]; a53 = a[53]; a54 = a[54]; a55 = a[55];
        a56 = a[56]; a57 = a[57]; a58 = a[58]; a59 = a[59];
        a60 = a[60]; a61 = a[61]; a62 = a[62]; a63 = a[63];
    }

    inline bool compute(const uint8_t *b8) const {
        const uint64_t *b = (uint64_t *)b8;
        const uint64_t& r0 = is_super ? b[0] : a0;
        const uint64_t& r1 = is_super ? b[1] : a1;
        const uint64_t& r2 = is_super ? b[2] : a2;
        const uint64_t& r3 = is_super ? b[3] : a3;
        const uint64_t& r4 = is_super ? b[4] : a4;
        const uint64_t& r5 = is_super ? b[5] : a5;
        const uint64_t& r6 = is_super ? b[6] : a6;
        const uint64_t& r7 = is_super ? b[7] : a7;
        const uint64_t& r8 = is_super ? b[8] : a8;
        const uint64_t& r9 = is_super ? b[9] : a9;
        const uint64_t& r10 = is_super ? b[10] : a10;
        const uint64_t& r11 = is_super ? b[11] : a11;
        const uint64_t& r12 = is_super ? b[12] : a12;
        const uint64_t& r13 = is_super ? b[13] : a13;
        const uint64_t& r14 = is_super ? b[14] : a14;
        const uint64_t& r15 = is_super ? b[15] : a15;
        const uint64_t& r16 = is_super ? b[16] : a16;
        const uint64_t& r17 = is_super ? b[17] : a17;
        const uint64_t& r18 = is_super ? b[18] : a18;
        const uint64_t& r19 = is_super ? b[19] : a19;
        const uint64_t& r20 = is_super ? b[20] : a20;
        const uint64_t& r21 = is_super ? b[21] : a21;
        const uint64_t& r22 = is_super ? b[22] : a22;
        const uint64_t& r23 = is_super ? b[23] : a23;
        const uint64_t& r24 = is_super ? b[24] : a24;
        const uint64_t& r25 = is_super ? b[25] : a25;
        const uint64_t& r26 = is_super ? b[26] : a26;
        const uint64_t& r27 = is_super ? b[27] : a27;
        const uint64_t& r28 = is_super ? b[28] : a28;
        const uint64_t& r29 = is_super ? b[29] : a29;
        const uint64_t& r30 = is_super ? b[30] : a30;
        const uint64_t& r31 = is_super ? b[31] : a31;
        const uint64_t& r32 = is_super ? b[32] : a32;
        const uint64_t& r33 = is_super ? b[33] : a33;
        const uint64_t& r34 = is_super ? b[34] : a34;
        const uint64_t& r35 = is_super ? b[35] : a35;
        const uint64_t& r36 = is_super ? b[36] : a36;
        const uint64_t& r37 = is_super ? b[37] : a37;
        const uint64_t& r38 = is_super ? b[38] : a38;
        const uint64_t& r39 = is_super ? b[39] : a39;
        const uint64_t& r40 = is_super ? b[40] : a40;
        const uint64_t& r41 = is_super ? b[41] : a41;
        const uint64_t& r42 = is_super ? b[42] : a42;
        const uint64_t& r43 = is_super ? b[43] : a43;
        const uint64_t& r44 = is_super ? b[44] : a44;
        const uint64_t& r45 = is_super ? b[45] : a45;
        const uint64_t& r46 = is_super ? b[46] : a46;
        const uint64_t& r47 = is_super ? b[47] : a47;
        const uint64_t& r48 = is_super ? b[48] : a48;
        const uint64_t& r49 = is_super ? b[49] : a49;
        const uint64_t& r50 = is_super ? b[50] : a50;
        const uint64_t& r51 = is_super ? b[51] : a51;
        const uint64_t& r52 = is_super ? b[52] : a52;
        const uint64_t& r53 = is_super ? b[53] : a53;
        const uint64_t& r54 = is_super ? b[54] : a54;
        const uint64_t& r55 = is_super ? b[55] : a55;
        const uint64_t& r56 = is_super ? b[56] : a56;
        const uint64_t& r57 = is_super ? b[57] : a57;
        const uint64_t& r58 = is_super ? b[58] : a58;
        const uint64_t& r59 = is_super ? b[59] : a59;
        const uint64_t& r60 = is_super ? b[60] : a60;
        const uint64_t& r61 = is_super ? b[61] : a61;
        const uint64_t& r62 = is_super ? b[62] : a62;
        const uint64_t& r63 = is_super ? b[63] : a63;
        return (a0 & b[0]) == r0 && (a1 & b[1]) == r1 &&
                (a2 & b[2]) == r2 && (a3 & b[3]) == r3 &&
                (a4 & b[4]) == r4 && (a5 & b[5]) == r5 &&
                (a6 & b[6]) == r6 && (a7 & b[7]) == r7 &&
                (a8 & b[8]) == r8 && (a9 & b[9]) == r9 &&
                (a10 & b[10]) == r10 && (a11 & b[11]) == r11 &&
                (a12 & b[12]) == r12 && (a13 & b[13]) == r13 &&
                (a14 & b[14]) == r14 && (a15 & b[15]) == r15 &&
                (a16 & b[16]) == r16 && (a17 & b[17]) == r17 &&
                (a18 & b[18]) == r18 && (a19 & b[19]) == r19 &&
                (a20 & b[20]) == r20 && (a21 & b[21]) == r21 &&
                (a22 & b[22]) == r22 && (a23 & b[23]) == r23 &&
                (a24 & b[24]) == r24 && (a25 & b[25]) == r25 &&
                (a26 & b[26]) == r26 && (a27 & b[27]) == r27 &&
                (a28 & b[28]) == r28 && (a29 & b[29]) == r29 &&
                (a30 & b[30]) == r30 && (a31 & b[31]) == r31 &&
                (a32 & b[32]) == r32 && (a33 & b[33]) == r33 &&
                (a34 & b[34]) == r34 && (a35 & b[35]) == r35 &&
                (a36 & b[36]) == r36 && (a37 & b[37]) == r37 &&
                (a38 & b[38]) == r38 && (a39 & b[39]) == r39 &&
                (a40 & b[40]) == r40 && (a41 & b[41]) == r41 &&
                (a42 & b[42]) == r42 && (a43 & b[43]) == r43 &&
                (a44 & b[44]) == r44 && (a45 & b[45]) == r45 &&
                (a46 & b[46]) == r46 && (a47 & b[47]) == r47 &&
                (a48 & b[48]) == r48 && (a49 & b[49]) == r49 &&
                (a50 & b[50]) == r50 && (a51 & b[51]) == r51 &&
                (a52 & b[52]) == r52 && (a53 & b[53]) == r53 &&
                (a54 & b[54]) == r54 && (a55 & b[55]) == r55 &&
                (a56 & b[56]) == r56 && (a57 & b[57]) == r57 &&
                (a58 & b[58]) == r58 && (a59 & b[59]) == r59 &&
                (a60 & b[60]) == r60 && (a61 & b[61]) == r61 &&
                (a62 & b[62]) == r62 && (a63 & b[63]) == r63;
        }
};

template<bool is_super>
struct StructureComputerDefault {
    const uint8_t *a;
    int n;

    StructureComputerDefault() {}

    StructureComputerDefault(const uint8_t *a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t *a8, int code_size) {
        a = a8;
        n = code_size;
    }

    bool compute(const uint8_t *b8) const {
        return (is_super ? is_subset(b8, a, n)
                            : is_subset(a, b8, n));
    }
};

}
}
}

#endif