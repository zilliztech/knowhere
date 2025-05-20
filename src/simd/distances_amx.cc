#if defined(__x86_64__)
 
#include "distances_amx.h"
 
#include <immintrin.h>
 
#include <cassert>
#include <cstdio>
#include <string>
 
#include "faiss/impl/platform_macros.h"
 
namespace faiss {
 
#if defined(USE_AMX)
float amx_inner_product_matrix_bf16( char **floatLibraryMatrix, char  *floatQueryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, float *results_ptr){
    int DIM=32;
    int blockDim = 96;
    int blockCount=((dims))/blockDim;
    size_t tailCount=dims%DIM;
    int tailBlock=dims%blockDim;
 
    thread_local char cfg[64]={0};
    thread_local bool init_mem=false;
 
    unsigned char ma1Bf16[1024] __attribute__((aligned(64)));
    unsigned char ma2Bf16[1024] __attribute__((aligned(64)));
    unsigned char ma3Bf16[1024] __attribute__((aligned(64)));
 
    float results[16*16] __attribute__((aligned(64)))={0};
 
    if(!init_mem){
        cfg[0]=1;
        cfg[16]=DIM*2;
        cfg[48] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
        cfg[48+1]   = DIM/2;   // row = K/4
 
        cfg[22]=DIM*2;
        cfg[51] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[24] = batchSizeB*2*2;   // col = N*4
        cfg[52]   = DIM/2;   // row = K/4
 
        cfg[26]= DIM*2;
        cfg[53] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[28] = batchSizeB*2*2;   // col = N*4
        cfg[54]   = DIM/2;   // row = K/4
 
        cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
        cfg[48+2] = 16;
        init_mem = true;
 
        _tile_loadconfig((void *)cfg);
    }
    //memset(maBf16,0,16*DIM*2);
 
    int i=0;
    for(int i=0;i<blockCount;i++){
 
      //int32_t stride=i*DIM;
      __m512i sa;
      size_t offset = i * blockDim *2;
     
      for(int j=0;j<batchSizeA;j++){  
        size_t destOffset1 = j * DIM * 2;
 
        _mm512_store_si512(ma1Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset));
        _mm512_store_si512(ma2Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset + 64));
        _mm512_store_si512(ma3Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset + 128));
      }
 
      _tile_loadd(1,floatQueryMatrix + offset , 4);
      _tile_loadd(4,floatQueryMatrix + offset + 64 , 4);
      _tile_loadd(6,floatQueryMatrix + offset + 128, 4);
      _tile_loadd(0,ma1Bf16, 64);
      _tile_loadd(3,ma2Bf16, 64);
      _tile_loadd(5,ma3Bf16, 64);
      _tile_dpbf16ps(2,3,4);
      _tile_dpbf16ps(2,0,1);
      _tile_dpbf16ps(2,5,6);
    //amx_int8_mul((u64*) cfg, maInt8,queryMatrix+stride,DIM,batchSizeB*4,(void*)results);
    }
    if(tailBlock >= DIM){
      for(int i=0;i<tailBlock/DIM;i++){
        __m512i sa;
        for(int j=0;j<batchSizeA;j++){  
          sa=_mm512_loadu_si512(floatLibraryMatrix[j]+blockCount*blockDim * 2 + i * DIM * 2 );
          _mm512_store_si512(ma1Bf16+j*DIM*2,sa);
        }
        _tile_loadd(0,ma1Bf16, 64);
        _tile_loadd(1,floatQueryMatrix + blockCount*blockDim*2 + i * DIM*2 , 4);
        _tile_dpbf16ps(2,0,1);
      }
    }
    _tile_stored(2, results, batchSizeB*2*2);
    _tile_zero(2);
   
    // if (tailCount != 0) {
    //     int32_t offset= dims/DIM*DIM;
    //     for (int k = 0; k < batchSizeA; k++) {
    //         for (int l = 0; l < batchSizeB; l++) {
    //             for (int m = 0; m < tailCount; m += 1) {
    //               //blockDim*blockCount+tailBlock/DIM*DIM+i
                 
    //               results[k * batchSizeB + l] += bf162float(*(uint16_t *)(floatLibraryMatrix[k]  + 2*(offset+m))) * bf162float(*(uint16_t *)(floatQueryMatrix + 2*(offset+m)));
    //                 // __m512 lib_vec = _mm512_loadu_ps((float *)(floatLibraryMatrix[k]  + 2*(DIM * blockCount + i)));
    //                 // __m512 query_vec = _mm512_loadu_ps((float *)(floatQueryMatrix + 2*(DIM * blockCount + i)));
    //                 // result_vec = _mm512_fmadd_ps(lib_vec, query_vec, result_vec);
    //             }
    //         }
    //     }
    // }
    memcpy(results_ptr, results, batchSizeA * batchSizeB * sizeof(float));
 
    return 0;
}
 
static float InnerProductDistanceBf16AVX512(const void* a, const void* b, const void *qty_ptr) {
  float result[16] = {0.0f}; // 用于存储中间结果
 
  uint16_t *x = (uint16_t *)a;
  uint16_t *y = (uint16_t *)b;
  __m512 vr_f32 = _mm512_setzero_ps(); // 初始化累积寄存器为0
 
  size_t dim = * (size_t*) qty_ptr ;
 
  size_t i = 0;
  // 每次处理32个元素（16个__bf16元素在__m512bh寄存器中存储为32个uint16_t）
  for (; i + 32 <= dim; i += 32) {
      // 加载32个uint16_t到__m512i类型的临时寄存器
      __m512i temp_x = _mm512_loadu_si512(x + i);
      __m512i temp_y = _mm512_loadu_si512(y + i);
 
      // 强制转换为__m512bh类型
      __m512bh v1_f16 = reinterpret_cast<__m512bh&>(temp_x);
      __m512bh v2_f16 = reinterpret_cast<__m512bh&>(temp_y);
 
      // 计算BF16的点积，并将结果累加到vr_f32
      vr_f32 = _mm512_dpbf16_ps(vr_f32, v1_f16, v2_f16);
  }
 
  // 将vr_f32寄存器的值存入result数组
  _mm512_storeu_ps(result, vr_f32);
 
  // 累加result数组的所有元素，获得最终的点积结果
  float dot_product = 0.0f;
  for (int j = 0; j < 16; j++) {
      dot_product += result[j];
  }
 
  // 处理剩余的元素（小于32的部分）
  for (; i < dim; i++) {
      float x_val = (float)(knowhere::bf16)(x[i]);
      float y_val = (float)(knowhere::bf16)(y[i]);
      dot_product += x_val * y_val;
  }
  //printf("%d %f ",dim,dot_product);
  return dot_product;
}
static float InnerProductBatchExtAMXBF16(void **pVect1v, void *pVect2v, void *qty_ptr, size_t nSize, size_t mSize, float * results_amx){
  unsigned int dims= *(unsigned int*)qty_ptr;
  char **floatLibraryMatrix = (char**) pVect1v;
  char *floatQueryMatrix = (char*) pVect2v;
 
  int batchSizeA = 16, batchSizeB = 16;
  int batchCountA = (nSize - 1) / batchSizeA + 1;
  int batchCountB = (mSize - 1) / batchSizeB + 1;
 
  int lastBatchSizeA = (nSize % batchSizeA == 0) ? batchSizeA : nSize % batchSizeA;
  int lastBatchSizeB = (mSize % batchSizeB == 0) ? batchSizeB : mSize % batchSizeB;
 
  int offsetA = batchSizeA * dims * 2;
  int offsetB = batchSizeB * dims * 2;
 
  float *results_ptr = results_amx;
 
  for (int i = 0; i < batchCountA; i++) {
      int currentBatchSizeA = (i == batchCountA - 1) ? lastBatchSizeA : batchSizeA;
      char **currentLibraryMatrixPtr = floatLibraryMatrix + i * 16;
 
      for (int j = 0; j < batchCountB; j++) {
          int currentBatchSizeB = (j == batchCountB - 1) ? lastBatchSizeB : batchSizeB;
          char *currentQueryMatrixPtr = floatQueryMatrix + j * offsetB;
 
          amx_inner_product_matrix_bf16(currentLibraryMatrixPtr, currentQueryMatrixPtr, dims, currentBatchSizeA, currentBatchSizeB, results_ptr);
 
          results_ptr += currentBatchSizeB * currentBatchSizeA;
      }
  }
 
  return 0;
}
float
bf16_vec_inner_product_amx_ref(void **p_bVect1v, void *p_qVect2v, void *dim_ptr, 
    size_t b_Size, size_t q_Size, float * results_amx) {
    size_t qty = *((size_t *) dim_ptr);
    size_t qty32 = qty >> 5 << 5;
 
    InnerProductBatchExtAMXBF16(p_bVect1v, p_qVect2v, &qty32,b_Size,q_Size,results_amx);
 
    size_t qty_left = qty - qty32;
 
    uint16_t *pVect2 = (uint16_t *) p_qVect2v + qty32;
    if(qty_left>0){
        for(size_t i = 0; i < b_Size; i++) {
            uint16_t *pVect1 = (uint16_t *) p_bVect1v[i] + qty32;
            results_amx[i] += InnerProductDistanceBf16AVX512(pVect1, pVect2, &qty_left);
        }
    }
 
    // float *results_avx512 = new float[b_Size*q_Size];
    // for(size_t i = 0; i < b_Size; i++) {
    //     results_avx512[i] = InnerProductDistanceBf16AVX512(p_bVect1v[i], p_qVect2v, &qty);
    // }
 
    // for(size_t i = 0; i < nSize; i++) {
    //     printf("amx:%f,avx512:%f\n",results_amx[i],results_avx512[i]);
    // }
    return 0;
}  
 
#endif
 
}
#endif