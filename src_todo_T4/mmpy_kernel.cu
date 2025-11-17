// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    extern __shared__ _FTYPE_ shared_mem_pool[];
    _FTYPE_ *As = (_FTYPE_ *)(&shared_mem_pool);
    _FTYPE_ *Bs = (_FTYPE_ *)(&shared_mem_pool[TILEDIM_M * TILEDIM_K]);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Each thread computes 8x8 output elements (64 elements per thread)
    _FTYPE_ C_local[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            C_local[i][j] = 0;
        }
    }

    int num_tiles = (N + TILEDIM_K - 1) / TILEDIM_K;

    // Thread linearization for loading
    int tid = ty * 8 + tx;

    for (int tile = 0; tile < num_tiles; tile++) {

        // Load matrix A tile (64x64 elements, 64 loads per thread)
        for (int i = 0; i < 64; i++) {
            int elem_id = tid * 64 + i;
            int a_row_shared = elem_id / TILEDIM_K;
            int a_col_shared = elem_id % TILEDIM_K;

            int a_row = by * TILEDIM_M + a_row_shared;
            int a_col = tile * TILEDIM_K + a_col_shared;

            if (a_row < N && a_col < N)
                As[a_row_shared * TILEDIM_K + a_col_shared] = A[a_row * N + a_col];
            else
                As[a_row_shared * TILEDIM_K + a_col_shared] = 0.0;
        }

        // Load matrix B tile (64x64 elements, 64 loads per thread)
        for (int i = 0; i < 64; i++) {
            int elem_id = tid * 64 + i;
            int b_row_shared = elem_id / TILEDIM_N;
            int b_col_shared = elem_id % TILEDIM_N;

            int b_row = tile * TILEDIM_K + b_row_shared;
            int b_col = bx * TILEDIM_N + b_col_shared;

            if (b_row < N && b_col < N)
                Bs[b_row_shared * TILEDIM_N + b_col_shared] = B[b_row * N + b_col];
            else
                Bs[b_row_shared * TILEDIM_N + b_col_shared] = 0.0;
        }

        __syncthreads();

        // Compute: each thread computes 8x8 output elements
        #pragma unroll
        for (int k = 0; k < TILEDIM_K; k++) {
            _FTYPE_ a[8];
            _FTYPE_ b[8];

            // Load 8 elements from A
            for (int i = 0; i < 8; i++)
                a[i] = As[(ty * 8 + i) * TILEDIM_K + k];

            // Load 8 elements from B
            for (int j = 0; j < 8; j++)
                b[j] = Bs[k * TILEDIM_N + (tx * 8 + j)];

            // Compute 8x8 block
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    C_local[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int c_row = by * TILEDIM_M + ty * 8 + i;
            int c_col = bx * TILEDIM_N + tx * 8 + j;

            if (c_row < N && c_col < N)
                C[c_row * N + c_col] = C_local[i][j];
        }
    }
}

#endif
