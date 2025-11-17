#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
#include <stdio.h>

using namespace std;

#ifdef NAIVE

__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    int I = blockIdx.y * blockDim.y + threadIdx.y;
    int J = blockIdx.x * blockDim.x + threadIdx.x;

    if (I < N && J < N) {
        _FTYPE_ sum = 0;

        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            sum += a * b;
        }

        C[I * N + J] = sum;
    }
}

#else

__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    extern __shared__ _FTYPE_ shared_mem_pool[];

    _FTYPE_ *As = (_FTYPE_*) shared_mem_pool;
    _FTYPE_ *Bs = (_FTYPE_*) &shared_mem_pool[TILEDIM_M * TILEDIM_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    _FTYPE_ C00 = 0, C01 = 0;
    _FTYPE_ C10 = 0, C11 = 0;

    int num_tiles = (N + TILEDIM_K - 1) / TILEDIM_K;
    int tid = ty * 32 + tx;

    for (int tile = 0; tile < num_tiles; tile++) {

        for (int i = 0; i < 4; i++) {
            int elem = tid * 4 + i;

            int a_row_s = elem / TILEDIM_K;
            int a_col_s = elem % TILEDIM_K;

            int a_row = by * TILEDIM_M + a_row_s;
            int a_col = tile * TILEDIM_K + a_col_s;

            if (a_row < N && a_col < N) {
                As[a_row_s * TILEDIM_K + a_col_s] = A[a_row * N + a_col];
            } else {
                As[a_row_s * TILEDIM_K + a_col_s] = 0;
            }
        }

        for (int i = 0; i < 4; i++) {
            int elem = tid * 4 + i;

            int b_row_s = elem / TILEDIM_N;
            int b_col_s = elem % TILEDIM_N;

            int b_row = tile * TILEDIM_K + b_row_s;
            int b_col = bx * TILEDIM_N + b_col_s;

            if (b_row < N && b_col < N) {
                Bs[b_row_s * TILEDIM_N + b_col_s] = B[b_row * N + b_col];
            } else {
                Bs[b_row_s * TILEDIM_N + b_col_s] = 0;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILEDIM_K; k++) {
            _FTYPE_ a0 = As[ty * TILEDIM_K + k];
            _FTYPE_ a1 = As[(ty + 32) * TILEDIM_K + k];

            _FTYPE_ b0 = Bs[k * TILEDIM_N + (2 * tx)];
            _FTYPE_ b1 = Bs[k * TILEDIM_N + (2 * tx + 1)];

            C00 += a0 * b0;
            C01 += a0 * b1;

            C10 += a1 * b0;
            C11 += a1 * b1;
        }

        __syncthreads();
    }

    int row0 = by * TILEDIM_M + ty;
    int row1 = row0 + 32;

    int col0 = bx * TILEDIM_N + 2 * tx;
    int col1 = col0 + 1;

    if (row0 < N && col0 < N) C[row0 * N + col0] = C00;
    if (row0 < N && col1 < N) C[row0 * N + col1] = C01;

    if (row1 < N && col0 < N) C[row1 * N + col0] = C10;
    if (row1 < N && col1 < N) C[row1 * N + col1] = C11;
}

#endif
