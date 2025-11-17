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
// Tiled matrix multiply kernel (optimized version)
// Computes C = A * B for N x N matrices using shared memory and per-thread 4x4 output blocks.
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    // Shared memory layout:
    //   tileA: TILEDIM_M x TILEDIM_K
    //   tileB: TILEDIM_K x TILEDIM_N
    extern __shared__ _FTYPE_ sharedMem[];
    _FTYPE_ *tileA = sharedMem;
    _FTYPE_ *tileB = sharedMem + TILEDIM_M * TILEDIM_K;

    // Thread indices within the block
    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;

    // Block indices within the grid
    const int blockCol = blockIdx.x;
    const int blockRow = blockIdx.y;

    // Each thread accumulates a 4x4 sub-block of C
    _FTYPE_ C_0_0 = 0, C_0_1 = 0, C_0_2 = 0, C_0_3 = 0;
    _FTYPE_ C_1_0 = 0, C_1_1 = 0, C_1_2 = 0, C_1_3 = 0;
    _FTYPE_ C_2_0 = 0, C_2_1 = 0, C_2_2 = 0, C_2_3 = 0;
    _FTYPE_ C_3_0 = 0, C_3_1 = 0, C_3_2 = 0, C_3_3 = 0;

    // Number of tiles to iterate over in the K dimension
    const int numTilesK = (N + TILEDIM_K - 1) / TILEDIM_K;

    // Loop over tiles in the K dimension
    for (int tileIdx = 0; tileIdx < numTilesK; tileIdx++) {
        // Load tile of A from global memory into shared memory (tileA)
        // Each thread loads multiple rows of A, spaced 4 rows apart
        int globalColA      = tileIdx * TILEDIM_K + threadCol + (threadRow % 4) * 16;
        int globalRowABase  = blockRow * TILEDIM_M + threadRow / 4;
        int sharedColA      = threadCol + (threadRow % 4) * 16;
        int sharedRowABase  = threadRow / 4;

        // For A, load rows: base, base+4, base+8, ..., base+60 (if in bounds)
        if (globalRowABase < N && globalColA < N) {
            tileA[sharedRowABase * TILEDIM_K + sharedColA] =
                A[globalRowABase * N + globalColA];
        } else {
            tileA[sharedRowABase * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 4) < N && globalColA < N) {
            tileA[(sharedRowABase + 4) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 4) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 4) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 8) < N && globalColA < N) {
            tileA[(sharedRowABase + 8) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 8) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 8) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 12) < N && globalColA < N) {
            tileA[(sharedRowABase + 12) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 12) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 12) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 16) < N && globalColA < N) {
            tileA[(sharedRowABase + 16) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 16) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 16) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 20) < N && globalColA < N) {
            tileA[(sharedRowABase + 20) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 20) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 20) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 24) < N && globalColA < N) {
            tileA[(sharedRowABase + 24) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 24) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 24) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 28) < N && globalColA < N) {
            tileA[(sharedRowABase + 28) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 28) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 28) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 32) < N && globalColA < N) {
            tileA[(sharedRowABase + 32) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 32) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 32) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 36) < N && globalColA < N) {
            tileA[(sharedRowABase + 36) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 36) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 36) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 40) < N && globalColA < N) {
            tileA[(sharedRowABase + 40) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 40) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 40) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 44) < N && globalColA < N) {
            tileA[(sharedRowABase + 44) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 44) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 44) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 48) < N && globalColA < N) {
            tileA[(sharedRowABase + 48) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 48) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 48) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 52) < N && globalColA < N) {
            tileA[(sharedRowABase + 52) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 52) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 52) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 56) < N && globalColA < N) {
            tileA[(sharedRowABase + 56) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 56) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 56) * TILEDIM_K + sharedColA] = 0.0;
        }

        if ((globalRowABase + 60) < N && globalColA < N) {
            tileA[(sharedRowABase + 60) * TILEDIM_K + sharedColA] =
                A[(globalRowABase + 60) * N + globalColA];
        } else {
            tileA[(sharedRowABase + 60) * TILEDIM_K + sharedColA] = 0.0;
        }

        // Load tile of B from global memory into shared memory (tileB)
        // Each thread loads multiple rows of B, spaced 4 rows apart
        int globalColBBase = blockCol * TILEDIM_N + threadCol + (threadRow % 4) * 16;
        int globalRowBBase = tileIdx * TILEDIM_K + threadRow / 4;
        int sharedColB     = threadCol + (threadRow % 4) * 16;
        int sharedRowBBase = threadRow / 4;

        if (globalRowBBase < N && globalColBBase < N) {
            tileB[sharedRowBBase * TILEDIM_N + sharedColB] =
                B[globalRowBBase * N + globalColBBase];
        } else {
            tileB[sharedRowBBase * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 4) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 4) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 4) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 4) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 8) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 8) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 8) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 8) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 12) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 12) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 12) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 12) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 16) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 16) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 16) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 16) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 20) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 20) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 20) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 20) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 24) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 24) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 24) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 24) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 28) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 28) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 28) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 28) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 32) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 32) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 32) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 32) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 36) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 36) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 36) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 36) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 40) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 40) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 40) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 40) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 44) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 44) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 44) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 44) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 48) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 48) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 48) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 48) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 52) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 52) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 52) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 52) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 56) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 56) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 56) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 56) * TILEDIM_N + sharedColB] = 0.0;
        }

        if ((globalRowBBase + 60) < N && globalColBBase < N) {
            tileB[(sharedRowBBase + 60) * TILEDIM_N + sharedColB] =
                B[(globalRowBBase + 60) * N + globalColBBase];
        } else {
            tileB[(sharedRowBBase + 60) * TILEDIM_N + sharedColB] = 0.0;
        }

        // Ensure all threads see a fully loaded tileA and tileB
        __syncthreads();

        // --------------------------------------------------------------------
        // Compute partial results for this tile
        // Each thread uses 4 rows from tileA and 4 columns from tileB
        // --------------------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < TILEDIM_K; k++) {
            _FTYPE_ a0 = tileA[threadRow * TILEDIM_K + k];
            _FTYPE_ a1 = tileA[(threadRow + 16) * TILEDIM_K + k];
            _FTYPE_ a2 = tileA[(threadRow + 32) * TILEDIM_K + k];
            _FTYPE_ a3 = tileA[(threadRow + 48) * TILEDIM_K + k];

            _FTYPE_ b0 = tileB[k * TILEDIM_N + (4 * threadCol)];
            _FTYPE_ b1 = tileB[k * TILEDIM_N + (4 * threadCol + 1)];
            _FTYPE_ b2 = tileB[k * TILEDIM_N + (4 * threadCol + 2)];
            _FTYPE_ b3 = tileB[k * TILEDIM_N + (4 * threadCol + 3)];

            C_0_0 += a0 * b0;
            C_0_1 += a0 * b1;
            C_0_2 += a0 * b2;
            C_0_3 += a0 * b3;

            C_1_0 += a1 * b0;
            C_1_1 += a1 * b1;
            C_1_2 += a1 * b2;
            C_1_3 += a1 * b3;

            C_2_0 += a2 * b0;
            C_2_1 += a2 * b1;
            C_2_2 += a2 * b2;
            C_2_3 += a2 * b3;

            C_3_0 += a3 * b0;
            C_3_1 += a3 * b1;
            C_3_2 += a3 * b2;
            C_3_3 += a3 * b3;
        }

        // Wait for all threads before starting next tile load
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Write the final 4x4 sub-block of C computed by this thread back to C
    // ------------------------------------------------------------------------
    int globalRowC0 = blockRow * TILEDIM_M + threadRow;
    int globalRowC1 = blockRow * TILEDIM_M + threadRow + 16;
    int globalRowC2 = blockRow * TILEDIM_M + threadRow + 32;
    int globalRowC3 = blockRow * TILEDIM_M + threadRow + 48;

    int globalColC0 = blockCol * TILEDIM_N + 4 * threadCol;
    int globalColC1 = blockCol * TILEDIM_N + 4 * threadCol + 1;
    int globalColC2 = blockCol * TILEDIM_N + 4 * threadCol + 2;
    int globalColC3 = blockCol * TILEDIM_N + 4 * threadCol + 3;

    if (globalRowC0 < N && globalColC0 < N) C[globalRowC0 * N + globalColC0] = C_0_0;
    if (globalRowC0 < N && globalColC1 < N) C[globalRowC0 * N + globalColC1] = C_0_1;
    if (globalRowC0 < N && globalColC2 < N) C[globalRowC0 * N + globalColC2] = C_0_2;
    if (globalRowC0 < N && globalColC3 < N) C[globalRowC0 * N + globalColC3] = C_0_3;

    if (globalRowC1 < N && globalColC0 < N) C[globalRowC1 * N + globalColC0] = C_1_0;
    if (globalRowC1 < N && globalColC1 < N) C[globalRowC1 * N + globalColC1] = C_1_1;
    if (globalRowC1 < N && globalColC2 < N) C[globalRowC1 * N + globalColC2] = C_1_2;
    if (globalRowC1 < N && globalColC3 < N) C[globalRowC1 * N + globalColC3] = C_1_3;

    if (globalRowC2 < N && globalColC0 < N) C[globalRowC2 * N + globalColC0] = C_2_0;
    if (globalRowC2 < N && globalColC1 < N) C[globalRowC2 * N + globalColC1] = C_2_1;
    if (globalRowC2 < N && globalColC2 < N) C[globalRowC2 * N + globalColC2] = C_2_2;
    if (globalRowC2 < N && globalColC3 < N) C[globalRowC2 * N + globalColC3] = C_2_3;

    if (globalRowC3 < N && globalColC0 < N) C[globalRowC3 * N + globalColC0] = C_3_0;
    if (globalRowC3 < N && globalColC1 < N) C[globalRowC3 * N + globalColC1] = C_3_1;
    if (globalRowC3 < N && globalColC2 < N) C[globalRowC3 * N + globalColC2] = C_3_2;
    if (globalRowC3 < N && globalColC3 < N) C[globalRowC3 * N + globalColC3] = C_3_3;
}
#endif
