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
// Optimized matrix multiply using tiling + shared memory
__global__ void matMul(int N, _FTYPE_ *C, const _FTYPE_ *A, const _FTYPE_ *B) {

    int tx = threadIdx.x;                     // thread column inside block
    int ty = threadIdx.y;                     // thread row inside block

    int row = blockIdx.y * TILEDIM_M + ty;    // global row in output C
    int col = blockIdx.x * TILEDIM_N + tx;    // global column in output C

    _FTYPE_ value = 0;                        // accumulator for C[row,col]

    extern __shared__ _FTYPE_ ShmMem[];       // shared memory for tiles
    _FTYPE_* As = ShmMem;                     // tile of A
    _FTYPE_* Bs = ShmMem + (TILEDIM_M * TILEDIM_K);  // tile of B

    // Loop over tiles along the K dimension
    for (int kk = 0; kk < N; kk += TILEDIM_K) {

        int aRow = row;                        // row of A to load
        int aCol = kk + tx;                    // column of A to load
        int aIndex = ty * TILEDIM_K + tx;      // index inside As

        // Load one element of A into shared memory
        As[aIndex] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0;

        int bRow = kk + ty;                    // row of B to load
        int bCol = col;                        // column of B to load
        int bIndex = ty * TILEDIM_N + tx;      // index inside Bs

        // Load one element of B into shared memory
        Bs[bIndex] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0;

        __syncthreads();                       // wait for full tile to load

        // Multiply the tiles and accumulate partial result
        #pragma unroll
        for (int k = 0; k < TILEDIM_K; k++) {
            value += As[ty * TILEDIM_K + k] *
                     Bs[k  * TILEDIM_N + tx];
        }

        __syncthreads();                       // wait before loading next tile
    }

    // Write final output value
    if (row < N && col < N)
        C[row * N + col] = value;
}
#endif