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
    // Allocate shared memory for tile caching
    extern __shared__ _FTYPE_ shared_mem_pool[];
    _FTYPE_ *tile_A = (_FTYPE_ (*))(&shared_mem_pool);
    _FTYPE_ *tile_B = (_FTYPE_ (*))(&shared_mem_pool[TILEDIM_M * TILEDIM_K]);
    
    // Thread and block indices
    int thread_x = threadIdx.x;  // 0-15
    int thread_y = threadIdx.y;  // 0-15
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    
    // Accumulator registers for 2x2 output block per thread
    _FTYPE_ accum_row0_col0 = 0;
    _FTYPE_ accum_row0_col1 = 0;
    _FTYPE_ accum_row1_col0 = 0;
    _FTYPE_ accum_row1_col1 = 0;
    
    // Calculate number of tiles to process along K dimension
    int num_tiles = (N + TILEDIM_K - 1) / TILEDIM_K;
    
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Load A tile from global memory (32x32 tile, 4 elements per thread)
        int global_A_col = tile_idx * TILEDIM_K + thread_x + (thread_y % 2) * 16;
        int global_A_row = block_y * TILEDIM_M + thread_y / 2;
        int shared_A_col = thread_x + (thread_y % 2) * 16;
        int shared_A_row = thread_y / 2;
       
        tile_A[shared_A_row * TILEDIM_K + shared_A_col] = A[global_A_row * N + global_A_col];
        tile_A[(shared_A_row + 8) * TILEDIM_K + shared_A_col] = A[(global_A_row + 8) * N + global_A_col];
        tile_A[(shared_A_row + 16) * TILEDIM_K + shared_A_col] = A[(global_A_row + 16) * N + global_A_col];
        tile_A[(shared_A_row + 24) * TILEDIM_K + shared_A_col] = A[(global_A_row + 24) * N + global_A_col];
        
        // Load B tile from global memory (32x32 tile, 4 elements per thread)
        int global_B_col = block_x * TILEDIM_N + thread_x + (thread_y % 2) * 16;
        int global_B_row = tile_idx * TILEDIM_K + thread_y / 2;
        int shared_B_col = thread_x + (thread_y % 2) * 16;
        int shared_B_row = thread_y / 2;
       
        tile_B[shared_B_row * TILEDIM_N + shared_B_col] = B[global_B_row * N + global_B_col];
        tile_B[(shared_B_row + 8) * TILEDIM_N + shared_B_col] = B[(global_B_row + 8) * N + global_B_col];
        tile_B[(shared_B_row + 16) * TILEDIM_N + shared_B_col] = B[(global_B_row + 16) * N + global_B_col];
        tile_B[(shared_B_row + 24) * TILEDIM_N + shared_B_col] = B[(global_B_row + 24) * N + global_B_col];
        
        __syncthreads();
        
        // Compute partial dot products for 2x2 output block
        #pragma unroll
        for(int k = 0; k < TILEDIM_K; k++) {
            _FTYPE_ a_val_row0 = tile_A[thread_y * TILEDIM_K + k];
            _FTYPE_ a_val_row1 = tile_A[(thread_y + 16) * TILEDIM_K + k];
            _FTYPE_ b_val_col0 = tile_B[k * TILEDIM_N + (2 * thread_x)];
            _FTYPE_ b_val_col1 = tile_B[k * TILEDIM_N + (2 * thread_x + 1)];
           
            accum_row0_col0 += a_val_row0 * b_val_col0;
            accum_row0_col1 += a_val_row0 * b_val_col1;
            accum_row1_col0 += a_val_row1 * b_val_col0;
            accum_row1_col1 += a_val_row1 * b_val_col1;
        }
        
        __syncthreads();
    }
    
    // Write 2x2 output block to global memory
    int output_row0 = block_y * TILEDIM_M + thread_y;
    int output_row1 = block_y * TILEDIM_M + thread_y + 16;
    int output_col0 = block_x * TILEDIM_N + 2 * thread_x;
    int output_col1 = block_x * TILEDIM_N + 2 * thread_x + 1;
   
    C[output_row0 * N + output_col0] = accum_row0_col0;
    C[output_row0 * N + output_col1] = accum_row0_col1;
    C[output_row1 * N + output_col0] = accum_row1_col0;
    C[output_row1 * N + output_col1] = accum_row1_col1;
}
#endif
