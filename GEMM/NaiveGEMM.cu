#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16  // 每个 blcok 处理 16 x 16的子矩阵

// --- version 1: naive ---
__global__ void sgemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        float sum = 0.0f;
        for (int k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
            // 瓶颈：频繁访问 Global Memory
        }
        C[row * N + col] = sum;
    }

}