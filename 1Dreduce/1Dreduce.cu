#include<cuda_runtime.h>

__global__ void reduce_sum(float* input, float* output, int N){
    extern __shared__ float sdata[]

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride>0; stride >>= 1){
        if(tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        output[blockIdx.x] = sdata[0];
    }
    
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}