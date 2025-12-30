// GEMM 标准定义 C = \alpha (A x B) + \beta C
#include <cuda_runtime.h>
#define TILE_SIZE = 32


/*
每个 thread 负责计算 C 中的一个元素
每个 thread 都要从 global memory 中读取整整一行的 A 和 整整一列 B 
*/

__device__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f;
        for(int i = 0, i < K, i++){
            sum += A[row * N + i] * B[i * K + col]; 
        }
        C[row * N + col] = sum;
    }
}


// 共享内存分块 (shared memory tiling)
/*
把数据从慢速的 global memory 提取到快速的 shared memory 中，然后再反复读取 SRAM
1. Tiling      把大矩阵分割成小 Tile
2. 协作搬运     Block 内的线程合作，把 A 的一小块和 B 的一小块加载到 SRAM 中
3. 同步         __syncthreads() 确保每个线程都搬完了
4. 计算         线程从 shared memory 中读取数据进行点积
5. 循环         移动到下一个 Tile
*/

__device__ void gemm_shared_mem(float* A, float* B, float* C, int M, int N, int K){
    // 声明共享内存，在 GPU 内部开辟两块高速缓存区
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 当前线程负责计算 C 的(row, col)
    int row = bx * TILE_SIZE + threadIdx.x;
    int col = by * TILE_SIZE + threadIdx.y;

    float val = 0.0f;

    // 外层循环，按块(Tile)遍历 K 维度
    // 每次处理一个 TILE_SIZE 宽度
    for(int ph = 0; ph < K / TILE_SIZE; ph++){

        // --- 阶段 A : 协作搬运数据 ---

        // 每个线程搬运一个元素 A 和一个元素 B 到共享内存
        // A 的全局位置: 行 row, 列 ph * TILE_SIZE + tx
        // B 的全局位置: 行 ph * TILE_SIZE + ty, 列 col
        As[ty][tx] = A[row * K + (ph * TILE_SIZE + tx)]; 
        // global memory 不可以二维访问 因为传入的是 float* A, 不允许访问浮点数的索引
        Bs[ty][tx] = B[(ph * TILE_SIZE + ty) * N + col];

        // 同步 确保 block 内的所有线程全部把数据搬完了
        __syncthreads();

        // --- 阶段 B : 在共享内存上计算 ---
        for(int k = 0; k < TILE_SIZE; k++){
            val += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
        // 此时一个小块计算完成，循环到下一个小块

    }
    C[row * N + col] = val;

}


