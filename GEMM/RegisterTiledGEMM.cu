#define BK 8 // K 维度的分块大小
#define BM 128
#define BN 128
#define TM 8 // 每个线程负责 M 维度的 4 个点
#define TN 8
// 线程块大小为 (BM / TM) x (BN / TN) = 16 x 16 = 256 线程
// 每个线程处理 TM x TN = 8 x 8 = 64 个 C 矩阵元素
#include<cuda_runtime.h>

__global__ void sgemm_register_tiled(const float* A, const float* B, float* C, int M, int N, int K){

    // 1. 计算当前线程处理的 C 矩阵元素的起始位置
    int c_row = blockIdx.y * BM + threadIdx.y * TM;
    int c_col = blockIdx.x * BN + threadIdx.x * TN;

    // 2. 在寄存器中为 C 子矩阵分配空间
    // 这里的 64 个 float 元素存储在寄存器中
    float thread_results[TM * TN] = {0.0f};

    // 3. 申请 Shared Memory 用于 A 和 B 子矩阵
    __shared__ float As[BK][BM]; // 转置存储，为了消除 Bank Conflict
    __shared__ float Bs[BK][BN];

}