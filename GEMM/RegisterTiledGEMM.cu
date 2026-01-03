#define BK 8 // K 维度的分块大小
#define BM 64
#define BN 64
#define TM 4 // 每个线程负责 M 维度的 4 个点
#define TN 4
// 线程块大小为 (BM / TM) x (BN / TN) = 16 x 16 = 256 线程
// 每个线程处理 TM x TN = 4 x 4 = 16 个 C 矩阵元素
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

    // 缓存寄存器
    float reg_M[TM];
    float reg_N[TN];

    // 线程 ID (0 - 255) 用于搬运数据的调度
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // --------------------------------------
    // 遍历 K 维度
    // --------------------------------------
    for (int k_idx = 0; k_dix < K; k_dix += BK){

        // --- float4 向量化加载 A (Global -> Shared) ---
        // 目标: 搬运 A 的 [BM x BK] = [64 x 8] = 512 floats = 128 float4
        // 我们有 256 个线程, 只需要前 128 个线程搬运
        if (tid < BM * BK / 4){
            // 算出这个线程负责搬 A 的哪个 float4
            // A 的 Tile 有 BM x BK
            // 每一行有 BK / 4 个 float4 (这里当 BK = 8 时, 每一行有 2 个 float4)

            // 两个线程负责搬完一行, 所以行号每 2 个线程变 1 次
            int row_a = tid / 2;

            // 偶数线程搬左边, 奇数线程搬右边
            int vec_idx = tid % 2; // 0, 1, 0, 1...
            int col_a = vec_idx * 4; // 0, 4, 0, 4...

            // =========================================
            // 构造 float4 指针并读取(一条指令读 128 bit)
            // 地址 = A 基地址 + 行偏移 + 列偏移
            // =========================================
            
            // blockIdx.y * BM + row_a 是在线程在行上的位置
            // k_idx + col_a 是在线程在列上的位置
            /*
                            矩阵 A (MxK)
                <----------- K (总宽度) ----------->
                
                0 1 2 ...                          (Row 0)
                ...
            (blockIdx.y * BM) ---------------------- (Block 起始行)
                |          |
                | row_a    | (局部偏移)
                v          |
            Target Row ----------------------------- (绝对行)
                            ^
                            |
                [ k_idx ] + [col_a]
                (Tile起点)   (Tile内偏移)
            
            */
            float4 v = reinterpret_cast<const float4*>(&A[(blockIDx.y * BM + row_a) * K + (k_idx + col_a)])[0];

            // 写入 Shared Memory (As 转置存[col][row])
            As[col_a + 0][row_a] = v.x;
            As[col_a + 1][row_a] = v.y;
            As[col_a + 2][row_a] = v.z;
            As[col_a + 3][row_a] = v.w;
        }

        // --- float4 向量化加载 A (Global -> Shared) ---
        // 目标: 搬运 B 的 [BK x BN] = [8 x 64] = 512 floats = 128 float4
        // B 的 Tile 是 8 行 64 列, 每一行有 16 个 float4
        if (tid < BK * BN / 4){
            int row_b = tid / 16;
            int vec_idx = tid % 16;
            int col_b = vec_idx * 4;

            // 构造 float4 读取
            float4 v = reinterpret_cast<const float4*>(&B[(k_idx + row_b) * N + (blockIdx.x * BN + col_b)])[0];

            // 写入 Shared Memory
            Bs[row_b][col_b + 0] = v.x; 
            Bs[row_b][col_b + 1] = v.y; 
            Bs[row_b][col_b + 2] = v.z; 
            Bs[row_b][col_b + 3] = v.w; 
        }

        __syncthreads(); // 等待所有线程搬运完成

        for (int k = 0; k < BK; ++k) {
            // 把 As 的一列 (TM=4) 加载到寄存器
            for (int i = 0; i < TM; ++i) {
                reg_M[i] = As[k][threadIdx.y * TM + i];
            }
            // 把 Bs 的一行 (TN=4) 加载到寄存器
            for (int i = 0; i < TN; ++i) {
                reg_N[i] = Bs[k][threadIdx.x * TN + i];
            }
            // 外积计算 (Outer Product) -> 16 次乘加 (FFMA)
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_results[i * TN + j] += reg_M[i] * reg_N[j];
                }
            }
        }
        __syncthreads();


    }
    // 写回 global memory
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            C[(c_row + i) * N + (c_col + j)] = thread_results[i * TN + j];
        }
    }


    

}