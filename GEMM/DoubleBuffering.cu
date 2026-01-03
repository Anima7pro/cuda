#include <torch/extension.h>
#include <cuda_runtime.h>

// ==========================================
// V5: Double Buffering (Global -> Shared 预取)
// ==========================================

#define BK 8    // K 维度步长
#define BM 64   // M 维度分块
#define BN 64   // N 维度分块
#define TM 4    // 线程 M 维工作量
#define TN 4    // 线程 N 维工作量

__global__ void sgemm_v5_double_buffering(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1. 坐标计算
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int c_row = blockIdx.y * BM + threadIdx.y * TM;
    int c_col = blockIdx.x * BN + threadIdx.x * TN;

    // 2. 双缓冲 Shared Memory
    // 注意：第一维变成了 [2]，代表两个缓冲区 (Buffer 0 和 Buffer 1)
    __shared__ float As[2][BK][BM]; // 转置存储
    __shared__ float Bs[2][BK][BN]; 

    // 寄存器 (计算用)
    float thread_results[TM * TN] = {0.0f};
    float reg_M[TM];
    float reg_N[TN];

    // 寄存器 (预取搬运用)
    // 我们需要把 Global 的数据先搬到这里，等计算完了再存入 Shared
    float4 load_a_reg;
    float4 load_b_reg;

    // 搬运工的坐标计算 (同 V4)
    // A: 64x8, float4 => 64x2 = 128 threads
    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 4;
    
    // B: 8x64, float4 => 8x16 = 128 threads (行优先)
    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 4;

    // ============================================================
    // PROLOGUE (序幕): 预加载第一个 Tile (k=0) 到 Shared Memory
    // ============================================================
    
    // 1. Load Global (k=0) -> Register
    if (tid < 128) {
        load_a_reg = reinterpret_cast<const float4*>(&A[(blockIdx.y * BM + load_a_row) * K + (0 + load_a_col)])[0];
        load_b_reg = reinterpret_cast<const float4*>(&B[(0 + load_b_row) * N + (blockIdx.x * BN + load_b_col)])[0];
    }
    
    // 2. Register -> Shared Memory (Buffer 0)
    if (tid < 128) {
        As[0][load_a_col][load_a_row]     = load_a_reg.x;
        As[0][load_a_col+1][load_a_row]   = load_a_reg.y;
        As[0][load_a_col+2][load_a_row]   = load_a_reg.z;
        As[0][load_a_col+3][load_a_row]   = load_a_reg.w;

        Bs[0][load_b_row][load_b_col]     = load_b_reg.x;
        Bs[0][load_b_row][load_b_col+1]   = load_b_reg.y;
        Bs[0][load_b_row][load_b_col+2]   = load_b_reg.z;
        Bs[0][load_b_row][load_b_col+3]   = load_b_reg.w;
    }
    __syncthreads();

    // ============================================================
    // MAIN LOOP (主体): 流水线模式
    // ============================================================
    
    // 指针切换标记
    int write_stage_idx = 1; // 下一次要把数据写入 Shared[1]
    int load_stage_idx = 0;  // 当前从 Shared[0] 读取数据进行计算

    // 我们每次循环处理 2 个事情：
    // 1. 算第 k 个 Tile (数据已经在 Shared[load_stage] 里了)
    // 2. 搬第 k+BK 个 Tile (搬到 Reg，最后存入 Shared[write_stage])
    
    for (int k = 0; k < K - BK; k += BK) {
        
        // --- 步骤 A: 发出预取指令 (Prefetch Next Tile) ---
        // 这里的关键是：Global Load 是异步的，发出指令后 GPU 不会卡住
        // 它会继续往下执行计算指令，直到真的需要用 load_a_reg 的时候
        if (tid < 128) {
            // 加载 k + BK (下一块)
            load_a_reg = reinterpret_cast<const float4*>(&A[(blockIdx.y * BM + load_a_row) * K + (k + BK + load_a_col)])[0];
            load_b_reg = reinterpret_cast<const float4*>(&B[(k + BK + load_b_row) * N + (blockIdx.x * BN + load_b_col)])[0];
        }

        // --- 步骤 B: 疯狂计算当前 Tile (Compute Current Tile) ---
        // 利用这段时间掩盖上面 Global Load 的延迟
        // 读的是 Shared[load_stage_idx]
        #pragma unroll
        for (int j = 0; j < BK; ++j) {
            // Load Shared -> Reg
            for (int i = 0; i < TM; ++i) reg_M[i] = As[load_stage_idx][j][threadIdx.y * TM + i];
            for (int i = 0; i < TN; ++i) reg_N[i] = Bs[load_stage_idx][j][threadIdx.x * TN + i];
            // Outer Product
            for (int i = 0; i < TM; ++i) {
                for (int l = 0; l < TN; ++l) {
                    thread_results[i * TN + l] += reg_M[i] * reg_N[l];
                }
            }
        }

        // --- 步骤 C: 将预取的数据写入 Shared Memory (Store to Next Buffer) ---
        // 此时，步骤 A 的数据应该已经（或者快要）到了
        // 我们把它写入 write_stage_idx
        if (tid < 128) {
            As[write_stage_idx][load_a_col][load_a_row]     = load_a_reg.x;
            As[write_stage_idx][load_a_col+1][load_a_row]   = load_a_reg.y;
            As[write_stage_idx][load_a_col+2][load_a_row]   = load_a_reg.z;
            As[write_stage_idx][load_a_col+3][load_a_row]   = load_a_reg.w;

            Bs[write_stage_idx][load_b_row][load_b_col]     = load_b_reg.x;
            Bs[write_stage_idx][load_b_row][load_b_col+1]   = load_b_reg.y;
            Bs[write_stage_idx][load_b_row][load_b_col+2]   = load_b_reg.z;
            Bs[write_stage_idx][load_b_row][load_b_col+3]   = load_b_reg.w;
        }

        // 等待大家写完
        __syncthreads();

        // --- 步骤 D: 乒乓切换 (Ping-Pong Switch) ---
        // 下一轮计算读取刚刚写入的 buffer，写入旧的 buffer
        write_stage_idx ^= 1; // 0变1, 1变0
        load_stage_idx ^= 1;
    }

    // ============================================================
    // EPILOGUE (尾声): 计算最后一个 Tile
    // ============================================================
    // 循环结束时，最后一个 Tile 的数据已经加载到 Shared Memory 里了
    // 但是还没有计算，所以要补上最后一次计算
    
    #pragma unroll
    for (int j = 0; j < BK; ++j) {
        for (int i = 0; i < TM; ++i) reg_M[i] = As[load_stage_idx][j][threadIdx.y * TM + i];
        for (int i = 0; i < TN; ++i) reg_N[i] = Bs[load_stage_idx][j][threadIdx.x * TN + i];
        for (int i = 0; i < TM; ++i) {
            for (int l = 0; l < TN; ++l) {
                thread_results[i * TN + l] += reg_M[i] * reg_N[l];
            }
        }
    }

    // 3. 写回 Global Memory (同 V4)
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            if ((c_row + i) < M && (c_col + j) < N)
                C[(c_row + i) * N + (c_col + j)] = thread_results[i * TN + j];
        }
    }
}


