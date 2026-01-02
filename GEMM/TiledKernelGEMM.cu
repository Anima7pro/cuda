__global__ void sgemm_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    // 1. 声明 Shared Memory (L1 Cache)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 局部累加寄存器
    float sum = 0.0f;

    // 2. 循环遍历每一个 Tile (步长为 BLOCK_SIZE)
    for (int idx = 0; idx < (K + BLOCK_SIZE -1) / BLOCK_SIZE; idx++){
        // [阶段 A: 协作搬运]
        // 每个 thread 负责搬运 A 和 B 中对应的那个点到 Shared Memory

        // 边界检查并加载 A
        int r = row;
        int c = idx * BLOCK_SIZE + threadIdx.x;
        if (r < M && c < K) As[threadIdx.y][threadIdx.x] = A[r * K + c];
        else                As[threadIdx.y][threadIdx.x] = 0.0f;

        // 边界检查并加载 B
        r = idx * BLOCK_SIZE + threadIdx.y;
        c = col;
        if (r < K && c < N) Bs[threadIdx.y][threadIdx.x] = B[r * N + c];
        else                Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // [阶段 B: 同步]
        // 必须等 Block 内的所有线程都搬完才可以计算
        __syncthreads();

        // [阶段 C: 计算]
        for (int k = 0; k < BLOCK_SIZE; k++){
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // [阶段 D: 同步]
        // 必须等 Block 内的所有线程计算完才可以进行下一轮循环
        __syncthreads();

    }

    // 写回结果
    if (row < M && col < N){
        C[row * N + col] = sum;
    }

}