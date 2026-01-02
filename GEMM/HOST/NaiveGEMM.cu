void run_sgemm_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0); int K = A.size(1); int N = B.size(1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemm_naive_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}