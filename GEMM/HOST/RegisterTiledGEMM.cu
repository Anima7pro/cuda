void run_sgemm_RegisterTiled(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0); int N = B.size(1); int K = A.size(1);
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_tiled<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
}