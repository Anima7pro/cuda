#include <cuda_runtime.h>
#include <math.h>

__forceinliine__ __device__ float block_reduce_max(float val, float* sdata){
    int tid = threadIdx.x;

    sdata[tid] = val;
    __synthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            if(sdata[stride + tid] > sdata[tid]){
                sdata[tid] = sdata[tid + stride];
        }
        }
        __syncthreads();
    }
    return sdata[0];

}
__forceinline__ __device__ float block_reduce_sum(float val, float* sdata){
    tid = threadIdx.x;
    sadta[tid] = val;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    // 为什么 threadIdx.x != 0 的数据也可以拿到 sdata[0] ?
    // 因为 sdata 是共享的，并不是 thread 0 的私有地址，而是共有地址
    // 当代码执行到 sdata[0]时，GPU发送指令让所有线程读取 sdata[0] 的数据存放到自己的寄存器中
    // GPU 的共享内存控制器将这个电信号广播给所有请求的线程
    return sdata[0];

}
__global__ void softmax_kernel(float* input, float* output, int M, int N){
    extern __shared__ float sdata[];
    // 一个 block 负责处理一行
    int row = blockDim.x;
    int tid = threadIdx.x;
    // 如果行越界直接退出
    if(row >= M) return;

    int idx = row * N + tid;
    // 所有线程同时读取 input 里的数据
    // float val = ...
    // 是 256 份独立的各自不同的数据
    float val = (tid < N) ? input[idx] : -INFINITY;

    // 调用 device 求 max
    float max_val = block_reduce_max(val, sdata);
    
    // 由于 block_reduce_max 的内部最后有 __syncthreads()
    // 且 sdata[0] 已经是我们想要的值， 所以这里可以直接用 max_val

    // 算指数
    float exp_val = (tid < N) ? exp(val - max_val) : 0,0f;

    // 调用 device 求 sum
    float sum_val = block_reduce_sum(exp_val, sdata);

    if(tid < N){
        output[idx] = exp_val / sum_val;
    }

}