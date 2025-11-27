#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "../tensor.cuh"
#include "../ops.cuh"


#define BLOCK_MAX_SIZE 256
#define BLOCK_MAX_COUNT 1024


//Sigmoid
__global__ void sigmoid(float* u, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = 1 / (1 + expf(-u[i])); //sigmoid
}

void sigmoidCuda(Tensor& u, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    sigmoid << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}

__global__ void sigmoidChainRule(float* u, float* dc_da, float* w, int n) {
    // assuming u is already sigmoided.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = dc_da[i] * u[i] * (1 - u[i]); //sigmoid derivative in chain rule.
}

void sigmoidChainRuleCuda(Tensor& u, Tensor& dc_da, Tensor& w) {
    //assuming u is activations
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    dc_da.toDevice();
    sigmoidChainRule << <blocks, threadsPerBlock >> > (u.getData(), dc_da.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}


//Relu
__global__ void relu(float* u, float* w, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = 0.0f < u[i] ? u[i] : alpha * u[i]; //Leaky Relu
}

void reluCuda(Tensor& u, Tensor& w, float alpha) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    relu << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n, alpha);
    cudaDeviceSynchronize();
}

__global__ void reluChainRule(float* u, float* dc_da, float* w, int n, float alpha) {
    //assuming u is activations.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = (0.0f < u[i]) ? dc_da[i] : dc_da[i] * alpha; //relu derivative in chain rule
}

void reluChainRuleCuda(Tensor& u, Tensor& dc_da, Tensor& w, float alpha) {
    //assuming u is activations
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    dc_da.toDevice();
    reluChainRule << <blocks, threadsPerBlock >> > (u.getData(), dc_da.getData(), w.getData(), n, alpha);
    cudaDeviceSynchronize();
}


//Softmax
__global__ void expAndSum(float* u, float* w, int n, float max, float* sum) {
    extern __shared__ float blockBuffer[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    int blockSize = blockDim.x;

    float localSum = 0.0f;
    for (; i < n; i += stride) {
        w[i] = expf(u[i] - max);
        localSum += w[i];
    }

    blockBuffer[tid] = localSum;
    __syncthreads();

    while (1 < blockSize) {
        if ((blockSize % 2) && (tid == 0))
            blockBuffer[0] += blockBuffer[blockSize - 1];
        blockSize /= 2;
        if (tid < blockSize)
            blockBuffer[tid] += blockBuffer[tid + blockSize];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(sum, blockBuffer[0]);
}

void softmaxCuda(Tensor& u, Tensor& w) {
    float sum;
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0.0f, sizeof(float));

    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();

    float max = maxCuda(u);
    expAndSum << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n, max, d_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    divideByScaler <<<blocks, threadsPerBlock >> > (w.getData(), w.getData(), n, sum);
    cudaDeviceSynchronize();
}


__global__ void softmaxChainRule(float* u, float* dc_da, float dot, float* w, int n) {
    // assuming u is already softmaxed - the activations.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride)
        w[i] = u[i] * (dc_da[i] - dot);
}

void softmaxChainRuleCuda(Tensor& u, Tensor& dc_da, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    dc_da.toDevice();
    w.toDevice();
    float dot = dotCuda(u, dc_da);

    softmaxChainRule << <blocks, threadsPerBlock >> > (u.getData(), dc_da.getData(), dot, w.getData(), n);
    cudaDeviceSynchronize();
}
