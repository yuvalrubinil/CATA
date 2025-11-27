#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "../tensor.cuh"
#include "../ops.cuh"

#define BLOCK_MAX_SIZE 256
#define BLOCK_MAX_COUNT 1024
#define EPSILON 1e-12f

//y : predicted,  y_ : expected
__global__ void mse(float* y, float* y_, int n, float* result) {
    extern __shared__ float blockBuffer[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    int blockSize = blockDim.x;

    float localSum = 0.0f;
    for (; i < n; i += stride)
        localSum += (y[i] - y_[i]) * (y[i] - y_[i]);

    blockBuffer[tid] = localSum;
    __syncthreads();

    while (blockSize > 1) {
        if ((blockSize % 2) && (tid == 0))
            blockBuffer[0] += blockBuffer[blockSize - 1];
        blockSize /= 2;
        if (tid < blockSize)
            blockBuffer[tid] += blockBuffer[tid + blockSize];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, blockBuffer[0] / n);
}

float mseCuda(Tensor& y_predicted, Tensor& y_expected) {
    float result;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    int n = y_predicted.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);

    y_predicted.toDevice();
    y_expected.toDevice();

    mse << <blocks, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (y_predicted.getData(), y_expected.getData(), n, d_result);

    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}


//y : predicted,  y_ : expected
__global__ void mseDer(float* y, float* y_, float* w, int n) {
    //assuming u is activations.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = (2.0f / n) * (y[i] - y_[i]);
}

void mseDerCuda(Tensor& y_predicted, Tensor& y_expected, Tensor& w) {
    int n = y_predicted.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    y_predicted.toDevice();
    y_expected.toDevice();
    w.toDevice();
    mseDer << <blocks, threadsPerBlock >> > (y_predicted.getData(), y_expected.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}


//y : predicted,  y_ : expected
__global__ void cce(float* y, float* y_, int n, float* result) {
    extern __shared__ float blockBuffer[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    int blockSize = blockDim.x;

    float localSum = 0.0f;
    for (; i < n; i += stride) {
        float p = fmaxf(y[i], EPSILON);   
        localSum += y_[i] * logf(p);      
    }

    blockBuffer[tid] = localSum;
    __syncthreads();

    while (blockSize > 1) {
        if ((blockSize % 2) && (tid == 0))
            blockBuffer[0] += blockBuffer[blockSize - 1];
        blockSize /= 2;
        if (tid < blockSize)
            blockBuffer[tid] += blockBuffer[tid + blockSize];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, -blockBuffer[0]);
}

float cceCuda(Tensor& y_predicted, Tensor& y_expected) {
    float result;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    int n = y_predicted.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);

    y_predicted.toDevice();
    y_expected.toDevice();

    cce << <blocks, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (y_predicted.getData(), y_expected.getData(), n, d_result);

    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}


//y : predicted,  y_ : expected
__global__ void cceDer(float* y, float* y_, float* w, int n) {
    //assuming u is activations.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = y[i] - y_[i];
}

void cceDerCuda(Tensor& y_predicted, Tensor& y_expected, Tensor& w) {
    int n = y_predicted.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    y_predicted.toDevice();
    y_expected.toDevice();
    w.toDevice();
    cceDer << <blocks, threadsPerBlock >> > (y_predicted.getData(), y_expected.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}
