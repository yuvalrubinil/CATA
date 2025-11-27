#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "../tensor.cuh"
#include "../ops.cuh"


#define BLOCK_MAX_SIZE 256
#define BLOCK_MAX_COUNT 1024
#define TILE_SIZE 16


// w[i] = u[i] + v[i]
__global__ void plus(float* u, float* v, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = u[i] + v[i];
}

void plusCuda(Tensor& u, Tensor& v, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    w.toDevice();
    plus <<<blocks, threadsPerBlock >>> (u.getData(), v.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}


//w[i] = u[i] - v[i]
__global__ void minus(float* u, float* v, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = u[i] - v[i];
}

void minusCuda(Tensor& u, Tensor& v, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    w.toDevice();
    minus << <blocks, threadsPerBlock >> > (u.getData(), v.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}


//w[i] = u[i] * v[i] 
__global__ void star(float* u, float* v, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = u[i] * v[i];
}

void starCuda(Tensor& u, Tensor& v, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    w.toDevice();
    star << <blocks, threadsPerBlock >> > (u.getData(), v.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}


//w[i] = u[i] / v[i] 
__global__ void slash(float* u, float* v, float* w, int n, int* errorFlag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        if (v[i] != 0.0f)
            w[i] = u[i] / v[i];
        else
            atomicExch(errorFlag, 1);
    }
}

void slashCuda(Tensor& u, Tensor& v, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    w.toDevice();
    int error_h = 0;
    int* error_d;
    cudaMalloc(&error_d, sizeof(int));
    cudaMemcpy(error_d, &error_h, sizeof(int), cudaMemcpyHostToDevice);
    slash << <blocks, threadsPerBlock >> > (u.getData(), v.getData(), w.getData(), n, error_d);
    cudaDeviceSynchronize();

    cudaMemcpy(&error_h, error_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(error_d);  // free memory

    if (error_h)
        throw std::runtime_error("Division by zero detected in Tensor.");
}


//w[i] = u[i]^pow 
__global__ void starStar(float* u, float* w, int n, float pow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = powf(u[i], pow);
}

void starStarCuda(Tensor& u, Tensor& w, float pow) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    starStar << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n, pow);
    cudaDeviceSynchronize();
}


//w[i] = e^u[i]
__global__ void exp(float* u, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = expf(u[i]);
}

void expCuda(Tensor& u, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    exp << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}


//w[i] = logb(u[i])
__global__ void logb(float* u, float* w, int n, float base) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = log10f(u[i]) / log10f(base); // log rules: logb(x) = log(x) / log(b)
}

void logCuda(Tensor& u, Tensor& w, float base) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    logb << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n, base);
    cudaDeviceSynchronize();
}


//w[i] = sqrt(u[i])
__global__ void sqrt(float* u, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = sqrtf(u[i]);
}

void sqrtCuda(Tensor& u, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    sqrt << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}


//w[i] = u[i] + scaler
__global__ void addScaler(float* u, float* w, int n, float scaler) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = u[i] + scaler;
}

void addScalerCuda(Tensor& u, Tensor& w, float scaler) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    addScaler << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n, scaler);
    cudaDeviceSynchronize();
}


//w[i] = scaler * u[i]
__global__ void multByScaler(float* u, float* w, int n, float scaler) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = scaler * u[i];
}

void multByScalerCuda(Tensor& u, Tensor& w, float scaler) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    multByScaler << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n, scaler);
    cudaDeviceSynchronize();
}


//w[i] = u[i] / scaler
__global__ void divideByScaler(float* u, float* w, int n, float scaler) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = u[i] / scaler;
}

void divideByScalerCuda(Tensor& u, Tensor& w, float scaler) {
    if (!scaler)
        throw std::runtime_error("Division by zero detected in Tensor.");
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    divideByScaler << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n, scaler);
    cudaDeviceSynchronize();
}


//dot(u*v)
__global__ void dot(float* u, float* v, int n, float* result) {
    extern __shared__ float blockBuffer[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    int blockSize = blockDim.x;

    float localSum = 0.0f;
    for (; i < n; i += stride)
        localSum += u[i] * v[i];

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

    if (tid == 0) atomicAdd(result, blockBuffer[0]);
}

float dotCuda(Tensor& u, Tensor& v) {
    float result;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0.0f, sizeof(float));

    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    dot << <blocks, threadsPerBlock >> > (u.getData(), v.getData(), n, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return result;
}


//sigma u[i]
__global__ void sum(float* u, int n, float* result) {
    extern __shared__ float blockBuffer[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    int blockSize = blockDim.x;

    float localSum = 0.0f;
    for (; i < n; i += stride)
        localSum += u[i];

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

    if (tid == 0) atomicAdd(result, blockBuffer[0]);
}

float sumCuda(Tensor& u) {
    float result;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0.0f, sizeof(float));

    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    sum << <blocks, threadsPerBlock >> > (u.getData(), n, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return result;
}


// 1/n * sigma u[i]
float meanCuda(Tensor& u) {
    float sum = sumCuda(u);
    return sum / u.getSize();
}


// W = u * v.T
__global__ void outerProduct(float* u, float* v, float* w, int m, int n)
{
    int strideY = gridDim.y * blockDim.y;
    int strideX = gridDim.x * blockDim.x;
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < m; i += strideY){
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += strideX)
            w[i * n + j] = u[i] * v[j];
    }
}

void outerProductCuda(Tensor& u, Tensor& v, Tensor& W) {
    int m = u.getShape()[0]; 
    int n = v.getShape()[0];

    dim3 blockShape(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE); // x=n, y=m
    u.toDevice();
    v.toDevice();
    W.toDevice();
    outerProduct << <gridSize, blockShape >> > (u.getData(), v.getData(), W.getData(), m, n);
    cudaDeviceSynchronize();
}


// W += u * v.T
__global__ void outerProductPlusEqual(float* u, float* v, float* w, int m, int n)
{
    int strideY = gridDim.y * blockDim.y;
    int strideX = gridDim.x * blockDim.x;
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < m; i += strideY) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += strideX)
            w[i * n + j] += u[i] * v[j];
    }
}

void outerProductPlusEqualCuda(Tensor& u, Tensor& v, Tensor& W) {
    int m = u.getShape()[0];
    int n = v.getShape()[0];

    dim3 blockShape(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE); // x=n, y=m
    u.toDevice();
    v.toDevice();
    W.toDevice();
    outerProductPlusEqual << <gridSize, blockShape >> > (u.getData(), v.getData(), W.getData(), m, n);
    cudaDeviceSynchronize();
}


// u[i] -= scaler * v[i]
__global__ void subtractByScaler(float* u, float* v, float scaler, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        u[i] -= scaler * v[i];
}

void subtractByScaleCuda(Tensor& u, Tensor& v, float scaler) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    subtractByScaler << <blocks, threadsPerBlock >> > (u.getData(), v.getData(), scaler, n);
    cudaDeviceSynchronize();
}


//argmax(u) - by the Chat
__global__ void argmaxSmall(const float* u, int n, int* result) {
    __shared__ float s_val[BLOCK_MAX_SIZE];
    __shared__ int s_idx[BLOCK_MAX_SIZE];

    int tid = threadIdx.x;

    s_val[tid] = -FLT_MAX;
    s_idx[tid] = -1;

    if (tid < n) {
        s_val[tid] = u[tid];
        s_idx[tid] = tid;
    }
    __syncthreads();

    for (int stride = BLOCK_MAX_SIZE / 2; stride > 0; stride >>= 1) {  // 256 → 128 → 64 ...
        if (tid < stride && tid + stride < n) {
            if (s_val[tid] < s_val[tid + stride]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) *result = s_idx[0];
}

int argmaxSmallCuda(Tensor& u) {
    int n = u.getSize();
    if (n > BLOCK_MAX_SIZE) {
        printf("Error: argmaxSmall only supports vectors <= 256\n");
        return -1;
    }

    u.toDevice();

    int threadsPerBlock = BLOCK_MAX_SIZE;   // always launch full 256 threads
    int blocks = 1;

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));

    argmaxSmall << <blocks, threadsPerBlock >> > (u.getData(), n, d_result);
    cudaDeviceSynchronize();

    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}


//max(u) 
__global__ void max(float* u, int n, float* max) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int blockStart = blockIdx.x * blockDim.x;
    int globalThreads = gridDim.x * blockDim.x;

    float localMax = -FLT_MAX;
    int i = blockStart + tid;
    for (; i < n; i += globalThreads)
        localMax = fmaxf(localMax, u[i]);

    sdata[tid] = localMax;
    __syncthreads();

    while (1 < blockSize) {
        if ((blockSize % 2) && (tid == 0))
            sdata[0] = fmaxf(sdata[0], sdata[blockSize - 1]);
        blockSize /= 2;
        if (tid < blockSize)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + blockSize]);
        __syncthreads();
    }

    if (tid == 0) *max = sdata[0];
}

float maxCuda(Tensor& u) {
    float result;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0.0f, sizeof(float));

    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    max << <blocks, threadsPerBlock >> > (u.getData(), n, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return result;
}
