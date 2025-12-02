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
#define EPSILON 1e-8


// u[i] = beta * u[i] + (1-beta) * v[i]
__global__ void momentum(float* u, float* v, int n, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        u[i] = beta * u[i] + (1-beta) * v[i];
}

void momentumCuda(Tensor& u, Tensor& v, float beta) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    momentum << <blocks, threadsPerBlock >> > (u.getData(), v.getData(), n, beta);
    cudaDeviceSynchronize();
}

// u[i] = beta * u[i] + (1 - beta) * (v[i] * v[i])
__global__ void momentumSquered(float* u, float* v, int n, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        u[i] = beta * u[i] + (1 - beta) * (v[i] * v[i]);
}

void momentumSqueredCuda(Tensor& u, Tensor& v, float beta) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    momentumSquered << <blocks, threadsPerBlock >> > (u.getData(), v.getData(), n, beta);
    cudaDeviceSynchronize();
}


__global__ void rmsPropSubtraction(float* w, float* v, float* u, int n, float learningRate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] -= (learningRate / (sqrtf(v[i]) + EPSILON)) * u[i]; //w -= lr / (sqrt(vt) + eps) * (new der)
}

void rmsPropSubtractionCuda(Tensor& w, Tensor& v, Tensor& u, float learningRate) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    w.toDevice();
    rmsPropSubtraction << <blocks, threadsPerBlock >> > (w.getData(), v.getData(), u.getData(), n, learningRate);
    cudaDeviceSynchronize();
}