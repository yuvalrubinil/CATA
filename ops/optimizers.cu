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



// u[i] = u[i] + (v[i] * v[i])
__global__ void accumulatedSquers(float* u, float* v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        u[i] = u[i] + (v[i] * v[i]);
}

void accumulatedSquersCuda(Tensor& u, Tensor& v) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    v.toDevice();
    accumulatedSquers << <blocks, threadsPerBlock >> > (u.getData(), v.getData(), n);
    cudaDeviceSynchronize();
}

//w[i] -= (learningRate / (sqrtf(v[i]) + EPSILON)) * u[i]
__global__ void momentumSqrtNormSubtraction(float* w, float* v, float* u, int n, float learningRate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] -= (learningRate / (sqrtf(v[i]) + EPSILON)) * u[i]; //w -= lr / (sqrt(vt) + eps) * (new der)
}


void adagradSubtractionCuda(Tensor& w, Tensor& accumulatedSquers, Tensor& newDer, float learningRate) {
    int n = newDer.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    newDer.toDevice();
    accumulatedSquers.toDevice();
    w.toDevice();
    momentumSqrtNormSubtraction << <blocks, threadsPerBlock >> > (w.getData(), accumulatedSquers.getData(), newDer.getData(), n, learningRate);
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


void rmsPropSubtractionCuda(Tensor& w, Tensor& squeredMomentum, Tensor& newDer, float learningRate) {
    int n = newDer.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    newDer.toDevice();
    squeredMomentum.toDevice();
    w.toDevice();
    momentumSqrtNormSubtraction << <blocks, threadsPerBlock >> > (w.getData(), squeredMomentum.getData(), newDer.getData(), n, learningRate);
    cudaDeviceSynchronize();
}