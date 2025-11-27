#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include "../tensor.cuh"
#include "../ops.cuh"


#define BLOCK_MAX_SIZE 256
#define BLOCK_MAX_COUNT 1024
#define TILE_SIZE 16

//Normal dist init
__global__ void normalInit(float* u, int n, unsigned long long seed, float mean, float std)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;
    curand_init(seed, i, 0, &state);
    for (; i < n; i += stride)
        u[i] = mean + std * curand_normal(&state);
}

void normalInitCuda(Tensor& u, float mean, float std) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    normalInit << <blocks, threadsPerBlock >> > (u.getData(), n, seed, mean, std);
    cudaDeviceSynchronize();
}


//Xavier init
__global__ void xavierInit(float* u, int n, int fan_in, int fan_out, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, tid, 0, &state);
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    for (int i = tid; i < n; i += stride) {
        float r = curand_uniform(&state); 
        u[i] = -limit + r * (2.0f * limit);
    }
}

void xavierInitCuda(Tensor& u, int fan_in, int fan_out) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    xavierInit << <blocks, threadsPerBlock >> > (u.getData(), n, fan_in, fan_out, seed);
    cudaDeviceSynchronize();
}


//HeNormal init
__global__ void heNormalInit(float* u, int n, int fan_in, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    curandState state;
    for (; i < n; i += stride) {
        curand_init(seed, i, 0, &state);  
        float r = curand_normal(&state);  
        u[i] = r * sqrtf(2.0f / fan_in);
    }
}

void heNormalInitCuda(Tensor& u, int fan_in) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    heNormalInit << <blocks, threadsPerBlock >> > (u.getData(), n, fan_in, seed);
    cudaDeviceSynchronize();
}


//HeUniform init
__global__ void heUniformInit(float* u, int n, int fan_in, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;
    curand_init(seed, i, 0, &state);
    float limit = sqrtf(6.0f / fan_in);
    for (; i < n; i += stride) {
        float r = curand_uniform(&state);
        float val = (r * 2.0f - 1.0f);
        u[i] = val * limit;
    }
}

void heUniformInitCuda(Tensor& u, int fan_in) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    heUniformInit << <blocks, threadsPerBlock >> > (u.getData(), n, fan_in, seed);
    cudaDeviceSynchronize();
}


//Zeros init
void zerosInitCuda(Tensor& u) {
    fillCuda(u.getData(), u.getSize(), 0.0f);
}



void initWeights(Tensor& weights, const std::string& method, int fan_in, int fan_out) {
    if (method == "normal")
        normalInitCuda(weights);
    else if (method == "xavier")
        xavierInitCuda(weights, fan_in, fan_out);
    else if (method == "he_normal")
        heNormalInitCuda(weights, fan_in);
    else if (method == "he_uniform")
        heUniformInitCuda(weights, fan_in);
    else if (method == "zeros")
        zerosInitCuda(weights);
    else
        throw std::runtime_error("Unknown initializer method");
}


