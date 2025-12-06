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


// v[i] = beta * v[i] + (1-beta) * new_v[i]
__global__ void momentum(float* v, float* new_v, int n, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        v[i] = beta * v[i] + (1-beta) * new_v[i];
}

void momentumCuda(Tensor& v, Tensor& newV, float beta) {
    int n = v.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    v.toDevice();
    newV.toDevice();
    momentum << <blocks, threadsPerBlock >> > (v.getData(), newV.getData(), n, beta);
    cudaDeviceSynchronize();
}



// s[i] += (new_v[i] * new_v[i])
__global__ void accumulatedSquers(float* s, float* new_v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        s[i] += (new_v[i] * new_v[i]);
}

void accumulatedSquersCuda(Tensor& s, Tensor& newV) {
    int n = s.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    s.toDevice();
    newV.toDevice();
    accumulatedSquers << <blocks, threadsPerBlock >> > (s.getData(), newV.getData(), n);
    cudaDeviceSynchronize();
}

//w[i] -= (learningRate / (sqrtf(v[i]) + EPSILON)) * new_v[i]
__global__ void sqrtNormSubtraction(float* w, float* v, float* new_v, int n, float learningRate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] -= (learningRate / (sqrtf(v[i]) + EPSILON)) * new_v[i]; //w -= lr / (sqrt(vt) + eps) * (new der)
}

void adagradSubtractionCuda(Tensor& w, Tensor& s, Tensor& newV, float learningRate) {
    int n = newV.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    newV.toDevice();
    s.toDevice();
    w.toDevice();
    sqrtNormSubtraction << <blocks, threadsPerBlock >> > (w.getData(), s.getData(), newV.getData(), n, learningRate);
    cudaDeviceSynchronize();
}




// v[i] = beta * v[i] + (1 - beta) * (new_v[i] * new_v[i])
__global__ void momentumSquered(float* v, float* new_v, int n, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        v[i] = beta * v[i] + (1 - beta) * (new_v[i] * new_v[i]);
}

void momentumSqueredCuda(Tensor& v, Tensor& newV, float beta) {
    int n = v.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    v.toDevice();
    newV.toDevice();
    momentumSquered << <blocks, threadsPerBlock >> > (v.getData(), newV.getData(), n, beta);
    cudaDeviceSynchronize();
}


void rmsPropSubtractionCuda(Tensor& w, Tensor& s, Tensor& newV, float learningRate) {
    int n = newV.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    newV.toDevice();
    s.toDevice();
    w.toDevice();
    sqrtNormSubtraction << <blocks, threadsPerBlock >> > (w.getData(), s.getData(), newV.getData(), n, learningRate);
    cudaDeviceSynchronize();
}

//w[i] -= ((learningRate * v_hat) / (sqrtf(s_hat) + EPSILON))
__global__ void adamSubtraction(float* w, float* s, float* v, int n, float learningRate, float bias_cor_beta1, float bias_cor_beta2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float v_hat = v[i] / bias_cor_beta1;
        float s_hat = s[i] / bias_cor_beta2;
        w[i] -= ((learningRate * v_hat) / (sqrtf(s_hat) + EPSILON)); //w -= (lr * v_hat) / (sqrt(s_hat) + eps)
    }
}

void adamSubtractionCuda(Tensor& w, Tensor& s, Tensor& v, float learningRate, float beta1, float beta2, int t) {
    int n = w.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    s.toDevice();
    v.toDevice();
    w.toDevice();
    float bias_cor_beta1 = 1 - powf(beta1, t);
    float bias_cor_beta2 = 1 - powf(beta2, t);
    adamSubtraction << <blocks, threadsPerBlock >> > (w.getData(), s.getData(), v.getData(), n, learningRate, bias_cor_beta1, bias_cor_beta2);
    cudaDeviceSynchronize();
}