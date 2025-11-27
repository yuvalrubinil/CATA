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


//C = A*B 
__global__ void matmul(float* A, float* B, float* C, int m, int n, int p) {
    //Basd on: https://www.youtube.com/watch?v=Q3GgbfGTnVc&t=345s
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float localSum = 0.0f;

    for (int k = 0; k < (n + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        if (row < m && k * TILE_SIZE + threadIdx.x < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + k * TILE_SIZE + threadIdx.x];
        if (col < p && k * TILE_SIZE + threadIdx.y < n)
            tileB[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * p + col];
        __syncthreads();

        for (int t = 0; t < TILE_SIZE; ++t)
            localSum += tileA[threadIdx.y][t] * tileB[t][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < p)
        C[row * p + col] = localSum;
}

void matmulCuda(Tensor& A, Tensor& B, Tensor& C) {
    int m = A.getShape()[0];
    int n = A.getShape()[1];
    int p = B.getShape()[1];
    dim3 blockShape(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((p + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    A.toDevice();
    B.toDevice();
    C.toDevice();
    matmul << <gridSize, blockShape >> > (A.getData(), B.getData(), C.getData(), m, n, p);
    cudaDeviceSynchronize();
}


//AT =  A.Transpose - (not really needed, can just reverse shape)
__global__ void transpose(float* A, float* AT, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
        AT[col * m + row] = A[row * n + col];
}


//w = A*u - (small sizes)
__global__ void matvec(float* A, float* u, int m, int n, float* w) {
    int threadLinear = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; threadLinear < m; threadLinear += stride) {
        float localSum = 0.0f;
        for (int i = 0; i < n; i++)
            localSum += A[threadLinear * n + i] * u[i];
        w[threadLinear] = localSum;
    }
}

void matvecCuda(Tensor& A, Tensor& u, Tensor& w) {
    int m = A.getShape()[0];
    int n = A.getShape()[1];
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, m);
    int blocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    A.toDevice();
    u.toDevice();
    w.toDevice();
    matvec << <blocks, threadsPerBlock >> > (A.getData(), u.getData(), m, n, w.getData());
    cudaDeviceSynchronize();
}
