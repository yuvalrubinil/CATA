#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <vector>
#include "../tensor.cuh"
#include "../ops.cuh"


#define BLOCK_MAX_SIZE 256
#define BLOCK_MAX_COUNT 1024

__host__ void printDeviceProperties()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0

    printf("--- Device Properties ---\n");
    printf("Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max block dimensions: (%d, %d, %d)\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Streaming Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("--------------------------------------------------\n\n");
    // Note: The maximum *concurrent* threads is generally prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount
    // The maxThreadsPerMultiProcessor varies by architecture (often 1024 or 2048).
}

void copyVectorToDevice_float(float** arr_d, const std::vector<float>& arr_h) {
    cudaMalloc(arr_d, sizeof(float) * arr_h.size());
    cudaMemcpy(*arr_d, arr_h.data(), sizeof(float) * arr_h.size(), cudaMemcpyHostToDevice);
}

void copyVectorToDevice_int(int** arr_d, const std::vector<int>& arr_h) {
    cudaMalloc(arr_d, sizeof(int) * arr_h.size());
    cudaMemcpy(*arr_d, arr_h.data(), sizeof(int) * arr_h.size(), cudaMemcpyHostToDevice);
}

void copyArrayToDevice_int(int** arr_d, int* arr_h, int n) {
    cudaMalloc(arr_d, sizeof(int) * n);
    cudaMemcpy(*arr_d, arr_h, sizeof(int) * n, cudaMemcpyHostToDevice);
}

void copySpecificToHost_int(int* ptr_h, int* ptr_d, int index) {
    cudaMemcpy(ptr_h, ptr_d + index, sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void fill(float* u, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        u[i] = value;
}

void fillCuda(float* u, int n, float value) {
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    fill << <blocks, threadsPerBlock >> > (u, n, value);
    cudaDeviceSynchronize();
}

void fillTensorCuda(Tensor& u, float value) {
    u.toDevice();
    fillCuda(u.getData(), u.getSize(), value);
}

__global__ void oneHot(float* u, int n, int hotIdx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        u[i] = (i == hotIdx) ? 1 : 0;
}

void oneHotCuda(Tensor& u, int hotIdx) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    oneHot << <blocks, threadsPerBlock >> > (u.getData(), n, hotIdx);
    cudaDeviceSynchronize();
}

__global__ void copy(float* u, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        w[i] = u[i];
}

void copyCuda(Tensor& u, Tensor& w) {
    int n = u.getSize();
    int threadsPerBlock = std::min(BLOCK_MAX_SIZE, n);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, BLOCK_MAX_COUNT);
    u.toDevice();
    w.toDevice();
    copy << <blocks, threadsPerBlock >> > (u.getData(), w.getData(), n);
    cudaDeviceSynchronize();
}
