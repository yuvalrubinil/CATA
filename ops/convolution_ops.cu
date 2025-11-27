#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "../tensor.cuh"
#include "../ops.cuh"


#define BLOCK_MAX_SIZE 256
#define BLOCK_MAX_COUNT 1024
#define TILE_SIZE 16

__global__ void convolution(
    float* tensor, int* tensorShape, int* tensorStrides,
    float* kernels, int kernelWidth, int k, int stride, int paddingFrame, int tileWidth, int tileHeight,
    float* result, int* resultShape, int* resultStrides)
{
    int featureMapIdx = blockIdx.z;
    int resultCol = blockIdx.x * blockDim.x + threadIdx.x;
    int resultRow = blockIdx.y * blockDim.y + threadIdx.y;
    int imageIdx = featureMapIdx / k;
    int kernelIdx = featureMapIdx % k;
    int threadLinear = threadIdx.y * blockDim.x + threadIdx.x;
    bool inRange = resultRow < resultShape[1] && resultCol < resultShape[2];
    int tileSize = tileWidth * tileHeight;
    int kernelSize = kernelWidth * kernelWidth;

    extern __shared__ float shared[];
    float* tile = shared;
    float* kernel = shared + tileSize;
    int totalThreads = blockDim.x * blockDim.y;

    //Every thread in the block fills the tile. (striding the threads accros the img window)
    for (int i = threadLinear; i < tileSize; i += totalThreads) {
        int tileRow = i / tileWidth;
        int tileCol = i % tileWidth;
        int tensorRow = blockIdx.y * blockDim.y * stride + tileRow - paddingFrame;
        int tensorCol = blockIdx.x * blockDim.x * stride + tileCol - paddingFrame;
        if (0 <= tensorRow && tensorRow < tensorShape[1] && 0 <= tensorCol && tensorCol < tensorShape[2])
            tile[i] = tensor[imageIdx * tensorStrides[0] + tensorRow * tensorStrides[1] + tensorCol * tensorStrides[2]];
        else
            tile[i] = 0.0f; //Padding
    }

    for (int k = threadLinear; k < kernelSize; k += totalThreads)
        kernel[k] = kernels[kernelIdx * kernelSize + k];

    __syncthreads();

    if (inRange) {
        float localSum = 0.0f;
        int tileRow = threadIdx.y * stride;
        int tileCol = threadIdx.x * stride;
        for (int i = 0; i < kernelWidth; i++) {
            for (int j = 0; j < kernelWidth; j++)
                localSum += tile[(tileRow + i) * tileWidth + tileCol + j] * kernel[i * kernelWidth + j];
        }
        result[featureMapIdx * resultStrides[0] + resultRow * resultStrides[1] + resultCol * resultStrides[2]] = localSum;
    }
}

void convolutionCuda(Tensor& tensor, Tensor& kernels, Tensor& featureMap, int paddingFrame, int stride) {
    int imgCount = tensor.getShape()[0];
    int kernelsCount = kernels.getShape()[0];
    int kernelWidth = kernels.getShape()[2];
    int fmWidth = featureMap.getShape()[2];
    int fmHeight = featureMap.getShape()[1];

    int blocksPerRow = ceilf((float)fmWidth / TILE_SIZE);
    int blocksPerCol = ceilf((float)fmHeight / TILE_SIZE);

    dim3 blockShape(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(blocksPerRow, blocksPerCol, imgCount * kernelsCount);

    int tileWidth = (TILE_SIZE - 1) * stride + kernelWidth;
    int tileHeight = (TILE_SIZE - 1) * stride + kernelWidth;
    int sharedSize = (tileWidth * tileHeight + kernelWidth * kernelWidth) * sizeof(float);

    tensor.toDevice();
    kernels.toDevice();
    featureMap.toDevice();


    convolution << <gridSize, blockShape, sharedSize >> > (
        tensor.getData(), tensor.getShape_d(), tensor.getStrides_d(),
        kernels.getData(), kernelWidth, kernelsCount, stride, paddingFrame, tileWidth, tileHeight,
        featureMap.getData(), featureMap.getShape_d(), featureMap.getStrides_d()
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error after kernel convolutionCuda: %s\n", cudaGetErrorString(err));
        abort();
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Runtime error after kernel convolutionCuda sync: %s\n", cudaGetErrorString(err));
        abort();
    }
}


__global__ void pooling(float* featureMap, int* featureMapShape, int* featureMapStrides,
                        float* result, int* resultShape, int* resultStrides, float* pooledIndices,
                        char mode, int poolingSize)
{
    int resultRow = blockDim.y * blockIdx.y + threadIdx.y;
    int resultCol = blockDim.x * blockIdx.x + threadIdx.x;
    int fmRow = resultRow * poolingSize;
    int fmCol = resultCol * poolingSize;
    int featureMapIdx = blockIdx.z * featureMapStrides[0] + fmRow * featureMapStrides[1] + fmCol * featureMapStrides[2];
    int resultIdx = blockIdx.z * resultStrides[0] + resultRow * resultStrides[1] + resultCol * resultStrides[2];
    bool inRange = resultRow < resultShape[1] && resultCol < resultShape[2];
    if (inRange) {
        float local = (mode == 'm') ? -FLT_MAX : 0.0f;
        float poolingWindowLinear = -1; //will be casted to integer in backprop.
        for (int i = 0; i < poolingSize; i++)
        {
            for (int j = 0; j < poolingSize; j++)
            {
                if ((fmRow + i) < featureMapShape[1] && (fmCol + j) < featureMapShape[2]) {
                    float val = featureMap[featureMapIdx + i * featureMapStrides[1] + j * featureMapStrides[2]];
                    if (mode == 'm') {
                        if (local < val) {
                            local = val;
                            poolingWindowLinear = i * poolingSize + j;
                        }
                    }
                    else
                        local += val;
                }
            }
        }
        if (mode == 'm') {
            result[resultIdx] = local;
            pooledIndices[resultIdx] = poolingWindowLinear;
        }
        else
            result[resultIdx] = local / (poolingSize * poolingSize);
    }
}

void poolingCuda(Tensor& featureMap, Tensor& pooledIndices, Tensor& result, char mode, int poolSize) {
    int fmCount = featureMap.getShape()[0];
    int fmWidth = featureMap.getShape()[1];
    int fmHeight = featureMap.getShape()[2];
    int poolingMapWidth = result.getShape()[2];
    int poolingMapHeight = result.getShape()[1];

    int blocksPerRow = ceilf((float)poolingMapWidth / TILE_SIZE);
    int blocksPerCol = ceilf((float)poolingMapHeight / TILE_SIZE);

    dim3 blockShape(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(blocksPerRow, blocksPerCol, fmCount);


    featureMap.toDevice();
    pooledIndices.toDevice();
    result.toDevice();

    pooling << <gridSize, blockShape >> > (
        featureMap.getData(), featureMap.getShape_d(), featureMap.getStrides_d(),
        result.getData(), result.getShape_d(), result.getStrides_d(), pooledIndices.getData(),
        mode, poolSize
        );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error after kernel poolingCuda: %s\n", cudaGetErrorString(err));
        abort();
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Runtime error after kernel poolingCuda sync: %s\n", cudaGetErrorString(err));
        abort();
    }
}



__global__ void convolutionChainRule(
    float* tensor, int* tensorShape, int* tensorStrides,
    float* pooled, int* pooledShape, int* pooledStrides,
    float* dc_dz,
    int stride, int paddingFrame, int poolingSize, char poolMode, int k,
    float* dc_dk, int* dc_dk_Shape, float* dc_db, int* dc_db_Strides)
{
    int poolMapIdx = blockIdx.z;
    int poolingCol = blockIdx.x * blockDim.x + threadIdx.x;
    int poolingRow = blockIdx.y * blockDim.y + threadIdx.y;
    int imgIdx = poolMapIdx / k;

    int blockThreads = blockDim.x * blockDim.y;
    int threadLinear = threadIdx.y * blockDim.x + threadIdx.x;

    bool inRange = (poolingRow < pooledShape[1] && poolingCol < pooledShape[2]);

    int kernelWidth = dc_dk_Shape[2];
    int kernelSize = kernelWidth * kernelWidth;
    int kernelIdx = blockIdx.z % k;

    int tileWidth = (blockDim.x - 1) * stride + kernelWidth;
    int tileHeight = (blockDim.y - 1) * stride + kernelWidth;
    int tileSize = tileWidth * tileHeight;

    extern __shared__ float shared[];
    float* tile = shared;
    float* dc_dk_shared = shared + tileSize;

    for (int i = threadLinear; i < tileSize; i += blockThreads) {
        int tileRow = i / tileWidth;
        int tileCol = i % tileWidth;
        int imgRow = blockIdx.y * blockDim.y * stride + tileRow - paddingFrame;
        int imgCol = blockIdx.x * blockDim.x * stride + tileCol - paddingFrame;

        if (imgRow >= 0 && imgRow < tensorShape[1] && imgCol >= 0 && imgCol < tensorShape[2])
            tile[i] = tensor[imgIdx * tensorStrides[0] + imgRow * tensorStrides[1] + imgCol * tensorStrides[2]];
        else
            tile[i] = 0.0f; // padding
    }

    __syncthreads();


    if (inRange) {
        int pooledLinear = poolMapIdx * pooledStrides[0] + poolingRow * pooledStrides[1] + poolingCol * pooledStrides[2];
        float gradValue = dc_dz[pooledLinear];
        if (poolMode == 'a') gradValue /= float(poolingSize * poolingSize);

        int fmBaseRow = poolingRow * poolingSize;
        int fmBaseCol = poolingCol * poolingSize;
        int tileBaseRow = fmBaseRow * stride; // base position of the pooling window in the tile
        int tileBaseCol = fmBaseCol * stride;
        if (poolMode == 'm') {
            int maxIndex = (int)pooled[pooledLinear]; // linear index inside pooling window 0,1,2,3
            int maxRowInWindow = maxIndex / poolingSize;
            int maxColInWindow = maxIndex % poolingSize;
            dc_db[poolMapIdx * dc_db_Strides[0] + (fmBaseRow + maxRowInWindow) * dc_db_Strides[1] + fmBaseCol + maxColInWindow] = gradValue; //updating bias grad
            for (int wr = 0; wr < kernelWidth; ++wr) {
                for (int wc = 0; wc < kernelWidth; ++wc) {
                    int tileRow = tileBaseRow + (maxRowInWindow)*stride + wr;
                    int tileCol = tileBaseCol + (maxColInWindow)*stride + wc;
                    if (tileRow < tileHeight && tileCol < tileWidth)
                        atomicAdd(&dc_dk_shared[wr * kernelWidth + wc], gradValue * tile[tileRow * tileWidth + tileCol]);
                }
            }
        }
        else {
            // sum contributions from all pooling window positions
            for (int pr = 0; pr < poolingSize; pr++) {
                for (int pc = 0; pc < poolingSize; pc++) {
                    if (poolingRow + pr < pooledShape[1] && poolingCol + pc < pooledShape[2]) {
                        dc_db[poolMapIdx * dc_db_Strides[0] + (fmBaseRow + pr) * dc_db_Strides[1] + fmBaseCol + pc] = gradValue; //updating bias grad
                        for (int wr = 0; wr < kernelWidth; ++wr) {
                            for (int wc = 0; wc < kernelWidth; ++wc) {
                                int tileRow = tileBaseRow + pr * stride + wr;
                                int tileCol = tileBaseCol + pc * stride + wc;
                                atomicAdd(&dc_dk_shared[wr * kernelWidth + wc], gradValue * tile[tileRow * tileWidth + tileCol]);
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // one kernel writes to global memo
    for (int k = 0; k < kernelSize && (threadLinear == 0); k++)
        dc_dk[kernelIdx * kernelSize + k] = dc_dk_shared[k];

}

void convolutionChainRuleCuda(Tensor& tensor, Tensor& pooledData, Tensor& dc_dz, Tensor& dc_db, Tensor& dc_dk,
    int paddingFrame, int stride, int kernelWidth, char poolMode, int poolSize, int k) {
    int poolMapCount = pooledData.getShape()[0];
    int poolWidth = pooledData.getShape()[2];
    int poolHeight = pooledData.getShape()[1];

    int blocksPerRow = ceilf((float)poolWidth / TILE_SIZE);
    int blocksPerCol = ceilf((float)poolHeight / TILE_SIZE);

    dim3 blockShape(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(blocksPerRow, blocksPerCol, poolMapCount);

    int tileWidth = (TILE_SIZE - 1) * stride + kernelWidth;
    int sharedSize = (tileWidth * tileWidth + kernelWidth * kernelWidth) * sizeof(float);

    tensor.toDevice();
    pooledData.toDevice();
    dc_dz.toDevice();
    dc_dk.toDevice();

    convolutionChainRule << <gridSize, blockShape, sharedSize >> > (
        tensor.getData(), tensor.getShape_d(), tensor.getStrides_d(),
        pooledData.getData(), pooledData.getShape_d(), pooledData.getStrides_d(),
        dc_dz.getData(),
        stride, paddingFrame, poolSize, poolMode, k,
        dc_dk.getData(), dc_dk.getShape_d(), dc_db.getData(), dc_db.getStrides_d());


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error after kernel convolutionChainRuleCuda: %s\n", cudaGetErrorString(err));
        abort();
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Runtime error after kernel convolutionChainRuleCuda sync: %s\n", cudaGetErrorString(err));
        abort();
    }
}



__global__ void convolutionChainRulePlusEqual(
    float* tensor, int* tensorShape, int* tensorStrides,
    float* pooled, int* pooledShape, int* pooledStrides,
    float* dc_dz,
    int stride, int paddingFrame, int poolingSize, char poolMode, int k,
    float* dc_dk, int* dc_dk_Shape, float* dc_db, int* dc_db_Strides)
{
    int poolMapIdx = blockIdx.z;
    int poolingCol = blockIdx.x * blockDim.x + threadIdx.x;
    int poolingRow = blockIdx.y * blockDim.y + threadIdx.y;
    int imgIdx = poolMapIdx / k;

    int blockThreads = blockDim.x * blockDim.y;
    int threadLinear = threadIdx.y * blockDim.x + threadIdx.x;

    bool inRange = (poolingRow < pooledShape[1] && poolingCol < pooledShape[2]);

    int kernelWidth = dc_dk_Shape[2];
    int kernelSize = kernelWidth * kernelWidth;
    int kernelIdx = blockIdx.z % k;

    int tileWidth = (blockDim.x - 1) * stride + kernelWidth;
    int tileHeight = (blockDim.y - 1) * stride + kernelWidth;
    int tileSize = tileWidth * tileHeight;

    extern __shared__ float shared[];
    float* tile = shared;
    float* dc_dk_shared = shared + tileSize;

    for (int i = threadLinear; i < tileSize; i += blockThreads) {
        int tileRow = i / tileWidth;
        int tileCol = i % tileWidth;
        int imgRow = blockIdx.y * blockDim.y * stride + tileRow - paddingFrame;
        int imgCol = blockIdx.x * blockDim.x * stride + tileCol - paddingFrame;

        if (imgRow >= 0 && imgRow < tensorShape[1] && imgCol >= 0 && imgCol < tensorShape[2])
            tile[i] = tensor[imgIdx * tensorStrides[0] + imgRow * tensorStrides[1] + imgCol * tensorStrides[2]];
        else
            tile[i] = 0.0f; // padding
    }

    __syncthreads();


    if (inRange) {
        int pooledLinear = poolMapIdx * pooledStrides[0] + poolingRow * pooledStrides[1] + poolingCol * pooledStrides[2];
        float gradValue = dc_dz[pooledLinear];
        if (poolMode == 'a') gradValue /= float(poolingSize * poolingSize);

        int fmBaseRow = poolingRow * poolingSize;
        int fmBaseCol = poolingCol * poolingSize;
        int tileBaseRow = fmBaseRow * stride; // base position of the pooling window in the tile
        int tileBaseCol = fmBaseCol * stride;
        if (poolMode == 'm') {
            int maxIndex = (int)pooled[pooledLinear]; // linear index inside pooling window 0,1,2,3
            int maxRowInWindow = maxIndex / poolingSize;
            int maxColInWindow = maxIndex % poolingSize;
            dc_db[poolMapIdx * dc_db_Strides[0] + (fmBaseRow + maxRowInWindow) * dc_db_Strides[1] + fmBaseCol + maxColInWindow] += gradValue; //updating bias grad
            for (int wr = 0; wr < kernelWidth; ++wr) {
                for (int wc = 0; wc < kernelWidth; ++wc) {
                    int tileRow = tileBaseRow + (maxRowInWindow)*stride + wr;
                    int tileCol = tileBaseCol + (maxColInWindow)*stride + wc;
                    if (tileRow < tileHeight && tileCol < tileWidth)
                        atomicAdd(&dc_dk_shared[wr * kernelWidth + wc], gradValue * tile[tileRow * tileWidth + tileCol]);
                }
            }
        }
        else {
            // sum contributions from all pooling window positions
            for (int pr = 0; pr < poolingSize; pr++) {
                for (int pc = 0; pc < poolingSize; pc++) {
                    if (poolingRow + pr < pooledShape[1] && poolingCol + pc < pooledShape[2]) {
                        dc_db[poolMapIdx * dc_db_Strides[0] + (fmBaseRow + pr) * dc_db_Strides[1] + fmBaseCol + pc] += gradValue; //updating bias grad
                        for (int wr = 0; wr < kernelWidth; ++wr) {
                            for (int wc = 0; wc < kernelWidth; ++wc) {
                                int tileRow = tileBaseRow + pr * stride + wr;
                                int tileCol = tileBaseCol + pc * stride + wc;
                                atomicAdd(&dc_dk_shared[wr * kernelWidth + wc], gradValue * tile[tileRow * tileWidth + tileCol]);
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // one kernel writes to global memo
    for (int k = 0; k < kernelSize && (threadLinear == 0); k++)
        atomicAdd(&dc_dk[kernelIdx * kernelSize + k], dc_dk_shared[k]);

}


void convolutionChainRulePlusEqualCuda(Tensor& tensor, Tensor& pooledData, Tensor& dc_dz, Tensor& dc_db, Tensor& dc_dk,
    int paddingFrame, int stride, int kernelWidth, char poolMode, int poolSize, int k) {
    int poolMapCount = pooledData.getShape()[0];
    int poolWidth = pooledData.getShape()[2];
    int poolHeight = pooledData.getShape()[1];

    int blocksPerRow = ceilf((float)poolWidth / TILE_SIZE);
    int blocksPerCol = ceilf((float)poolHeight / TILE_SIZE);

    dim3 blockShape(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(blocksPerRow, blocksPerCol, poolMapCount);

    int tileWidth = (TILE_SIZE - 1) * stride + kernelWidth;
    int sharedSize = (tileWidth * tileWidth + kernelWidth * kernelWidth) * sizeof(float);

    tensor.toDevice();
    pooledData.toDevice();
    dc_dz.toDevice();
    dc_dk.toDevice();

    convolutionChainRulePlusEqual << <gridSize, blockShape, sharedSize >> > (
        tensor.getData(), tensor.getShape_d(), tensor.getStrides_d(),
        pooledData.getData(), pooledData.getShape_d(), pooledData.getStrides_d(),
        dc_dz.getData(),
        stride, paddingFrame, poolSize, poolMode, k,
        dc_dk.getData(), dc_dk.getShape_d(), dc_db.getData(), dc_db.getStrides_d());


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error after kernel convolutionChainRulePlusEqualCuda: %s\n", cudaGetErrorString(err));
        abort();
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Runtime error after kernel convolutionChainRulePlusEqualCuda sync: %s\n", cudaGetErrorString(err));
        abort();
    }
}
