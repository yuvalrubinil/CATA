#pragma once
#include <cuda_runtime.h>
#include "tensor.cuh"


// --- Memory operations ---
void printDeviceProperties();
void fillCuda(float* u, int n, float value);
void fillTensorCuda(Tensor& u, float value);
void oneHotCuda(Tensor& u, int hotIdx);
void copyCuda(Tensor& u, Tensor& w);
void copyVectorToDevice_float(float** arr_d, const std::vector<float>& arr_h);
void copyVectorToDevice_int(int** arr_d, const std::vector<int>& arr_h);
void copyArrayToDevice_int(int** arr_d, int* arr_h, int n);
void copySpecificToHost_int(int* ptr_h, int* ptr_d, int index);
__global__ void divideByScaler(float* u, float* w, int n, float scaler);


// --- Vector operations ---
void plusCuda(Tensor& u, Tensor& v, Tensor& w); 
void minusCuda(Tensor& u, Tensor& v, Tensor& w);
void starCuda(Tensor& u, Tensor& v, Tensor& w);
void slashCuda(Tensor& u, Tensor& v, Tensor& w);
void starStarCuda(Tensor& u, Tensor& w, float pow);
void expCuda(Tensor& u, Tensor& w);
void logCuda(Tensor& u, Tensor& w, float base);
void sqrtCuda(Tensor& u, Tensor& w);
void addScalerCuda(Tensor& u, Tensor& w, float scaler);
void multByScalerCuda(Tensor& u, Tensor& w, float scaler);
void divideByScalerCuda(Tensor& u, Tensor& w, float scaler);
float dotCuda(Tensor& u, Tensor& v);
float sumCuda(Tensor& u);
float meanCuda(Tensor& u);
void outerProductCuda(Tensor& u, Tensor& v, Tensor& W);
void outerProductPlusEqualCuda(Tensor& u, Tensor& v, Tensor& W);
void subtractByScaleCuda(Tensor& u, Tensor& v, float scaler);
int argmaxSmallCuda(Tensor& u);
float maxCuda(Tensor& u);

// --- Matrix operations ---
void matmulCuda(Tensor& A, Tensor& B, Tensor& C);
void matvecCuda(Tensor& A, Tensor& u, Tensor& w);

// --- Activations ---
void sigmoidCuda(Tensor& u, Tensor& w);
void sigmoidChainRuleCuda(Tensor& u, Tensor& dc_da, Tensor& w);
void reluCuda(Tensor& u, Tensor& w, float alpha=0.1f);
void reluChainRuleCuda(Tensor& u, Tensor& dc_da, Tensor& w, float alpha = 0.1f);
void softmaxCuda(Tensor& u, Tensor& w);
void softmaxChainRuleCuda(Tensor& u, Tensor& dc_da, Tensor& w);

// --- Loss functions ---
float mseCuda(Tensor& y_predicted, Tensor& y_expected);
void mseDerCuda(Tensor& y_predicted, Tensor& y_expected, Tensor& w);
float cceCuda(Tensor& y_predicted, Tensor& y_expected);
void cceDerCuda(Tensor& y_predicted, Tensor& y_expected, Tensor& w);

// --- Convolution ---
void convolutionCuda(Tensor& tensor, Tensor& kernels, Tensor& featureMap, int paddingFrame, int stride);
void poolingCuda(Tensor& featureMap, Tensor& pooledIndices, Tensor& result, char mode, int poolSize);
void convolutionChainRuleCuda(Tensor& tensor, Tensor& pooledData, Tensor& dc_dz, Tensor& dc_db, Tensor& dc_dk,
    int paddingFrame, int stride, int kernelWidth, char poolMode, int poolSize, int k);
void convolutionChainRulePlusEqualCuda(Tensor& tensor, Tensor& pooledData, Tensor& dc_dz, Tensor& dc_db, Tensor& dc_dk,
    int paddingFrame, int stride, int kernelWidth, char poolMode, int poolSize, int k);

// --- Weight init ---
void normalInitCuda(Tensor& u, float mean = 0.0f, float std = 1.0f);
void xavierInitCuda(Tensor& u, int fan_in, int fan_out);
void heNormalInitCuda(Tensor& u, int fan_in);
void heUniformInitCuda(Tensor& u, int fan_in);
void zerosInitCuda(Tensor& u);
void initWeights(Tensor& weights, const std::string& method, int fan_in = 0, int fan_out = 0);

// --- Optimizers ---
void momentumCuda(Tensor& v, Tensor& newV, float beta);
void accumulatedSquersCuda(Tensor& s, Tensor& newV);
void adagradSubtractionCuda(Tensor& w, Tensor& s, Tensor& newV, float learningRate);
void momentumSqueredCuda(Tensor& v, Tensor& newV, float beta);
void rmsPropSubtractionCuda(Tensor& w, Tensor& s, Tensor& newV, float learningRate);
void adamSubtractionCuda(Tensor& w, Tensor& s, Tensor& v, float learningRate, float beta1, float beta2, int t);