#include "cuda_runtime.h"
#include "../ops.cuh"
#include "../tensor.cuh"
#include <sstream>
#include <iomanip>

Tensor::Tensor(const int* shape, int ndim) {
	this->ndim = ndim;
	this->h_shape = new int[ndim];
	this->h_strides = new int[ndim];
	for (int i = ndim - 1; 0 <= i; --i) {
		h_strides[i] = this->size;
		this->size *= shape[i];
		h_shape[i] = shape[i];
	}
	cudaMalloc(&this->data, this->size * sizeof(float));
	copyArrayToDevice_int(&this->d_shape, h_shape, ndim);
	copyArrayToDevice_int(&this->d_strides, h_strides, ndim);
	this->on_device = new bool(true);
	this->is_view = false;
}

Tensor::Tensor(Tensor& origin, int* shape, int* strides, int ndim) {
	this->ndim = ndim;
	this->h_shape = shape;
	this->h_strides = strides;
	this->size = origin.size;
	this->data = origin.data;
	this->on_device = origin.on_device;
	if (this->onDevice()) {
		copyArrayToDevice_int(&this->d_shape, h_shape, ndim);
		copyArrayToDevice_int(&this->d_strides, h_strides, ndim);
	}
	this->is_view = true;
}

Tensor::Tensor(const std::vector<int>& shape) {
	this->ndim = (int)shape.size();
	this->h_shape = new int[ndim];
	this->h_strides = new int[ndim];
	for (int i = ndim - 1; 0 <= i; --i) {
		h_strides[i] = this->size;
		this->size *= shape[i];
		h_shape[i] = shape[i];
	}
	cudaMalloc(&this->data, this->size * sizeof(float));
	copyArrayToDevice_int(&this->d_shape, h_shape, ndim);
	copyArrayToDevice_int(&this->d_strides, h_strides, ndim);
	this->on_device = new bool(true);
	this->is_view = false;
}

Tensor::Tensor(const std::vector<float>& values, const std::vector<int>& shape) {
	this->ndim = (int)shape.size();
	this->h_shape = new int[ndim];
	this->h_strides = new int[ndim];
	for (int i = ndim - 1; 0 <= i; --i) {
		h_strides[i] = this->size;
		this->size *= shape[i];
		h_shape[i] = shape[i];
	}
	copyVectorToDevice_float(&this->data, values);
	copyArrayToDevice_int(&this->d_shape, h_shape, ndim);
	copyArrayToDevice_int(&this->d_strides, h_strides, ndim);
	this->on_device = new bool(true);
	this->is_view = false;
}

Tensor::Tensor(const std::vector<int>& shape, float initialValue) {
	this->ndim = (int)shape.size();
	this->h_shape = new int[ndim];
	this->h_strides = new int[ndim];
	for (int i = ndim - 1; 0 <= i; --i) {
		h_strides[i] = this->size;
		this->size *= shape[i];
		h_shape[i] = shape[i];
	}
	cudaMalloc(&this->data, this->size * sizeof(float));
	fillCuda(this->data, this->size, initialValue);
	copyArrayToDevice_int(&this->d_shape, h_shape, ndim);
	copyArrayToDevice_int(&this->d_strides, h_strides, ndim);
	this->on_device = new bool(true);
	this->is_view = false;
}

Tensor::Tensor(Tensor& origin, const std::vector<int>& shape) {
	this->ndim = (int)shape.size();
	this->h_shape = new int[ndim];
	this->h_strides = new int[ndim];
	for (int i = ndim - 1; 0 <= i; --i) {
		h_strides[i] = this->size;
		this->size *= shape[i];
		h_shape[i] = shape[i];
	}
	this->data = origin.data;
	this->on_device = origin.on_device;
	if (this->onDevice()) {
		copyArrayToDevice_int(&this->d_shape, h_shape, ndim);
		copyArrayToDevice_int(&this->d_strides, h_strides, ndim);
	}
	this->is_view = true;
}

Tensor::Tensor(Tensor&& other) noexcept {
	this->data = other.data;
	this->on_device = other.on_device;
	this->d_shape = other.d_shape;
	this->h_shape = other.h_shape;
	this->d_strides = other.d_strides;
	this->h_strides = other.h_strides;
	this->ndim = other.ndim;
	this->size = other.size;
	this->is_view = other.is_view;
	other.data = nullptr;
	other.on_device = nullptr;
	other.d_shape = nullptr;
	other.h_shape = nullptr;
	other.d_strides = nullptr;
	other.h_strides = nullptr;
	other.size = 0;
}

Tensor::~Tensor() {
	if (this->onDevice()) {
		if (!this->isView()) 
			cudaFree(this->data);
		cudaFree(this->d_strides);
		cudaFree(this->d_shape);
	}
	else if (!this->isView()) {
		delete[] this->data;
		delete this->on_device;
	}
	delete[] this->h_shape;
	delete[] this->h_strides;
	this->data = nullptr;
	this->on_device = nullptr;
	this->d_shape = nullptr;
	this->h_shape = nullptr;
	this->d_strides = nullptr;
	this->h_strides = nullptr;
}

void Tensor::free() {
	this->~Tensor();
}

bool Tensor::onDevice() const {
	if (!this->on_device)
		return false;
	return *this->on_device;
}

bool Tensor::isView() const {
	return this->is_view;
}

void Tensor::toDevice() {
	if (!this->onDevice()) {
		float* d_data = nullptr;
		cudaMalloc(&d_data, this->size * sizeof(float));
		cudaMemcpy(d_data, this->data, this->size * sizeof(float), cudaMemcpyHostToDevice);
		delete[] this->data;
		this->data = d_data;
		*this->on_device = true;
	}
	if (!this->d_shape) {
		cudaMalloc(&d_shape, this->ndim * sizeof(int));
		cudaMemcpy(d_shape, this->h_shape, this->ndim * sizeof(int), cudaMemcpyHostToDevice);
	}
	if (!this->d_strides) {
		cudaMalloc(&d_strides, this->ndim * sizeof(int));
		cudaMemcpy(d_strides, this->h_strides, this->ndim * sizeof(int), cudaMemcpyHostToDevice);
	}
}

void Tensor::toHost() {
	if (this->onDevice()) {
		float* h_data = new float[this->size];
		cudaMemcpy(h_data, this->data, this->size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(this->data);
		this->data = h_data;
		*this->on_device = false;
	}
	if (this->d_shape) {
		cudaFree(this->d_shape);
		this->d_shape = nullptr;
	}
	if (this->d_strides) {
		cudaFree(this->d_strides);
		this->d_strides = nullptr;
	}
}

float* Tensor::getData() const {
	return this->data;
}

int Tensor::getSize() const {
	return this->size;
}

int* Tensor::getShape() const {
	return this->h_shape;
}

int* Tensor::getShape_d() const {
	return this->d_shape;
}

int* Tensor::getStrides() const {
	return this->h_strides;
}

int* Tensor::getStrides_d() const {
	return this->d_strides;
}

int Tensor::getNdim() const {
	return this->ndim;
}

std::string Tensor::to_string() const {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(4);
	if (!this) {
		oss << "Tensor pointer is null\n";
		return oss.str();
	}

	// Shape and strides
	oss << "Tensor(shape: ";
	for (int i = 0; i < ndim; ++i) {
		oss << h_shape[i];
		if (i != ndim - 1) oss << " x ";
	}

	oss << ", strides: [";
	if (h_strides) {
		for (int i = 0; i < ndim; ++i) {
			oss << h_strides[i];
			if (i != ndim - 1) oss << ", ";
		}
	}
	else {
		oss << "null";
	}
	oss << "], on_device: " << (this->onDevice() ? "true" : "false") << ")\n";
	// Handle null data
	if (!data) {
		oss << "Data pointer is null\n";
		return oss.str();
	}

	// Copy entire data to host if on device
	float* host_data = nullptr;
	bool copied = false;
	if (this->onDevice()) {
		host_data = new float[size];
		cudaMemcpy(host_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
		copied = true;
	}
	else {
		host_data = data;
	}

	// Print all elements linearly
	oss << "Data: [ ";
	for (int i = 0; i < size; ++i) {
		oss << host_data[i] << " ";
	}
	oss << "]\n";

	if (copied) delete[] host_data;

	return oss.str();
}

void Tensor::operator+=(Tensor& v) {
	plusCuda(*this, v, *this);
}

Tensor* Tensor::T() {
	int* reversedShape = new int[ndim];
	int* reversedStrides = new int[ndim];
	for (int i = 0; i < ndim; ++i) {
		reversedShape[i] = this->h_shape[ndim-1-i];
		reversedStrides[i] = this->h_strides[ndim-1-i];
	}
	return new Tensor(*this, reversedShape, reversedStrides, ndim);
}


