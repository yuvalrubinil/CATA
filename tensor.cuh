#pragma once
#include <vector>
#include <string>
#include <sstream>


class Tensor {
private:
	float* data = nullptr;
	bool* on_device = nullptr;
	int* d_shape = nullptr;
	int* h_shape = nullptr;
	int size = 1;
	int ndim = 0;
	int* d_strides = nullptr;
	int* h_strides = nullptr;
	bool is_view = false;

public:
	Tensor(const int* shape, int ndim);
	Tensor(Tensor& origin, int* shape, int* strides, int ndim);
	Tensor(const std::vector<int>& shape);
	Tensor(const std::vector<float>& values, const std::vector<int>& shape);
	Tensor(const std::vector<int>& shape, float initialValue);
	Tensor(Tensor& origin, const std::vector<int>& shape); // for reshaping to view.
	Tensor(Tensor&& other) noexcept;
	~Tensor();

	void free();
	bool onDevice() const;
	bool isView() const;
	void toDevice();
	void toHost();
	float* getData() const;
	int getSize() const;
	int* getShape() const;
	int* getShape_d() const;
	int* getStrides() const;
	int* getStrides_d() const;
	int getNdim() const;
	Tensor* T();
	std::string to_string() const;
	void operator+=(Tensor& v);
};