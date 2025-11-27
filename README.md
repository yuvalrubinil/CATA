


# CULA
CULA - CUDA Linear Algebra and Tensor Operations Lightweight C++ library for GPU-accelerated computations, supporting on-device and on-host data, linear algebra primitives, and convolutional neural network operations.

# About
I'm a cs student with a strong interest in the field of deep learning. During my own private projects I have faced many preformance issues when dealing with Convolutional Neural Networks or even with regualr ones. The performance bottlenecks led me into this project with two main goals:
1) Learn CUDA - so I can design and customize my own gpu accelerated operations.
2) Create code that can be used for more deep learning projects in the future.
 

# How to use
1. Download the zip.
2. Extarct files into your project directory (I recommand placing all of them under a 'cula' folder).
3. Include:
    #include "cula/tensor.cuh"
    #include "cula/ops.cuh"
4. Have fun!

*To see the operations that are currently supported, please check: ops.cuh, tensor.cuh

# Code Sample
  #include <iostream>
  #include "include/cula/tensor.cuh"
  #include "include/cula/ops.cuh"
  
  int main() {
      std::vector<int> shape = { 2, 2 };
      Tensor v(shape, 1.0f);  // Fill with 1.0
      Tensor u(shape, 2.0f);  // Fill with 2.0
      std::cout << v.to_string() << std::endl;
      std::cout << u.to_string() << std::endl;
      std::cout << dotCuda(u, v) << std::endl;
      return 0;
  }
