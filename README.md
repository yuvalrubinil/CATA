![CULA Logo](images/cula_logo.png)
# CULA - CUDA Linear Algebra Library
This is a lightweight C++ library for GPU-accelerated tensor computations.

## Features
* On-device and on-host data support
* Linear algebra primitives
* Convolutional neural network operations
* Please check: ops.cuh, tensor.cuh to see the full list of operations that are currently supported

# About
I'm a cs student with a strong interest in the field of deep learning. During my own private projects I have faced many preformance issues when dealing with Convolutional Neural Networks or even with regualr ones. The performance bottlenecks led me into this project with two main goals:
1) Learn CUDA - so I can design and customize my own gpu accelerated operations.
2) Create code that can be used for more deep learning projects in the future.
 
# How to use
1. Extarct the files into your project directory under 'cula' dir.
2. Add to your project:  ops.cuh & tensor.cuh from 'cula'.
3. Add to your project: all the files from 'ops' & 'tensor' dirs.
4. Place all files under one 'cula' filter in the solution explorer (optional).
5. Include:
   ```cpp
    #include "cula/tensor.cuh"
    #include "cula/ops.cuh"
6. Have fun!

# Code Sample
```cpp
  #include <iostream>
  #include "cula/tensor.cuh"
  #include "cula/ops.cuh"
  
  int main() {
      std::vector<int> shape = { 3 };
      Tensor v(shape, 1.0f);  // Fill with 1.0
      Tensor u(shape, 2.0f);  // Fill with 2.0
      std::cout << v.to_string() << std::endl;
      std::cout << u.to_string() << std::endl;
      std::cout << dotCuda(u, v) << std::endl;
      return 0;
  }

