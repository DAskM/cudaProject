#!/bin/bash

# 运行 CMake 配置命令
cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc)

# 输出完成信息
echo "CMake configuration completed in the '$BUILD_DIR' directory."