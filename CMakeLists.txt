cmake_minimum_required(VERSION 3.9) # CMake 最低版本
project(cudaPrint LANGUAGES CXX CUDA) # 指定支持 C++ 和 CUDA 

# 设置构建类型
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# 设置语言标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)

# 设置编译后的程序存放位置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 添加 src 目录下的所有源文件
aux_source_directory(src SRC_LIST)

# 添加头文件路径
include_directories(include)

# 编译列表
add_executable(${PROJECT_NAME} ${SRC_LIST})