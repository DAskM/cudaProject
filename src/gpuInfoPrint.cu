#include "device_launch_parameters.h"
#include <iostream>
#include <cuda_runtime_api.h>

#include "gpuInfoPrint.cuh"

void getGpuInfo(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0; i<deviceCount; i++){
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "total memory: " << devProp.totalGlobalMem /1024 /1024 << "MB" << std::endl;
        std::cout << "SM number: " << devProp.multiProcessorCount << std::endl;
        std::cout << "shared memoty size for each thread: " << devProp.sharedMemPerBlock << std::endl;
        std::cout << "max thread per block: " << devProp.maxThreadsPerBlock << std::endl; 
    }
    
}