// #include "device_launch_parameters.h"
// #include <iostream>
// #include <cuda_runtime_api.h>

// int main()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);
//     for(int i=0;i<deviceCount;i++)
//     {
//         cudaDeviceProp devProp;
//         cudaGetDeviceProperties(&devProp, i);
//         std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
//         std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
//         std::cout << "SM的数量:" << devProp.multiProcessorCount << std::endl;
//         std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
//         std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
//         std::cout << "设备上一个线程块(Block)种可用的32位寄存器数量: " << devProp.regsPerBlock << std::endl;
//         std::cout << "每个EM的最大线程数:" << devProp.maxThreadsPerMultiProcessor << std::endl;
//         std::cout << "每个EM的最大线程束数:" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
//         std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
//         std::cout << "======================================================" << std::endl;     
//     }
//     return 0;
// }

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include "device_launch_parameters.h"

#define pi 3.1415926535
#define LENGTH 100 //signal sampling points 采样点数
int main()
{
    // data gen 生成测试信号
    float Data[LENGTH] = { 1,2,3,4 };
    float fs = 1000000.000;//sampling frequency
    float f0 = 200000.00;// signal frequency
    for (int i = 0; i < LENGTH; i++)
    {
        Data[i] = 1.35*cos(2 * pi*f0*i / fs);//signal gen,
    }

    cufftComplex *CompData = (cufftComplex*)malloc(LENGTH * sizeof(cufftComplex));//allocate memory for the data in host
    int i;
    for (i = 0; i < LENGTH; i++)
    {
        CompData[i].x = Data[i];
        CompData[i].y = 0;
    }

    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData, LENGTH * sizeof(cufftComplex));// allocate memory for the data in device
    cudaMemcpy(d_fftData, CompData, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);// copy data from host to device

    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LENGTH, CUFFT_C2C, 1);//declaration
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done
    cudaMemcpy(CompData, d_fftData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);// copy the result from device to host

    for (i = 0; i < LENGTH / 2; i++)
    {
        printf("i=%d\tf= %6.1fHz\tRealAmp=%3.1f\t", i, fs*i / LENGTH, CompData[i].x*2.0 / LENGTH);
        printf("ImagAmp=+%3.1fi", CompData[i].y*2.0 / LENGTH);
        printf("\n");
    }
    cufftDestroy(plan);
    free(CompData);
    cudaFree(d_fftData);

}