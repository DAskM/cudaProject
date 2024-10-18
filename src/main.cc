#include <iostream>
#include <stdio.h>
#include "gpuInfoPrint.cuh"

int main(){
    getGpuInfo();
    std::cout << "------------------------------" << std::endl;
    return 0;
}