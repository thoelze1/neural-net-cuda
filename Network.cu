/*
 * Tanner Hoelzel
 */

#include <cmath>
#include <iostream>

#include "Network.h"

__global__ void
forward(float *input, unsigned int input_size, float *weights, float *output, unsigned int output_size, bool relu) {

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    weights = weights + id*input_size;

    float dp = 0;
    for (unsigned int i = 0; i < input_size; i++) {
        dp += weights[i]*input[i];
    }
    output[id] = (!relu || dp > 0)? dp : 0;
}

__global__ void
softmax_forward(float *input, float *output, unsigned int n) {
    unsigned int i;
    float max = 0;
    for(i = 0; i < n; i++) {
        if(input[i] > max) max = input[i];
    }
    float sum = 0;
    for(i = 0; i < n; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    for(i = 0; i < n; i++) {
        output[i] = output[i]/sum;
    }
}

void
Network::train() {
    forward<<<1, 1024>>>(this->inputs, 28*28, this->weights1, this->outputs, 1024, true);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());
    forward<<<1, 10>>>(this->outputs, 1024, this->weights2, this->classes, 10, false);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());
    softmax_forward<<<1, 1>>>(this->classes, this->softmax, 10);
    /*
    float mem[10];
    cudaMemcpy(mem, this->softmax, 10*sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for(unsigned int i = 0; i < 10; i++) {
        sum += mem[i];
    }
    std::cout << sum << std::endl;
    */
}

void
Network::test() {

}

Network::~Network() {
    cudaFree(this->labels);
    cudaFree(this->inputs);
    cudaFree(this->weights1);
    cudaFree(this->outputs);
    cudaFree(this->weights2);
    cudaFree(this->classes);
    cudaFree(this->softmax);
}
