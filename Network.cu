/*
 * Tanner Hoelzel
 */

#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

#include "Network.h"

#define N_NODES       1024
#define BATCH_SIZE     100
#define RATE          0.01

Network::Network(float *inputs, unsigned char *labels) {

    float weights1[28*28*N_NODES];
    float weights2[N_NODES*10];

    this->eng = new std::default_random_engine(std::random_device{}());

    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    for(unsigned int i = 0; i < 28*28*N_NODES; i++) {
        weights1[i] = dist(*(this->eng));
    }
    for(unsigned int i = 0; i < N_NODES*10; i++) {
        weights2[i] = dist(*(this->eng));
    }

    this->host_labels = labels;
    cudaMalloc(&this->inputs, 28*28*60000*sizeof(float));
    cudaMalloc(&this->weights1, 28*28*N_NODES*sizeof(float));
    cudaMalloc(&this->outputs, N_NODES*sizeof(float));
    cudaMalloc(&this->weights2, N_NODES*10*sizeof(float));
    cudaMalloc(&this->classes, 10*sizeof(float));
    cudaMalloc(&this->softmax, 10*sizeof(float));
    cudaMalloc(&this->softmax_ds, 10*sizeof(float));
    cudaMalloc(&this->hidden_ds, 1024*sizeof(float));
    cudaMalloc(&this->weights1_grad, 28*28*1024*sizeof(float));
    cudaMalloc(&this->weights2_grad, 1024*10*sizeof(float));

    cudaMemcpy(this->inputs, inputs, 60000*28*28*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->weights1, weights1, 28*28*1024*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->weights2, weights2, 1024*10*sizeof(float), cudaMemcpyHostToDevice);
}

Network::~Network() {
    cudaFree(this->inputs);
    cudaFree(this->weights1);
    cudaFree(this->outputs);
    cudaFree(this->weights2);
    cudaFree(this->classes);
    cudaFree(this->softmax);
    delete this->eng;
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

__global__ void
softmax_back(float *sm_out, unsigned char label, float *sm_ds) {

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

    float us = -1/sm_out[label];
    if (id == (unsigned int)label) {
        sm_ds[id] = (sm_out[label]*(1 - sm_out[id]))*us;
    } else {
        sm_ds[id] = (-1*sm_out[id]*sm_out[label])*us;
    }
}

__global__ void
hidden_forward(float *input, unsigned int input_size, float *weights, float *output, unsigned int output_size, bool relu) {

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    weights = weights + id*input_size;

    float dp = 0;
    for (unsigned int i = 0; i < input_size; i++) {
        dp += weights[i]*input[i];
    }
    output[id] = (!relu || dp > 0)? dp : 0;
}

//TODO use multiple blocks to make this faster
__global__ void
hidden_back(float *hidden_out, float *sm_ds, float *hidden_ds, float *weights2, float *weights2_grad, float *layer) {

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

    for(unsigned int i = 0; i < 10; i++) {
        hidden_ds[id] += sm_ds[i]*weights2[id*10+i];
        weights2_grad[id*10+i] += sm_ds[i]*layer[id];
    }
}

void
Network::run(unsigned int i) {

    hidden_forward<<<1, 1024>>>(this->inputs+i*28*28, 28*28, this->weights1, this->outputs, 1024, true);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());

    hidden_forward<<<1, 10>>>(this->outputs, 1024, this->weights2, this->classes, 10, false);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());

    softmax_forward<<<1, 1>>>(this->classes, this->softmax, 10);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());

    softmax_back<<<1, 10>>>(this->softmax, this->host_labels[i], this->softmax_ds);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());

    hidden_back<<<1, 1024>>>(this->classes, this->softmax_ds, this->hidden_ds, this->weights2, this->weights2_grad, this->outputs);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());
}

void
Network::update() {

}

void
Network::train() {
    std::vector<unsigned int> indices(60000);
    for(unsigned int i = 0; i < 60000; i++) {
        indices[i] = i;
    }
    std::shuffle(std::begin(indices), std::end(indices), *(this->eng));
    /*
    for(unsigned int i = 0; i < (60000/BATCH_SIZE); i++) {
        cudaMemset(this->weights1_grad, 0, 28*28*1024*sizeof(float))
        cudaMemset(this->weights2_grad, 0, 28*28*1024*sizeof(float))
        for(unsigned int j = 0; j < BATCH_SIZE; j++) {
            run(indices[i*BATCH_SIZE+j]);
        }
        update();
    }
    */
    run(0);
}

float
Network::test(float *tests, unsigned char *labels) {

    float *d_tests;
    cudaMalloc(&d_tests, 28*28*10000*sizeof(float));
    cudaMemcpy(d_tests, tests, 28*28*10000*sizeof(float), cudaMemcpyHostToDevice);

    unsigned int acc = 0;
    //@TODO change back to 10000
    for(unsigned int i = 0; i < 1000; i++) {
        hidden_forward<<<1, 1024>>>(d_tests + i*28*28, 28*28, this->weights1, this->outputs, 1024, true);
        gpu_assert(cudaPeekAtLastError());
        gpu_assert(cudaDeviceSynchronize());
        hidden_forward<<<1, 10>>>(this->outputs, 1024, this->weights2, this->classes, 10, false);
        gpu_assert(cudaPeekAtLastError());
        gpu_assert(cudaDeviceSynchronize());
        float mem[10];
        cudaMemcpy(mem, this->classes, 10*sizeof(float), cudaMemcpyDeviceToHost);
        float max = -100000;
        unsigned int max_j = 0;
        for(unsigned int j = 0; j < 10; j++) {
            if(mem[i] > max) {
                max = mem[j];
                max_j = j;
            }
        }
        if(((int)labels[i]) == max_j) acc += 1;
    }
    return (float)acc/1000;
}

