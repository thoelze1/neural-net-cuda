/*
 * Tanner Hoelzel
 */

#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

#include "Network.h"

#define N_NODES       1024
#define BATCH_SIZE     600
#define RATE         0.002
#define DO_RATE        0.4
#define IMG_SIZE     28*28

Network::Network(float *inputs, unsigned char *labels) {

    float input_w[IMG_SIZE*N_NODES];
    float hidden_w[N_NODES*10];

    this->eng = new std::default_random_engine(std::random_device{}());

    std::normal_distribution<float> dist;
    for(unsigned int i = 0; i < IMG_SIZE*N_NODES; i++) {
        input_w[i] = dist(*(this->eng))/sqrt(IMG_SIZE);
    }
    for(unsigned int i = 0; i < N_NODES*10; i++) {
        hidden_w[i] = dist(*(this->eng))/sqrt(N_NODES);
    }

    this->host_labels = labels;
    cudaMalloc(&this->input_l, IMG_SIZE*60000*sizeof(float));
    cudaMalloc(&this->input_w, IMG_SIZE*N_NODES*sizeof(float));
    cudaMalloc(&this->input_w_grad, IMG_SIZE*N_NODES*sizeof(float));
    cudaMalloc(&this->input_bias, N_NODES*sizeof(float));
    cudaMalloc(&this->input_bias_grad, N_NODES*sizeof(float));
    cudaMalloc(&this->hidden_l, N_NODES*sizeof(float));
    cudaMalloc(&this->hidden_w, N_NODES*10*sizeof(float));
    cudaMalloc(&this->hidden_w_grad, N_NODES*10*sizeof(float));
    cudaMalloc(&this->hidden_bias, 10*sizeof(float));
    cudaMalloc(&this->hidden_bias_grad, 10*sizeof(float));
    cudaMalloc(&this->dropouts, N_NODES*sizeof(float));
    cudaMalloc(&this->output_l, 10*sizeof(float));
    cudaMalloc(&this->softmax_l, 10*sizeof(float));
    cudaMalloc(&this->softmax_ds, 10*sizeof(float));
    cudaMalloc(&this->hidden_ds, N_NODES*sizeof(float));

    cudaMemset(this->input_bias, 0, N_NODES*sizeof(float));
    cudaMemset(this->hidden_bias, 0, 10*sizeof(float));

    cudaMemcpy(this->input_l, inputs, 60000*IMG_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->input_w, input_w, IMG_SIZE*N_NODES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->hidden_w, hidden_w, N_NODES*10*sizeof(float), cudaMemcpyHostToDevice);
}

Network::~Network() {
    cudaFree(this->input_l);
    cudaFree(this->input_w);
    cudaFree(this->input_w_grad);
    cudaFree(this->input_bias);
    cudaFree(this->input_bias_grad);
    cudaFree(this->hidden_l);
    cudaFree(this->hidden_w);
    cudaFree(this->hidden_w_grad);
    cudaFree(this->hidden_bias);
    cudaFree(this->hidden_bias_grad);
    cudaFree(this->output_l);
    cudaFree(this->softmax_l);
    cudaFree(this->softmax_ds);
    cudaFree(this->hidden_ds);
    delete this->eng;
}

__global__ void
softmax_forward(float *input, float *output, unsigned int n) {
    unsigned int i;
    float max = input[0];
    for(i = 1; i < n; i++) {
        if(input[i] > max) max = input[i];
    }
    float sum = 0;
    for(i = 0; i < n; i++) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }
    for(i = 0; i < n; i++) {
        output[i] = output[i]/sum;
    }
}

__global__ void
softmax_back(float *softmax_l, float *softmax_ds, unsigned char label) {

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

    float us = -1/softmax_l[(unsigned int)label];
    if (id == (unsigned int)label) {
        softmax_ds[id] = (softmax_l[(unsigned int)label]*(1 - softmax_l[id]))*us;
    } else {
        softmax_ds[id] = -1*softmax_l[id]*softmax_l[(unsigned int)label]*us;
    }
}

__global__ void
hidden_forward(float *input, unsigned int input_size, float *weights, float *output, unsigned int output_size,
               float *bias, bool relu, float *dropouts) {

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    float dp = 0;
    for (unsigned int i = 0; i < input_size; i++) {
        dp += weights[id*input_size+i]*input[i];
    }
    dp += bias[id];
    if(dropouts) {
        dp *= dropouts[id];
    }
    output[id] = (!relu || (dp > 0))? dp : 0;
}

//TODO use multiple blocks to make this faster
__global__ void
hidden_back(float *input, unsigned int input_size, float *output, unsigned int output_size,
            float *us, float *ds, float *weights, float *weights_grad, float *bias, float *bias_grad,
            bool relu) {

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

    for(unsigned int i = 0; i < output_size; i++) {
        if(!relu || output[i] > 0) {
            if(ds) {
                ds[id] += us[i]*weights[id*output_size+i];
            }
            weights_grad[id*output_size+i] += (us[i]*input[id]/BATCH_SIZE);
        }
        if(id == 0) {
            bias_grad[i] += us[i]/BATCH_SIZE;
        }
    }
}

__global__ void
update_weights(float *weights, float *weights_grad) {
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    weights[id] -= weights_grad[id]*RATE;
}

void
Network::train(unsigned int i) {

    float dropouts[N_NODES];
    std::uniform_real_distribution<float> dist;
    for(unsigned int i = 0; i < N_NODES; i++) {
        if(dist(*this->eng) < DO_RATE) dropouts[i] = 0;
        else dropouts[i] = 1/DO_RATE;
    }
    cudaMemcpy(this->dropouts, dropouts, N_NODES*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(this->hidden_ds, 0, N_NODES*sizeof(float));

    hidden_forward<<<1, N_NODES>>>(this->input_l+i*IMG_SIZE, IMG_SIZE, this->input_w, this->hidden_l, N_NODES,
                                this->input_bias, true, 0);
    hidden_forward<<<1, 10>>>(this->hidden_l, N_NODES, this->hidden_w, this->output_l, 10,
                              this->hidden_bias, false, this->dropouts);
    softmax_forward<<<1, 1>>>(this->output_l, this->softmax_l, 10);
    softmax_back<<<1, 10>>>(this->softmax_l, this->softmax_ds, this->host_labels[i]);
    hidden_back<<<1, N_NODES>>>(this->hidden_l, N_NODES, this->output_l, 10,
                             this->softmax_ds, this->hidden_ds, this->hidden_w, this->hidden_w_grad,
                             this->hidden_bias, this->hidden_bias_grad, false);
    hidden_back<<<1, IMG_SIZE>>>(this->input_l+i*IMG_SIZE, IMG_SIZE, this->hidden_l, N_NODES,
                              this->hidden_ds, 0, this->input_w, this->input_w_grad,
                              this->input_bias, this->input_bias_grad, true);

    float mem[10];
    std::cout << (unsigned int)this->host_labels[i] << std::endl;
    cudaMemcpy(mem, this->softmax_l, 10*sizeof(float), cudaMemcpyDeviceToHost);
    for(unsigned int j = 0; j < 10; j++) {
        std::cout << mem[j] << " ";
    }
    std::cout << std::endl;
}

void
Network::train() {
    std::vector<unsigned int> indices(60000);
    for(unsigned int i = 0; i < 60000; i++) {
        indices[i] = i;
    }
    std::shuffle(std::begin(indices), std::end(indices), *(this->eng));
    for(unsigned int i = 0; i < (60000/BATCH_SIZE); i++) {
        std::cout << "Batch " << i << std::endl;
        cudaMemset(this->input_w_grad, 0, IMG_SIZE*N_NODES*sizeof(float));
        cudaMemset(this->input_bias_grad, 0, N_NODES*sizeof(float));
        cudaMemset(this->hidden_w_grad, 0, N_NODES*10*sizeof(float));
        cudaMemset(this->hidden_bias_grad, 0, 10*sizeof(float));
        for(unsigned int j = 0; j < BATCH_SIZE; j++) {
            train(indices[i*BATCH_SIZE+j]);
        }
        update_weights<<<IMG_SIZE, N_NODES>>>(this->input_w, this->input_w_grad);
        update_weights<<<N_NODES, 10>>>(this->hidden_w, this->hidden_w_grad);
        update_weights<<<1, N_NODES>>>(this->input_bias, this->input_bias_grad);
        update_weights<<<1, 10>>>(this->hidden_bias, this->hidden_bias_grad);
    }
}

float
Network::test(float *tests, unsigned char *labels) {

    float *d_tests;
    cudaMalloc(&d_tests, IMG_SIZE*10000*sizeof(float));
    cudaMemcpy(d_tests, tests, IMG_SIZE*10000*sizeof(float), cudaMemcpyHostToDevice);

    unsigned int acc = 0;
    for(unsigned int i = 0; i < 1000; i++) {
        hidden_forward<<<1, N_NODES>>>(d_tests + i*IMG_SIZE, IMG_SIZE, this->input_w, this->hidden_l, N_NODES, this->input_bias, true, 0);
        hidden_forward<<<1, 10>>>(this->hidden_l, N_NODES, this->hidden_w, this->output_l, 10, this->hidden_bias, false, 0);
        float mem[10];
        cudaMemcpy(mem, this->output_l, 10*sizeof(float), cudaMemcpyDeviceToHost);
        float max = mem[0];
        unsigned int max_j = 0;
        for(unsigned int j = 0; j < 10; j++) {
            if(mem[j] > max) {
                max = mem[j];
                max_j = j;
            }
        }
        if(((unsigned int)labels[i]) == max_j) acc += 1;
    }
    return (float)acc/1000;
}
