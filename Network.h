/*
 * Tanner Hoelzel
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <random>
#include <iostream>

#include "assert.h"

#define N_NODES       1024

class Network {
public:
    template <int N, int M>
    Network(float (&inputs)[N][M], unsigned char (&labels)[N]);
    ~Network();
    float *inputs;
    char *labels;
    float *weights1;
    float *weights2;
    float *scratch;
private:
    template <int N>
    void
    random_weights(float (&weights)[N]);
};

template <int N, int M>
Network::Network(float (&inputs)[N][M], unsigned char (&labels)[N]) {

    float weights1[M*N_NODES];// = new float[M*N_NODES];
    float weights2[N_NODES*10];// = new float[N_NODES*10];

    random_weights(weights1);
    random_weights(weights2);

    cudaMalloc(&this->inputs, N*M*sizeof(float));
    cudaMalloc(&this->labels, N*sizeof(char));
    cudaMalloc(&this->weights1, M*N_NODES*sizeof(float));
    cudaMalloc(&this->weights2, N_NODES*10*sizeof(float));
    cudaMalloc(&this->scratch, N_NODES*sizeof(float));

    cudaMemcpy(this->labels, labels, 60000*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(this->inputs, inputs, 60000*28*28*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->weights1, weights1, 28*28*1024*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->weights2, weights2, 1024*10*sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Hi" << std::endl;
}

template <int N>
void
Network::random_weights(float (&weights)[N]) {
    std::default_random_engine eng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    for(unsigned int i = 0; i < N; i++) {
        weights[i] = dist(eng);
    }
}

#endif
