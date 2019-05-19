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

/*
void
softmax_layer(float *input, float *output) {
    for(int class = 0; class < 10; class++) {
        float max = 0;
        for(int i = 0; i < 1024; i++) {
            if(input[i] > max) max = input[i];
        }
        float sum = 0;
        for(int i = 0; i < 1024; i++) {
            output[i] = std::exp(in[i] - C);
        }
    }
}

    T sum = 0;
    for (size_t i = 0; i < N; i++) {
        out[i] = std::exp(in[i] - C);
        sum += out[i];
    }
    std::transform(out.begin(), out.end(), out.begin(), [sum](auto e) { return e/sum; });

    // Verify that it is a probability: Sums to 1 and all >= 0.
    assert(approx_equal(std::accumulate(out.begin(), out.end(), T(0)), 1));
    #ifndef NDEBUG
    std::for_each(out.begin(), out.end(), [](auto e) { assert(e >= 0); });
    #endif

    return out;
}
*/

void
Network::train() {
    forward<<<1, 1024>>>(this->inputs, 28*28, this->weights1, this->outputs, 1024, true);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());
    forward<<<1, 10>>>(this->outputs, 1024, this->weights1, this->classes, 10, false);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());
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
