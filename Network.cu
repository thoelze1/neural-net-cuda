/*
 * Tanner Hoelzel
 */

#include <cmath>
#include <iostream>

#include "Network.h"

__global__ void
hidden_layer(float *input, float *weights, float *output) {

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    weights = weights + id*28*28;

    float dp = 0;
    for (unsigned int i = 0; i < 28*28; i++) {
        dp += weights[i]*input[i];
    }
    output[id] = dp > 0? dp : 0; //ReLU
}

void
Network::train() {
    /*
    hidden_layer<<<1, 1024>>>(images, weights, hidden);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());
    */
}

void
Network::test() {

}

Network::~Network() {
    cudaFree(this->inputs);
    cudaFree(this->labels);
    cudaFree(this->weights1);
    cudaFree(this->weights2);
    cudaFree(this->scratch);
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
